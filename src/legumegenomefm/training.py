from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from legumegenomefm.masking import mask_span_mlm
from legumegenomefm.model import LegumeGenomeConfig, LegumeGenomeModel
from legumegenomefm.training_data import GenomeWindowSampler


@dataclass(frozen=True)
class TrainConfig:
    dataset_manifest: Path
    project_root: Path
    output_dir: Path
    context_length: int
    micro_batch_size: int
    global_batch_tokens: int
    max_tokens: int
    warmup_tokens: int
    learning_rate: float
    min_lr_ratio: float
    weight_decay: float
    gradient_clip: float
    seed: int
    precision: str
    checkpoint_every_steps: int
    log_every_steps: int
    model: dict[str, Any]
    mask_probability: float = 0.15
    mean_mask_span: float = 3.0


@dataclass(frozen=True)
class TrainingResult:
    step: int
    tokens_seen: int
    global_microstep: int
    loss: float
    checkpoint_dir: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _implementation_hash() -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"legumegenomefm-training-closure-v1\0")
    for name in (
        "masking.py",
        "model.py",
        "sequence_store.py",
        "tokenizer.py",
        "training.py",
        "training_data.py",
    ):
        digest.update(name.encode("ascii") + b"\0")
        digest.update((package_root / name).read_bytes())
    return digest.hexdigest()


def _canonical_config(config: TrainConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["dataset_manifest"] = str(Path(config.dataset_manifest).resolve())
    payload["project_root"] = str(Path(config.project_root).resolve())
    payload["output_dir"] = str(Path(config.output_dir).resolve())
    payload["dataset_manifest_sha256"] = _sha256(Path(config.dataset_manifest))
    payload["implementation_sha256"] = _implementation_hash()
    return payload


def _config_hash(config: TrainConfig) -> str:
    encoded = json.dumps(_canonical_config(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_config(values: dict[str, Any]) -> LegumeGenomeConfig:
    normalized = dict(values)
    if "dilations" in normalized:
        normalized["dilations"] = tuple(int(value) for value in normalized["dilations"])
    return LegumeGenomeConfig(**normalized)


def _learning_rate(config: TrainConfig, tokens_after_step: int) -> float:
    if config.warmup_tokens > 0 and tokens_after_step <= config.warmup_tokens:
        return config.learning_rate * tokens_after_step / config.warmup_tokens
    decay_denominator = max(1, config.max_tokens - config.warmup_tokens)
    progress = min(1.0, max(0.0, (tokens_after_step - config.warmup_tokens) / decay_denominator))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.learning_rate * (config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine)


def _validate_config(config: TrainConfig, world_size: int) -> int:
    if config.context_length < 1 or config.micro_batch_size < 1:
        raise ValueError("context length and micro batch size must be positive")
    tokens_per_microstep = config.context_length * config.micro_batch_size * world_size
    if config.global_batch_tokens % tokens_per_microstep:
        raise ValueError("global_batch_tokens must be divisible by context_length * micro_batch_size * world_size")
    if config.max_tokens < config.global_batch_tokens or config.max_tokens % config.global_batch_tokens:
        raise ValueError("max_tokens must be a positive multiple of global_batch_tokens")
    if config.warmup_tokens < 0 or config.warmup_tokens > config.max_tokens:
        raise ValueError("invalid warmup token budget")
    if config.learning_rate <= 0 or not 0 <= config.min_lr_ratio <= 1:
        raise ValueError("invalid learning-rate contract")
    if config.precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    if config.checkpoint_every_steps < 1 or config.log_every_steps < 1:
        raise ValueError("checkpoint and log intervals must be positive")
    return config.global_batch_tokens // tokens_per_microstep


def _resolve_device(device_override: str | None, local_rank: int) -> torch.device:
    if device_override == "cuda":
        return torch.device("cuda", local_rank)
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _checkpoint_valid(path: Path, expected_config_hash: str) -> tuple[bool, dict[str, object] | None]:
    try:
        receipt_bytes = (path / "receipt.json").read_bytes()
        ready = (path / "READY").read_text(encoding="ascii").strip()
        if hashlib.sha256(receipt_bytes).hexdigest() != ready:
            return False, None
        receipt = json.loads(receipt_bytes)
        if receipt.get("config_sha256") != expected_config_hash:
            return False, None
        if _sha256(path / "state.pt") != receipt.get("state_sha256"):
            return False, None
        return True, receipt
    except (OSError, ValueError, json.JSONDecodeError):
        return False, None


def _checkpoint_integrity_valid(path: Path) -> bool:
    try:
        receipt_bytes = (path / "receipt.json").read_bytes()
        ready = (path / "READY").read_text(encoding="ascii").strip()
        if hashlib.sha256(receipt_bytes).hexdigest() != ready:
            return False
        receipt = json.loads(receipt_bytes)
        return _sha256(path / "state.pt") == receipt.get("state_sha256")
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _latest_checkpoint(root: Path, expected_config_hash: str) -> Path | None:
    if not root.is_dir():
        return None
    valid: list[tuple[int, Path]] = []
    for path in root.glob("step_*" ):
        try:
            step = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        is_valid, _ = _checkpoint_valid(path, expected_config_hash)
        if is_valid:
            valid.append((step, path))
    return max(valid, default=(0, None), key=lambda item: item[0])[1]


def _save_checkpoint(
    root: Path,
    *,
    model: LegumeGenomeModel,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    tokens_seen: int,
    global_microstep: int,
    loss: float,
    config_hash: str,
) -> Path:
    implementation_sha256 = _implementation_hash()
    target = root / f"step_{step:08d}"
    valid, _ = _checkpoint_valid(target, config_hash) if target.exists() else (False, None)
    if valid:
        return target
    if target.exists():
        shutil.rmtree(target)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=root))
    try:
        state_path = staging / "state.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "step": step,
                "tokens_seen": tokens_seen,
                "global_microstep": global_microstep,
                "loss": loss,
                "config_sha256": config_hash,
                "implementation_sha256": implementation_sha256,
            },
            state_path,
        )
        state_sha = _sha256(state_path)
        receipt = {
            "schema_version": "1.0",
            "state": "READY",
            "step": step,
            "tokens_seen": tokens_seen,
            "global_microstep": global_microstep,
            "loss": loss,
            "config_sha256": config_hash,
            "implementation_sha256": implementation_sha256,
            "state_sha256": state_sha,
        }
        receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        (staging / "receipt.json").write_bytes(receipt_bytes)
        (staging / "READY").write_text(hashlib.sha256(receipt_bytes).hexdigest() + "\n", encoding="ascii")
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _mask_generator(seed: int, microstep: int, rank: int) -> torch.Generator:
    material = f"mask\0{seed}\0{microstep}\0{rank}".encode("ascii")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
    return torch.Generator().manual_seed(derived)


def run_training(
    config: TrainConfig,
    *,
    device_override: str | None = None,
    resume: bool = False,
    initialize_checkpoint: Path | None = None,
    stop_after_steps: int | None = None,
) -> TrainingResult:
    if resume and initialize_checkpoint is not None:
        raise ValueError("resume and initialize_checkpoint are mutually exclusive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and device_override != "cpu" else "gloo"
        dist.init_process_group(backend=backend)
        initialized_here = True
    accumulation_steps = _validate_config(config, world_size)
    device = _resolve_device(device_override, local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if device.type == "cpu" and config.precision != "fp32":
        raise ValueError("CPU training requires fp32 precision")
    if device.type == "cuda" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support bf16")
    memory_fraction = os.environ.get("LEGUME_GPU_MEMORY_FRACTION")
    if device.type == "cuda" and memory_fraction is not None:
        fraction = float(memory_fraction)
        if not 0 < fraction <= 1:
            raise ValueError("LEGUME_GPU_MEMORY_FRACTION must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(fraction, device.index or 0)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model_config = _model_config(config.model)
    model = LegumeGenomeModel(model_config).to(device)
    model.set_gradient_checkpointing(True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.precision == "fp16")
    config_hash = _config_hash(config)
    step = 0
    tokens_seen = 0
    global_microstep = 0
    last_loss = float("nan")
    checkpoint_root = Path(config.output_dir) / "checkpoints"
    checkpoint = _latest_checkpoint(checkpoint_root, config_hash) if resume else None
    if resume and checkpoint is None:
        raise ValueError("resume requested but no valid compatible checkpoint exists")
    if checkpoint is not None:
        state = torch.load(checkpoint / "state.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        step = int(state["step"])
        tokens_seen = int(state["tokens_seen"])
        global_microstep = int(state["global_microstep"])
        last_loss = float(state["loss"])
    elif initialize_checkpoint is not None:
        initialization_path = Path(initialize_checkpoint)
        if not _checkpoint_integrity_valid(initialization_path):
            raise ValueError("initialization checkpoint failed integrity validation")
        state = torch.load(initialization_path / "state.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)

    train_model: torch.nn.Module = model
    if world_size > 1:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    sampler = GenomeWindowSampler(
        config.dataset_manifest,
        config.project_root,
        context_length=config.context_length,
        split="pretrain",
        seed=config.seed,
    )
    target_steps = config.max_tokens // config.global_batch_tokens
    if stop_after_steps is not None:
        target_steps = min(target_steps, stop_after_steps)
    log_path = Path(config.output_dir) / "metrics.jsonl"
    tokens_at_start = tokens_seen
    started = time.monotonic()
    consecutive_overflows = 0
    try:
        while step < target_steps:
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for accumulation_index in range(accumulation_steps):
                tokens = sampler.sample_batch(
                    batch_size=config.micro_batch_size,
                    global_microstep=global_microstep,
                    rank=rank,
                )
                inputs, labels = mask_span_mlm(
                    tokens,
                    _mask_generator(config.seed, global_microstep, rank),
                    mask_probability=config.mask_probability,
                    mean_span_length=config.mean_mask_span,
                )
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                synchronize = accumulation_index == accumulation_steps - 1
                sync_context = (
                    train_model.no_sync() if world_size > 1 and not synchronize else nullcontext()
                )
                if device.type == "cuda" and config.precision in {"fp16", "bf16"}:
                    dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
                    autocast_context = torch.autocast(device_type="cuda", dtype=dtype)
                else:
                    autocast_context = nullcontext()
                with sync_context, autocast_context:
                    output = train_model(inputs, labels=labels)
                    if output.loss is None:
                        raise RuntimeError("training model returned no loss")
                    loss = output.loss / accumulation_steps
                scaler.scale(loss).backward()
                accumulated_loss += float(output.loss.detach().float().item())
                global_microstep += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            tokens_after_step = tokens_seen + config.global_batch_tokens
            lr = _learning_rate(config, tokens_after_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            overflow_skipped = scaler.is_enabled() and scaler.get_scale() < scale_before_step
            if world_size > 1:
                overflow_count = torch.tensor(int(overflow_skipped), device=device)
                dist.all_reduce(overflow_count, op=dist.ReduceOp.SUM)
                if int(overflow_count.item()) not in {0, world_size}:
                    raise RuntimeError("gradient-overflow decision diverged across DDP ranks")
            if overflow_skipped:
                consecutive_overflows += 1
                if rank == 0:
                    _append_jsonl(
                        log_path,
                        {
                            "event": "gradient_overflow_retry",
                            "global_microstep": global_microstep,
                            "scale_before": scale_before_step,
                            "scale_after": scaler.get_scale(),
                            "consecutive_overflows": consecutive_overflows,
                            "world_size": world_size,
                        },
                    )
                if world_size > 1:
                    dist.barrier()
                if consecutive_overflows > 8:
                    raise RuntimeError("more than 8 consecutive gradient-overflow retries")
                continue
            consecutive_overflows = 0
            step += 1
            tokens_seen = tokens_after_step
            local_loss = torch.tensor(accumulated_loss / accumulation_steps, device=device)
            if world_size > 1:
                dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                local_loss /= world_size
            last_loss = float(local_loss.item())
            if rank == 0 and (step % config.log_every_steps == 0 or step == target_steps):
                elapsed = max(time.monotonic() - started, 1e-9)
                _append_jsonl(
                    log_path,
                    {
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "loss": last_loss,
                        "learning_rate": lr,
                        "tokens_per_second_since_start": (tokens_seen - tokens_at_start) / elapsed,
                        "world_size": world_size,
                    },
                )
            if rank == 0 and step % config.checkpoint_every_steps == 0:
                _save_checkpoint(
                    checkpoint_root,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=step,
                    tokens_seen=tokens_seen,
                    global_microstep=global_microstep,
                    loss=last_loss,
                    config_hash=config_hash,
                )
            if world_size > 1:
                dist.barrier()
        if rank == 0:
            checkpoint = _save_checkpoint(
                checkpoint_root,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                tokens_seen=tokens_seen,
                global_microstep=global_microstep,
                loss=last_loss,
                config_hash=config_hash,
            )
        else:
            checkpoint = checkpoint_root / f"step_{step:08d}"
        if world_size > 1:
            dist.barrier()
        return TrainingResult(step, tokens_seen, global_microstep, last_loss, checkpoint)
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()

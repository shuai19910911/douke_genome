#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import torch

from legumegenomefm.masking import mask_span_mlm
from legumegenomefm.model import LegumeGenomeConfig, LegumeGenomeModel
from legumegenomefm.tokenizer import complement_logits, reverse_complement_tokens
from legumegenomefm.training_data import GenomeWindowSampler


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def git_head(project_root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen formal model on one real genome batch")
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA-visible GPU is required")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda", 0)
    config = LegumeGenomeConfig(dropout=0.0)
    model = LegumeGenomeModel(config).to(device)
    model.set_gradient_checkpointing(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    sampler = GenomeWindowSampler(
        args.dataset_manifest,
        args.project_root,
        context_length=args.context_length,
        split="pretrain",
        seed=args.seed,
    )
    tokens = sampler.sample_batch(batch_size=args.micro_batch_size, global_microstep=0, rank=0)
    generator = torch.Generator().manual_seed(args.seed)
    inputs, labels = mask_span_mlm(tokens, generator)
    inputs = inputs.to(device)
    labels = labels.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(inputs, labels=labels)
    if output.loss is None or not torch.isfinite(output.loss):
        raise RuntimeError("non-finite formal-model loss")
    scaler.scale(output.loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("non-finite formal-model gradient")
    scaler.step(optimizer)
    scaler.update()
    model.eval()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        probe = inputs[:, : min(128, inputs.shape[1])]
        logits = model(probe).logits.float()
        rc_logits = model(reverse_complement_tokens(probe)).logits.float()
        aligned = complement_logits(rc_logits.flip(1))
        rc_max_abs_error = float((logits - aligned).abs().max().item())
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    result = {
        "schema_version": "1.0",
        "state": "PASS",
        "git_commit": git_head(args.project_root),
        "dataset_manifest_sha256": hashlib.sha256(args.dataset_manifest.read_bytes()).hexdigest(),
        "gpu_name": properties.name,
        "gpu_uuid": args.gpu_uuid,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "allocator_fraction": args.memory_fraction,
        "context_length": args.context_length,
        "micro_batch_size": args.micro_batch_size,
        "parameter_count": model.parameter_count(),
        "loss": float(output.loss.detach().float().item()),
        "gradient_norm": float(gradient_norm.detach().float().item()),
        "rc_max_abs_error": rc_max_abs_error,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    if result["parameter_count"] != 88_946_028 or not math.isfinite(result["loss"]):
        raise RuntimeError("formal model identity or loss validation failed")
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

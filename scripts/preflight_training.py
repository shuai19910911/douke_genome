#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confined_relative(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != value:
        raise ValueError(f"unsafe {field}: {value}")
    return Path(*pure.parts)


def resolve_project_root(config_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("project_root must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or pure.as_posix() != value:
        raise ValueError(f"unsafe project_root: {value}")
    root = (config_path.parent / Path(*pure.parts)).resolve()
    try:
        config_path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("training config must remain inside project_root") from exc
    if not (root / "pyproject.toml").is_file():
        raise ValueError("project_root does not contain pyproject.toml")
    return root


def resolved_config(config_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    project_root = resolve_project_root(config_path, payload.get("project_root"))
    dataset = project_root / confined_relative(payload.get("dataset_manifest"), "dataset_manifest")
    output = project_root / confined_relative(payload.get("output_dir"), "output_dir")
    return payload, project_root, dataset, output


def validate_contract_status(payload: dict[str, object]) -> None:
    status = payload.get("contract_status", "missing")
    if status != "frozen":
        raise ValueError(f"training contract is not launchable: {status}")


def validate_ultralong_static_contract(payload: dict[str, object]) -> None:
    expected_contexts = [1024, 8192, 32768, 65536, 131072, 262144]
    contexts = [int(value) for value in payload.get("contexts", [])]
    if contexts != expected_contexts:
        raise ValueError(f"formal contexts must equal {expected_contexts}: {contexts}")
    model = payload.get("model", {})
    if not isinstance(model, dict) or int(model.get("maximum_context_bp", 0)) != contexts[-1]:
        raise ValueError("model maximum context does not equal the formal maximum")
    channels = [int(value) for value in model.get("encoder_channels", [])]
    global_core = model.get("global_core", {})
    if (
        not channels
        or int(model.get("base_embedding_dim", 0)) != channels[0]
        or not isinstance(global_core, dict)
        or int(global_core.get("d_model", 0)) != channels[-1]
    ):
        raise ValueError("embedding, encoder and global-core channel dimensions do not close")
    if model.get("tie_input_output_embeddings") is not True or model.get("base_resolution_skip") is not True:
        raise ValueError("the candidate requires tied embeddings and base-resolution skip features")
    reverse_complement = model.get("reverse_complement", {})
    if (
        not isinstance(reverse_complement, dict)
        or reverse_complement.get("mechanism") != "whole_model_conjoin"
        or reverse_complement.get("combine") != "aligned_logit_mean"
        or reverse_complement.get("exact_output_contract") is not True
    ):
        raise ValueError("the whole-model reverse-complement contract is incomplete")
    if (
        global_core.get("operator") != "mamba2_bidirectional"
        or global_core.get("bidirectional_strategy") != "add"
        or global_core.get("tie_forward_reverse_projections") is not True
    ):
        raise ValueError("the bidirectional Mamba-2 contract does not match the implementation")
    strides = [int(value) for value in model.get("encoder_strides", [])]
    if strides != [1, *([2] * (len(strides) - 1))]:
        raise ValueError("encoder strides must be one followed by stride-two transitions")
    stride_product = 1
    for stride in strides:
        if stride < 1:
            raise ValueError("encoder strides must be positive")
        stride_product *= stride
    latent_stride = int(model.get("latent_stride_bp", 0))
    if stride_product != latent_stride or latent_stride != 128:
        raise ValueError("encoder stride product must equal the 128-bp latent stride")
    if any(context % latent_stride for context in contexts):
        raise ValueError("every formal context must be divisible by the latent stride")
    if contexts[-1] // latent_stride != 2048:
        raise ValueError("maximum global latent length must be 2048")

    fractions = payload.get("context_token_fractions", {})
    if not isinstance(fractions, dict) or {int(key) for key in fractions} != set(contexts):
        raise ValueError("token fractions must cover every formal context exactly once")
    fraction_sum = sum(float(fractions[context]) for context in contexts)
    if abs(fraction_sum - 1.0) > 1e-12 or any(float(fractions[context]) <= 0 for context in contexts):
        raise ValueError(f"context token fractions must be positive and sum to one: {fraction_sum}")

    micro_batches = payload.get("micro_batch_per_gpu", {})
    expected_tokens = int(payload.get("microstep_tokens_per_gpu", 0))
    if not isinstance(micro_batches, dict) or expected_tokens < 1:
        raise ValueError("micro-batch contract is incomplete")
    per_context_tokens = {context: context * int(micro_batches[context]) for context in contexts}
    if set(per_context_tokens.values()) != {expected_tokens}:
        raise ValueError(f"per-context microstep tokens are inconsistent: {per_context_tokens}")

    mixed = payload.get("mixed_context", {})
    if not isinstance(mixed, dict) or mixed.get("allocation_unit") != "tokens":
        raise ValueError("mixed contexts require token allocation, not sample allocation")
    if mixed.get("bucket_choice") != "optimizer_step":
        raise ValueError("the length bucket must be chosen at optimizer-step granularity")
    if mixed.get("synchronize_length_across_ranks") is not True:
        raise ValueError("the length bucket must be synchronized across all ranks")
    if mixed.get("single_optimizer_lineage") is not True or mixed.get("single_scheduler_lineage") is not True:
        raise ValueError("mixed contexts require one optimizer and scheduler lineage")
    if mixed.get("progress_unit") != "tokens_seen" or mixed.get("loss_normalization") != "global_masked_tokens":
        raise ValueError("progress and loss must use tokens_seen and global masked tokens")


def validate_h20_runtime_evidence(payload: dict[str, object], project_root: Path) -> None:
    hardware = payload.get("hardware_contract", {})
    if not isinstance(hardware, dict) or hardware.get("profile_receipt_status") != "PASS":
        raise ValueError("H20 profile receipt is not verified")
    model = payload.get("model", {})
    if not isinstance(model, dict) or not isinstance(model.get("parameter_count"), int):
        raise ValueError("the model parameter count is not frozen as an integer")
    global_core = model.get("global_core", {})
    kernel = payload.get("kernel_contract", {})
    if not isinstance(global_core, dict) or global_core.get("backend_status") != "h20_verified":
        raise ValueError("the Mamba-2 backend is not H20 verified")
    if not isinstance(kernel, dict) or kernel.get("selection_status") != "frozen":
        raise ValueError("the equal-budget global-kernel comparison is not frozen")

    receipt_path = project_root / confined_relative(hardware.get("profile_receipt"), "profile_receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("state") != "PASS" or receipt.get("gpu_model") != hardware.get("target_gpu"):
        raise ValueError("H20 profile receipt state or GPU model mismatch")
    supported_world_sizes = {int(value) for value in payload.get("distributed", {}).get("supported_world_sizes", [])}
    if supported_world_sizes != {2, 3} or int(receipt.get("gpu_count", 0)) not in supported_world_sizes:
        raise ValueError("H20 receipt does not bind a supported two- or three-GPU topology")
    expected_contexts = {str(int(value)) for value in payload["contexts"]}
    profiles = receipt.get("context_profiles", {})
    if not isinstance(profiles, dict) or set(profiles) != expected_contexts:
        raise ValueError("H20 receipt does not cover every formal context exactly once")
    minimum_margin = float(hardware["minimum_free_memory_fraction_after_peak"])
    for context, profile in profiles.items():
        if (
            not isinstance(profile, dict)
            or profile.get("state") != "PASS"
            or profile.get("finite") is not True
            or profile.get("optimizer_step_completed") is not True
            or float(profile.get("free_memory_fraction_after_peak", -1.0)) < minimum_margin
        ):
            raise ValueError(f"H20 context profile is incomplete or unsafe: {context}")
    ddp = receipt.get("ddp_profiles", {})
    if not isinstance(ddp, dict) or any(ddp.get(str(size), {}).get("state") != "PASS" for size in (2, 3)):
        raise ValueError("both two- and three-GPU DDP profiles must pass")
    selection = receipt.get("kernel_selection", {})
    if (
        not isinstance(selection, dict)
        or selection.get("state") != "PASS"
        or selection.get("selected") != kernel.get("primary_operator")
        or selection.get("comparator") != kernel.get("comparator")
    ):
        raise ValueError("H20 kernel-selection receipt mismatch")


def validate_dataset(project_root: Path, dataset_path: Path) -> dict[str, int]:
    release_path = dataset_path.parent / "training_dataset.release.json"
    release_bytes = release_path.read_bytes()
    release_ready = (dataset_path.parent / "TRAINING_DATASET_READY").read_text(encoding="ascii").strip()
    if hashlib.sha256(release_bytes).hexdigest() != release_ready:
        raise ValueError("training dataset release READY mismatch")
    release = json.loads(release_bytes)
    if release.get("state") != "READY" or release.get("dataset_manifest_sha256") != sha256(dataset_path):
        raise ValueError("training dataset release does not bind the dataset manifest")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    store_root = project_root / confined_relative(payload.get("store_root"), "store_root")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset manifest has no sources")
    candidate_ids: set[str] = set()
    counts = {"pretrain": 0, "cold_genus_holdout": 0}
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("dataset source must be a mapping")
        candidate_id = source.get("candidate_id")
        if not isinstance(candidate_id, str) or len(candidate_id) != 16 or candidate_id in candidate_ids:
            raise ValueError(f"invalid or duplicate candidate_id: {candidate_id}")
        candidate_ids.add(candidate_id)
        split = source.get("split")
        if split not in counts:
            raise ValueError(f"invalid dataset split: {split}")
        counts[str(split)] += 1
        if split == "pretrain" and float(source.get("sampling_weight", 0.0)) <= 0:
            raise ValueError(f"pretrain source has non-positive sampling weight: {candidate_id}")
        directory = store_root / candidate_id
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"missing regular store directory: {candidate_id}")
        manifest = directory / "manifest.json"
        ready = (directory / "READY").read_text(encoding="ascii").strip()
        if ready != source.get("store_manifest_sha256") or sha256(manifest) != ready:
            raise ValueError(f"store READY/hash mismatch: {candidate_id}")
        store_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if store_payload.get("identity", {}).get("candidate_id") != candidate_id:
            raise ValueError(f"store candidate identity mismatch: {candidate_id}")
        packed = directory / "sequence.2bit"
        if packed.is_symlink() or not packed.is_file() or packed.stat().st_size != store_payload.get("packed_size_bytes"):
            raise ValueError(f"store payload size/type mismatch: {candidate_id}")
    if not all(counts.values()):
        raise ValueError("both pretrain and cold_genus_holdout roles must be non-empty")
    return {"source_count": len(sources), **counts}


def validate_mode(mode: str, output: Path, initialize_from: Path | None) -> None:
    if mode == "fresh":
        if output.exists():
            raise ValueError(f"fresh launch refuses existing output: {output}")
        if initialize_from is not None:
            raise ValueError("fresh launch does not accept initialize_from")
    elif mode == "resume":
        if initialize_from is not None:
            raise ValueError("resume launch does not accept initialize_from")
        if not any((output / "checkpoints").glob("step_*/READY")):
            raise ValueError("resume launch requires an existing checkpoint READY")
    elif mode == "initialize":
        if output.exists():
            raise ValueError(f"initialize launch refuses existing output: {output}")
        if initialize_from is None or not (initialize_from / "READY").is_file():
            raise ValueError("initialize launch requires a checkpoint directory with READY")
    else:
        raise ValueError(f"unsupported launch mode: {mode}")


def validate_gpu_contract(payload: dict[str, object], nproc: int, minimum_free_mib: int) -> list[dict[str, object]]:
    if nproc < 1 or minimum_free_mib < 1:
        raise ValueError("nproc and minimum free memory must be positive")
    global_batch = int(payload["global_batch_tokens"])
    if "contexts" in payload:
        supported = {int(value) for value in payload["distributed"]["supported_world_sizes"]}
        if nproc not in supported:
            raise ValueError(f"requested world size {nproc} is outside the frozen set {sorted(supported)}")
        micro_batches = payload["micro_batch_per_gpu"]
        for context in payload["contexts"]:
            if global_batch % (int(context) * int(micro_batches[int(context)]) * nproc):
                raise ValueError(f"global batch tokens are not divisible at context {context} and world size {nproc}")
        max_tokens = int(payload["optimizer"]["total_tokens"])
    else:
        context = int(payload["context_length"])
        micro_batch = int(payload["micro_batch_size"])
        if global_batch % (context * micro_batch * nproc):
            raise ValueError("global batch tokens are not divisible by the requested GPU count")
        max_tokens = int(payload["max_tokens"])
    if max_tokens <= 0 or max_tokens % global_batch:
        raise ValueError("max_tokens must be a positive whole number of global token batches")
    if not torch.cuda.is_available() or torch.cuda.device_count() < nproc:
        raise ValueError(f"requested {nproc} CUDA devices are not available")
    devices: list[dict[str, object]] = []
    for index in range(nproc):
        properties = torch.cuda.get_device_properties(index)
        if payload.get("precision") == "bf16" and properties.major < 8:
            raise ValueError(f"GPU {index} does not support the frozen BF16 path")
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        free_mib = free_bytes // (1024 * 1024)
        if free_mib < minimum_free_mib:
            raise ValueError(f"GPU {index} free memory {free_mib} MiB is below {minimum_free_mib} MiB")
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "free_mib": free_mib,
                "total_mib": total_bytes // (1024 * 1024),
            }
        )
    return devices


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed preflight for a frozen training stage")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mode", choices=("fresh", "resume", "initialize"), required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--minimum-free-mib", type=int, default=6000)
    args = parser.parse_args()
    config_path = args.config.resolve()
    payload, project_root, dataset, output = resolved_config(config_path)
    validate_contract_status(payload)
    validate_ultralong_static_contract(payload)
    validate_h20_runtime_evidence(payload, project_root)
    counts = validate_dataset(project_root, dataset)
    initialization = args.initialize_from.resolve() if args.initialize_from else None
    validate_mode(args.mode, output, initialization)
    devices = validate_gpu_contract(payload, args.nproc_per_node, args.minimum_free_mib)
    result = {
        "schema_version": "1.0",
        "state": "PASS",
        "mode": args.mode,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "output": str(output),
        "nproc_per_node": args.nproc_per_node,
        "counts": counts,
        "devices": devices,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

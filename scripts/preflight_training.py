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


def resolved_config(config_path: Path) -> tuple[dict[str, object], Path, Path, Path]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be a mapping")
    project_reference = confined_relative(payload.get("project_root"), "project_root")
    project_root = (config_path.parent / project_reference).resolve()
    dataset = project_root / confined_relative(payload.get("dataset_manifest"), "dataset_manifest")
    output = project_root / confined_relative(payload.get("output_dir"), "output_dir")
    return payload, project_root, dataset, output


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
    context = int(payload["context_length"])
    micro_batch = int(payload["micro_batch_size"])
    global_batch = int(payload["global_batch_tokens"])
    if global_batch % (context * micro_batch * nproc):
        raise ValueError("global batch tokens are not divisible by the requested GPU count")
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

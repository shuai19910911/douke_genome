from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_reference_receipt(
    project_root: Path,
    reference_root: Path,
    receipt_path: Path,
    ready_path: Path,
    *,
    full: bool = False,
) -> str:
    project_root = project_root.resolve()
    reference_root = reference_root.resolve()
    receipt_bytes = receipt_path.resolve().read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if ready_path.resolve().read_text(encoding="ascii").strip() != receipt_sha:
        raise ValueError("reference READY/hash mismatch")
    receipt = json.loads(receipt_bytes)
    if receipt.get("state") != "READY":
        raise ValueError("reference receipt is not READY")
    relative = Path(str(receipt.get("lineage_relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts or (project_root / relative).resolve() != reference_root:
        raise ValueError("reference receipt path mismatch")

    files: list[Path] = []
    for item in sorted(reference_root.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"reference contains a symlink: {item}")
        if item.is_file():
            files.append(item)
        elif not item.is_dir():
            raise ValueError(f"reference contains an unsafe entry: {item}")
    inventory = [[path.relative_to(reference_root).as_posix(), path.stat().st_size] for path in files]
    inventory_sha = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
    if (
        len(files) != int(receipt.get("file_count", -1))
        or sum(size for _, size in inventory) != int(receipt.get("total_bytes", -1))
        or inventory_sha != receipt.get("path_size_inventory_sha256")
    ):
        raise ValueError("reference path/size inventory mismatch")
    dataset_cfg = reference_root / "dataset.cfg"
    if not dataset_cfg.is_file() or sha256(dataset_cfg) != receipt.get("dataset_cfg_sha256"):
        raise ValueError("reference dataset.cfg hash mismatch")
    if full:
        full_inventory = [[relative_path, size, sha256(reference_root / relative_path)] for relative_path, size in inventory]
        full_sha = hashlib.sha256(json.dumps(full_inventory, separators=(",", ":")).encode()).hexdigest()
        if full_sha != receipt.get("full_tree_sha256"):
            raise ValueError("reference full-tree hash mismatch")
    return receipt_sha


def validate_busco_lineage(project_root: Path, lineage_root: Path, *, full: bool = False) -> str:
    lineage_id = lineage_root.resolve().name
    manifests = project_root.resolve() / "data_manifests"
    return validate_reference_receipt(
        project_root,
        lineage_root,
        manifests / f"busco_lineage_{lineage_id}.receipt.json",
        manifests / f"busco_lineage_{lineage_id}.READY",
        full=full,
    )

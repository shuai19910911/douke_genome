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


def validate_contamination_references(
    project_root: Path,
    tiara_image: Path,
    univec_prefix: Path,
    blastn: Path,
    *,
    full: bool = False,
) -> str:
    project_root = project_root.resolve()
    tiara_image = tiara_image.resolve()
    univec_prefix = univec_prefix.resolve()
    blastn = blastn.resolve()
    manifests = project_root / "data_manifests"
    receipt_path = manifests / "contamination_references.receipt.json"
    ready_path = manifests / "contamination_references.READY"
    receipt_bytes = receipt_path.read_bytes()
    receipt_sha = hashlib.sha256(receipt_bytes).hexdigest()
    if ready_path.read_text(encoding="ascii").strip() != receipt_sha:
        raise ValueError("contamination reference READY/hash mismatch")
    receipt = json.loads(receipt_bytes)
    if receipt.get("state") != "READY":
        raise ValueError("contamination reference receipt is not READY")
    tiara = receipt.get("tiara")
    univec = receipt.get("univec")
    blast = receipt.get("blast")
    if not all(isinstance(value, dict) for value in (tiara, univec, blast)):
        raise ValueError("contamination reference receipt sections are missing")
    assert isinstance(tiara, dict) and isinstance(univec, dict) and isinstance(blast, dict)
    if (project_root / str(tiara.get("container_relative_path", ""))).resolve() != tiara_image:
        raise ValueError("Tiara image path mismatch")
    if not tiara_image.is_file() or tiara_image.is_symlink() or tiara_image.stat().st_size != int(tiara.get("size_bytes", -1)):
        raise ValueError("Tiara image size/type mismatch")
    if (project_root / str(univec.get("database_relative_prefix", ""))).resolve() != univec_prefix:
        raise ValueError("UniVec database path mismatch")
    univec_files = sorted(univec_prefix.parent.glob(f"{univec_prefix.name}*"))
    if not univec_files or any(path.is_symlink() or not path.is_file() for path in univec_files):
        raise ValueError("UniVec database contains unsafe or missing files")
    inventory = [[path.name, path.stat().st_size] for path in univec_files]
    inventory_sha = hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest()
    if (
        len(inventory) != int(univec.get("file_count", -1))
        or sum(size for _, size in inventory) != int(univec.get("total_bytes", -1))
        or inventory_sha != univec.get("path_size_inventory_sha256")
    ):
        raise ValueError("UniVec path/size inventory mismatch")
    if not blastn.is_file() or blastn.is_symlink() or blastn.stat().st_size != int(blast.get("executable_size_bytes", -1)):
        raise ValueError("blastn executable size/type mismatch")
    if full:
        if sha256(tiara_image) != tiara.get("sha256") or sha256(blastn) != blast.get("executable_sha256"):
            raise ValueError("Tiara or blastn full hash mismatch")
        full_inventory = [[name, size, sha256(univec_prefix.parent / name)] for name, size in inventory]
        full_sha = hashlib.sha256(json.dumps(full_inventory, separators=(",", ":")).encode()).hexdigest()
        if full_sha != univec.get("full_tree_sha256"):
            raise ValueError("UniVec full-tree hash mismatch")
    return receipt_sha


def load_contamination_legacy_bindings(project_root: Path, reference_receipt_sha256: str) -> dict[str, str]:
    manifests = project_root.resolve() / "data_manifests"
    receipt_path = manifests / "contamination_legacy_binding.receipt.json"
    ready_path = manifests / "contamination_legacy_binding.READY"
    receipt_bytes = receipt_path.read_bytes()
    if ready_path.read_text(encoding="ascii").strip() != hashlib.sha256(receipt_bytes).hexdigest():
        raise ValueError("contamination legacy binding READY/hash mismatch")
    receipt = json.loads(receipt_bytes)
    if receipt.get("state") != "READY" or receipt.get("reference_receipt_sha256") != reference_receipt_sha256:
        raise ValueError("contamination legacy binding reference mismatch")
    rows = receipt.get("bound_shards")
    if not isinstance(rows, list) or len(rows) != int(receipt.get("bound_shard_count", -1)):
        raise ValueError("contamination legacy binding count mismatch")
    bindings: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid contamination legacy binding row")
        candidate_id = str(row.get("candidate_id", ""))
        shard_sha = str(row.get("shard_sha256", ""))
        if (
            len(candidate_id) != 16
            or any(character not in "0123456789abcdef" for character in candidate_id)
            or len(shard_sha) != 64
            or any(character not in "0123456789abcdef" for character in shard_sha)
            or candidate_id in bindings
        ):
            raise ValueError("invalid or duplicate contamination legacy binding")
        bindings[candidate_id] = shard_sha
    return bindings

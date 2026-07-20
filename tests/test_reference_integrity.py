from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from legumegenomefm.reference_integrity import (
    load_contamination_legacy_bindings,
    validate_busco_lineage,
    validate_contamination_references,
    validate_reference_receipt,
)


def _write_receipt(project: Path, lineage: Path) -> tuple[Path, Path]:
    files = sorted(path for path in lineage.rglob("*") if path.is_file())
    inventory = [[path.relative_to(lineage).as_posix(), path.stat().st_size] for path in files]
    full = [
        [relative, size, hashlib.sha256((lineage / relative).read_bytes()).hexdigest()]
        for relative, size in inventory
    ]
    receipt = {
        "schema_version": "1.0",
        "state": "READY",
        "lineage_relative_path": lineage.relative_to(project).as_posix(),
        "file_count": len(files),
        "total_bytes": sum(size for _, size in inventory),
        "dataset_cfg_sha256": hashlib.sha256((lineage / "dataset.cfg").read_bytes()).hexdigest(),
        "path_size_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, separators=(",", ":")).encode()
        ).hexdigest(),
        "full_tree_sha256": hashlib.sha256(
            json.dumps(full, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    receipt_path = project / "data_manifests/reference.receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    ready_path = project / "data_manifests/reference.READY"
    ready_path.write_text(hashlib.sha256(receipt_bytes).hexdigest() + "\n")
    return receipt_path, ready_path


def test_reference_receipt_validates_ready_and_path_size_inventory(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lineage = project / "data/reference/busco/lineages/eudicots_odb10"
    lineage.mkdir(parents=True)
    (lineage / "dataset.cfg").write_text("name=eudicotyledons_odb10\n")
    (lineage / "marker.txt").write_text("marker\n")
    receipt, ready = _write_receipt(project, lineage)

    receipt_sha = validate_reference_receipt(project, lineage, receipt, ready)

    assert receipt_sha == ready.read_text().strip()


def test_reference_receipt_rejects_path_size_inventory_change(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lineage = project / "data/reference/busco/lineages/eudicots_odb10"
    lineage.mkdir(parents=True)
    (lineage / "dataset.cfg").write_text("name=eudicotyledons_odb10\n")
    marker = lineage / "marker.txt"
    marker.write_text("marker\n")
    receipt, ready = _write_receipt(project, lineage)
    marker.write_text("marker changed\n")

    with pytest.raises(ValueError, match="path/size inventory"):
        validate_reference_receipt(project, lineage, receipt, ready)


def test_busco_lineage_resolves_project_receipt_by_lineage_id(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lineage = project / "data/reference/busco/lineages/eudicots_odb10"
    lineage.mkdir(parents=True)
    (lineage / "dataset.cfg").write_text("name=eudicotyledons_odb10\n")
    receipt, ready = _write_receipt(project, lineage)
    expected_receipt = project / "data_manifests/busco_lineage_eudicots_odb10.receipt.json"
    expected_ready = project / "data_manifests/busco_lineage_eudicots_odb10.READY"
    receipt.rename(expected_receipt)
    ready.rename(expected_ready)

    assert validate_busco_lineage(project, lineage) == expected_ready.read_text().strip()


def test_contamination_reference_receipt_validates_full_payloads(tmp_path: Path) -> None:
    project = tmp_path / "project"
    image = project / "data/reference/containers/tiara.sif"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"tiara")
    prefix = project / "data/reference/univec/UniVec_Core"
    prefix.parent.mkdir(parents=True)
    prefix.write_bytes(b">vector\nACGT\n")
    (prefix.parent / "UniVec_Core.ndb").write_bytes(b"database")
    blastn = tmp_path / "env/bin/blastn"
    blastn.parent.mkdir(parents=True)
    blastn.write_bytes(b"blastn")
    univec_files = sorted(prefix.parent.glob("UniVec_Core*"))
    inventory = [[path.name, path.stat().st_size] for path in univec_files]
    full_inventory = [
        [name, size, hashlib.sha256((prefix.parent / name).read_bytes()).hexdigest()]
        for name, size in inventory
    ]
    receipt = {
        "state": "READY",
        "tiara": {
            "container_relative_path": image.relative_to(project).as_posix(),
            "size_bytes": image.stat().st_size,
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        },
        "univec": {
            "database_relative_prefix": prefix.relative_to(project).as_posix(),
            "file_count": len(inventory),
            "total_bytes": sum(size for _, size in inventory),
            "path_size_inventory_sha256": hashlib.sha256(
                json.dumps(inventory, separators=(",", ":")).encode()
            ).hexdigest(),
            "full_tree_sha256": hashlib.sha256(
                json.dumps(full_inventory, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "blast": {
            "executable_size_bytes": blastn.stat().st_size,
            "executable_sha256": hashlib.sha256(blastn.read_bytes()).hexdigest(),
        },
    }
    receipt_path = project / "data_manifests/contamination_references.receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    ready = project / "data_manifests/contamination_references.READY"
    ready.write_text(hashlib.sha256(receipt_bytes).hexdigest() + "\n")

    assert validate_contamination_references(project, image, prefix, blastn, full=True) == ready.read_text().strip()


def test_contamination_legacy_binding_receipt_returns_bound_shard_hashes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    manifests = project / "data_manifests"
    manifests.mkdir(parents=True)
    reference_sha = "a" * 64
    binding = {
        "state": "READY",
        "reference_receipt_sha256": reference_sha,
        "bound_shard_count": 1,
        "bound_shards": [{"candidate_id": "b" * 16, "shard_sha256": "c" * 64}],
    }
    receipt = manifests / "contamination_legacy_binding.receipt.json"
    payload = (json.dumps(binding, sort_keys=True) + "\n").encode()
    receipt.write_bytes(payload)
    (manifests / "contamination_legacy_binding.READY").write_text(
        hashlib.sha256(payload).hexdigest() + "\n"
    )

    assert load_contamination_legacy_bindings(project, reference_sha) == {"b" * 16: "c" * 64}

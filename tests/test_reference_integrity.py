from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from legumegenomefm.reference_integrity import validate_busco_lineage, validate_reference_receipt


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

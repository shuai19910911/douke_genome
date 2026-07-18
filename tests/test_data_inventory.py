from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from legumegenomefm.data_inventory import classify_path, scan_tree, write_inventory


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("assembly.fna", "fasta"),
        ("assembly.fa.gz", "fasta"),
        ("genes.gff3.gz", "annotation"),
        ("variants.vcf.gz", "variant"),
        ("regions.bed", "interval"),
        ("checksums.sha256", "checksum"),
        ("bundle.tar.gz", "archive"),
        ("notes.txt", "other"),
    ],
)
def test_classify_path_recognizes_genomics_files(name: str, expected: str) -> None:
    assert classify_path(Path(name)) == expected


def test_scan_tree_is_relative_deterministic_and_does_not_follow_symlinks(
    tmp_path: Path,
) -> None:
    (tmp_path / "species_b").mkdir()
    (tmp_path / "species_a").mkdir()
    (tmp_path / "species_b" / "genes.gff3").write_text("##gff-version 3\n")
    (tmp_path / "species_a" / "genome.fna").write_text(">chr1\nACGTN\n")
    (tmp_path / "species_a" / "genome-link.fna").symlink_to("genome.fna")

    first = scan_tree(tmp_path)
    second = scan_tree(tmp_path)

    assert first == second
    assert [record.relative_path for record in first] == [
        "species_a/genome-link.fna",
        "species_a/genome.fna",
        "species_b/genes.gff3",
    ]
    link = first[0]
    assert link.kind == "symlink"
    assert link.link_target == "genome.fna"
    assert link.size_bytes == 0
    assert first[1].file_type == "fasta"
    assert first[2].file_type == "annotation"


def test_write_inventory_emits_tsv_summary_and_content_id(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    output_dir = tmp_path / "manifests"
    data_root.mkdir()
    (data_root / "a.fna").write_text(">a\nACGT\n")
    (data_root / "a.gff3").write_text("##gff-version 3\n")

    result = write_inventory(data_root, output_dir)

    assert result.inventory_tsv == output_dir / "raw_inventory.tsv"
    assert result.summary_json == output_dir / "raw_inventory.summary.json"
    assert result.inventory_tsv.is_file()
    assert result.summary_json.is_file()
    assert not list(output_dir.glob("*.tmp"))
    summary = json.loads(result.summary_json.read_text())
    assert summary["schema_version"] == "1.0"
    assert summary["file_count"] == 2
    assert summary["symlink_count"] == 0
    expected_bytes = (data_root / "a.fna").stat().st_size + (data_root / "a.gff3").stat().st_size
    assert summary["total_bytes"] == expected_bytes
    assert summary["counts_by_type"] == {"annotation": 1, "fasta": 1}
    assert len(summary["inventory_sha256"]) == 64
    inventory_text = result.inventory_tsv.read_text()
    assert str(data_root) not in inventory_text
    assert all(not line.endswith(("\t", " ")) for line in inventory_text.splitlines())


def test_inventory_cli_writes_outputs_and_prints_json(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    output_dir = tmp_path / "manifests"
    data_root.mkdir()
    (data_root / "assembly.fna").write_text(">chr1\nACGT\n")
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "inventory_raw_data.py"),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    status = json.loads(completed.stdout)
    assert status["state"] == "PASS"
    assert status["file_count"] == 1
    assert status["inventory_sha256"] == json.loads(
        (output_dir / "raw_inventory.summary.json").read_text()
    )["inventory_sha256"]

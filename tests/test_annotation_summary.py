from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from legumegenomefm.annotation_summary import aggregate_annotation_audit, write_annotation_audit


FIELDS = [
    "candidate_id",
    "relative_path",
    "source",
    "annotation_role",
    "is_primary_gene_model",
    "assembly_key",
    "paired_assembly_ids",
    "size_bytes",
    "mtime_ns",
]


def write_registry(path: Path) -> None:
    rows = [
        ["a" * 16, "x/a.gff3.gz", "x", "gene_models_main", "true", "x|a", "1" * 16, 10, 1],
        ["b" * 16, "x/b.gff3.gz", "x", "gene_models_main", "true", "x|b", "2" * 16, 11, 2],
        ["c" * 16, "x/c.gff3.gz", "x", "repeat", "false", "x|c", "", 12, 3],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)


def pass_payload(candidate_id: str, relative_path: str, genes: int, implementation: str) -> dict:
    return {
        "candidate": {
            "candidate_id": candidate_id,
            "relative_path": relative_path,
            "size_bytes": 10 if candidate_id.startswith("a") else 11,
            "mtime_ns": 1 if candidate_id.startswith("a") else 2,
        },
        "implementation_sha256": implementation,
        "state": "PASS",
        "stats": {
            "cds_count": genes,
            "comment_line_count": 1,
            "compression": "gzip",
            "duplicate_gene_id_count": 0,
            "duplicate_transcript_id_count": 0,
            "embedded_fasta": False,
            "exon_count": genes,
            "feature_count": genes * 4 + 1,
            "file_sha256": "1" * 64,
            "file_size_bytes": 10,
            "format": "GFF3",
            "gene_count": genes,
            "gff_version": "3",
            "invalid_coordinate_count": 0,
            "invalid_phase_count": 0,
            "invalid_strand_count": 0,
            "malformed_line_count": 0,
            "seqid_count": 1,
            "seqid_max_end": {"chr1": 100},
            "transcript_count": genes,
            "unique_gene_id_count": genes,
            "unique_transcript_id_count": genes,
        },
    }


def test_aggregate_annotation_audit_summarizes_primary_quality_and_pairing(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_root = tmp_path / "results"
    annotations = result_root / "annotations"
    runs = result_root / "runs"
    annotations.mkdir(parents=True)
    runs.mkdir()
    write_registry(registry)
    implementation = "f" * 64
    (annotations / f"{'a' * 16}.json").write_text(
        json.dumps(pass_payload("a" * 16, "x/a.gff3.gz", 2, implementation))
    )
    (annotations / f"{'b' * 16}.json").write_text(
        json.dumps(pass_payload("b" * 16, "x/b.gff3.gz", 0, implementation))
    )
    (runs / "shard_00.json").write_text(
        json.dumps(
            {
                "implementation_sha256": implementation,
                "state": "FAIL",
                "failures": [
                    {
                        "candidate_id": "c" * 16,
                        "error_type": "EOFError",
                        "error": "truncated",
                    }
                ],
            }
        )
    )

    audit = aggregate_annotation_audit(registry, result_root)

    assert [record.status for record in audit.records] == ["PASS", "PASS", "FAIL"]
    assert audit.summary["pass_count"] == 2
    assert audit.summary["fail_count"] == 1
    assert audit.summary["missing_count"] == 0
    assert audit.summary["primary_gene_model_pass_count"] == 2
    assert audit.summary["primary_gene_model_without_genes_count"] == 1
    assert audit.summary["paired_candidate_count"] == 2
    assert audit.summary["exact_file_duplicate_group_count"] == 1


def test_aggregate_annotation_audit_rejects_mixed_implementation_hashes(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_root = tmp_path / "results"
    annotations = result_root / "annotations"
    annotations.mkdir(parents=True)
    write_registry(registry)
    (annotations / f"{'a' * 16}.json").write_text(
        json.dumps(pass_payload("a" * 16, "x/a.gff3.gz", 2, "e" * 64))
    )
    (annotations / f"{'b' * 16}.json").write_text(
        json.dumps(pass_payload("b" * 16, "x/b.gff3.gz", 2, "f" * 64))
    )

    with pytest.raises(ValueError, match="implementation hashes"):
        aggregate_annotation_audit(registry, result_root)


def test_write_annotation_audit_is_deterministic(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_root = tmp_path / "results"
    annotations = result_root / "annotations"
    annotations.mkdir(parents=True)
    write_registry(registry)
    (annotations / f"{'a' * 16}.json").write_text(
        json.dumps(pass_payload("a" * 16, "x/a.gff3.gz", 2, "f" * 64))
    )
    audit = aggregate_annotation_audit(registry, result_root)

    first = write_annotation_audit(audit, tmp_path / "out")
    tsv = first.manifest_path.read_bytes()
    summary = first.summary_path.read_bytes()
    second = write_annotation_audit(audit, tmp_path / "out")

    assert second.manifest_path.read_bytes() == tsv
    assert second.summary_path.read_bytes() == summary
    assert all(not line.endswith((b"\t", b" ")) for line in tsv.splitlines())

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from legumegenomefm.audit_summary import aggregate_fasta_audit, write_fasta_audit


FIELDS = [
    "candidate_id",
    "relative_path",
    "source",
    "genus",
    "species",
    "assembly_label",
    "genome_role",
    "size_bytes",
    "mtime_ns",
]


def write_registry(path: Path) -> None:
    rows = [
        ["a" * 16, "source/a.fna.gz", "source", "A", "A one", "asm1", "main", 10, 1],
        ["b" * 16, "source/b.fna.gz", "source", "A", "A one", "asm2", "softmasked", 11, 2],
        ["c" * 16, "source/c.fna.gz", "source", "B", "B two", "asm3", "main", 12, 3],
        ["d" * 16, "source/d.fna.gz", "source", "C", "C three", "asm4", "main", 13, 4],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(rows)


def pass_payload(candidate_id: str, relative_path: str, sequence_sha: str, implementation: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "relative_path": relative_path,
        "input_size_bytes": 10 if candidate_id.startswith("a") else 11,
        "input_mtime_ns": 1 if candidate_id.startswith("a") else 2,
        "implementation_sha256": implementation,
        "schema_version": "1.0",
        "state": "PASS",
        "stats": {
            "acgt_count": 90,
            "iupac_ambiguous_count": 0,
            "canonical_sequence_sha256": sequence_sha,
            "compression": "gzip",
            "duplicate_header_count": 0,
            "empty_sequence_count": 0,
            "file_sha256": candidate_id * 4,
            "file_size_bytes": 10,
            "gc_count": 40,
            "invalid_symbol_count": 0,
            "lowercase_count": 0,
            "max_sequence_length": 100,
            "min_sequence_length": 100,
            "n50": 100,
            "n_count": 10,
            "sequence_count": 1,
            "total_symbols": 100,
        },
    }


def test_aggregate_fasta_audit_accounts_for_pass_fail_and_missing(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_root = tmp_path / "results"
    assemblies = result_root / "assemblies"
    runs = result_root / "runs"
    assemblies.mkdir(parents=True)
    runs.mkdir()
    write_registry(registry)
    implementation = "f" * 64
    duplicate_hash = "1" * 64
    for candidate_id, relative_path in [
        ("a" * 16, "source/a.fna.gz"),
        ("b" * 16, "source/b.fna.gz"),
    ]:
        (assemblies / f"{candidate_id}.json").write_text(
            json.dumps(pass_payload(candidate_id, relative_path, duplicate_hash, implementation))
        )
    (runs / "shard_00.json").write_text(
        json.dumps(
            {
                "failures": [
                    {
                        "candidate_id": "c" * 16,
                        "relative_path": "source/c.fna.gz",
                        "error_type": "EOFError",
                        "message": "truncated gzip",
                    }
                ],
                "implementation_sha256": implementation,
                "state": "FAIL",
            }
        )
    )

    audit = aggregate_fasta_audit(registry, result_root)

    assert [record.status for record in audit.records] == ["PASS", "PASS", "FAIL", "MISSING"]
    assert audit.summary["candidate_count"] == 4
    assert audit.summary["pass_count"] == 2
    assert audit.summary["fail_count"] == 1
    assert audit.summary["missing_count"] == 1
    assert audit.summary["exact_duplicate_group_count"] == 1
    assert audit.records[0].duplicate_group == audit.records[1].duplicate_group
    assert audit.records[0].duplicate_group_size == 2
    assert audit.records[2].error_type == "EOFError"
    assert audit.records[2].error == "truncated gzip"


def test_aggregate_fasta_audit_rejects_mixed_implementation_hashes(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_root = tmp_path / "results"
    assemblies = result_root / "assemblies"
    assemblies.mkdir(parents=True)
    write_registry(registry)
    (assemblies / f"{'a' * 16}.json").write_text(
        json.dumps(pass_payload("a" * 16, "source/a.fna.gz", "1" * 64, "e" * 64))
    )
    (assemblies / f"{'b' * 16}.json").write_text(
        json.dumps(pass_payload("b" * 16, "source/b.fna.gz", "2" * 64, "f" * 64))
    )

    with pytest.raises(ValueError, match="implementation hashes"):
        aggregate_fasta_audit(registry, result_root)


def test_write_fasta_audit_is_deterministic_and_has_no_trailing_whitespace(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_root = tmp_path / "results"
    assemblies = result_root / "assemblies"
    assemblies.mkdir(parents=True)
    write_registry(registry)
    payload = pass_payload("a" * 16, "source/a.fna.gz", "3" * 64, "f" * 64)
    (assemblies / f"{'a' * 16}.json").write_text(json.dumps(payload))
    audit = aggregate_fasta_audit(registry, result_root)

    first = write_fasta_audit(audit, tmp_path / "out")
    first_tsv = first.manifest_path.read_bytes()
    first_summary = first.summary_path.read_bytes()
    second = write_fasta_audit(audit, tmp_path / "out")

    assert second.manifest_path.read_bytes() == first_tsv
    assert second.summary_path.read_bytes() == first_summary
    assert all(not line.endswith((b"\t", b" ")) for line in first_tsv.splitlines())
    summary = json.loads(first_summary)
    assert summary["candidate_count"] == 4
    assert summary["manifest_sha256"]

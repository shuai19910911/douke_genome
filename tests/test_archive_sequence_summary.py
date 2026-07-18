from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path

from legumegenomefm.archive_sequence_audit import ArchiveGenomeCandidate
from legumegenomefm.archive_sequence_summary import (
    aggregate_archive_genome_audit,
    write_archive_genome_audit,
)


def candidate(value: str, material: str) -> ArchiveGenomeCandidate:
    return ArchiveGenomeCandidate(
        candidate_id=value * 16,
        archive_relative_path=f"soyod/{material}/genome/{material}.zip",
        member_name=f"{material}.fasta",
        material=material,
        archive_size_bytes=100,
        archive_mtime_ns=1,
        archive_sha256=value * 64,
        member_uncompressed_bytes=200,
        member_crc32_hex=value * 8,
    )


def write_registry(path: Path, candidates: list[ArchiveGenomeCandidate]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[field.name for field in fields(ArchiveGenomeCandidate)],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for value in candidates:
            writer.writerow(asdict(value))


def pass_payload(value: ArchiveGenomeCandidate, sequence_sha: str, implementation: str) -> dict:
    return {
        "schema_version": "1.0",
        "state": "PASS",
        **asdict(value),
        "implementation_sha256": implementation,
        "source": "soyod_zip",
        "stats": {
            "acgt_count": 90,
            "canonical_sequence_sha256": sequence_sha,
            "compression": "none",
            "duplicate_header_count": 0,
            "empty_sequence_count": 0,
            "file_sha256": "f" * 64,
            "file_size_bytes": 200,
            "gc_count": 40,
            "invalid_symbol_count": 0,
            "iupac_ambiguous_count": 0,
            "lowercase_count": 0,
            "max_sequence_length": 100,
            "min_sequence_length": 100,
            "n50": 100,
            "n_count": 10,
            "sequence_count": 2,
            "symbol_counts": {"A": 50, "C": 20, "G": 20, "N": 10},
            "total_symbols": 100,
        },
    }


def test_archive_genome_aggregate_accounts_for_pass_missing_and_duplicates(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    candidates = [candidate("a", "M1"), candidate("b", "M2"), candidate("c", "M3")]
    write_registry(registry, candidates)
    for value in candidates[:2]:
        (result_dir / f"{value.candidate_id}.json").write_text(
            json.dumps(pass_payload(value, "1" * 64, "e" * 64))
        )

    audit = aggregate_archive_genome_audit(registry, result_dir)

    assert [record.status for record in audit.records] == ["PASS", "PASS", "MISSING"]
    assert audit.summary["candidate_count"] == 3
    assert audit.summary["pass_count"] == 2
    assert audit.summary["missing_count"] == 1
    assert audit.summary["exact_duplicate_group_count"] == 1
    assert audit.summary["exact_duplicate_member_count"] == 2
    assert audit.records[0].duplicate_group == audit.records[1].duplicate_group


def test_write_archive_genome_audit_is_deterministic(tmp_path: Path) -> None:
    registry = tmp_path / "registry.tsv"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    value = candidate("a", "M1")
    write_registry(registry, [value])
    (result_dir / f"{value.candidate_id}.json").write_text(
        json.dumps(pass_payload(value, "1" * 64, "e" * 64))
    )
    audit = aggregate_archive_genome_audit(registry, result_dir)

    first = write_archive_genome_audit(audit, tmp_path / "out")
    first_manifest = first[0].read_bytes()
    first_summary = first[1].read_bytes()
    second = write_archive_genome_audit(audit, tmp_path / "out")

    assert second[0].read_bytes() == first_manifest
    assert second[1].read_bytes() == first_summary
    assert all(not line.endswith((b"\t", b" ")) for line in first_manifest.splitlines())

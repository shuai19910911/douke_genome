from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from legumegenomefm.archive_sequence_audit import ArchiveGenomeCandidate


@dataclass(frozen=True)
class ArchiveGenomeAuditRecord:
    candidate_id: str
    archive_relative_path: str
    member_name: str
    material: str
    archive_size_bytes: int
    archive_mtime_ns: int
    archive_sha256: str
    member_uncompressed_bytes: int
    member_crc32_hex: str
    sequence_count: int | None
    total_symbols: int | None
    acgt_count: int | None
    n_count: int | None
    gc_count: int | None
    lowercase_count: int | None
    iupac_ambiguous_count: int | None
    invalid_symbol_count: int | None
    duplicate_header_count: int | None
    empty_sequence_count: int | None
    min_sequence_length: int | None
    max_sequence_length: int | None
    n50: int | None
    member_file_sha256: str
    canonical_sequence_sha256: str
    duplicate_group: str
    duplicate_group_size: int
    status: str


@dataclass(frozen=True)
class ArchiveGenomeAudit:
    records: tuple[ArchiveGenomeAuditRecord, ...]
    summary: dict[str, Any]


def _read_registry(path: Path) -> list[ArchiveGenomeCandidate]:
    field_names = [field.name for field in fields(ArchiveGenomeCandidate)]
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != field_names:
            raise ValueError("archive genome registry schema mismatch")
        rows = list(reader)
    candidates = [
        ArchiveGenomeCandidate(
            candidate_id=row["candidate_id"],
            archive_relative_path=row["archive_relative_path"],
            member_name=row["member_name"],
            material=row["material"],
            archive_size_bytes=int(row["archive_size_bytes"]),
            archive_mtime_ns=int(row["archive_mtime_ns"]),
            archive_sha256=row["archive_sha256"],
            member_uncompressed_bytes=int(row["member_uncompressed_bytes"]),
            member_crc32_hex=row["member_crc32_hex"],
        )
        for row in rows
    ]
    ids = [candidate.candidate_id for candidate in candidates]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("archive genome registry IDs are empty or duplicated")
    return candidates


def aggregate_archive_genome_audit(registry_path: Path, result_dir: Path) -> ArchiveGenomeAudit:
    candidates = _read_registry(registry_path)
    expected = {candidate.candidate_id: candidate for candidate in candidates}
    payloads: dict[str, dict[str, Any]] = {}
    implementations: set[str] = set()
    for path in sorted(Path(result_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("state") != "PASS":
            raise ValueError(f"invalid archive genome result: {path.name}")
        candidate_id = str(payload.get("candidate_id") or "")
        if candidate_id not in expected or path.stem != candidate_id:
            raise ValueError(f"unexpected archive genome result: {path.name}")
        if candidate_id in payloads:
            raise ValueError(f"duplicate archive genome result: {candidate_id}")
        candidate = expected[candidate_id]
        for key, value in asdict(candidate).items():
            if payload.get(key) != value:
                raise ValueError(f"archive genome identity mismatch: {candidate_id}:{key}")
        implementation = str(payload.get("implementation_sha256") or "")
        if len(implementation) != 64:
            raise ValueError(f"invalid implementation SHA-256: {candidate_id}")
        implementations.add(implementation)
        stats = payload.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"missing archive genome stats: {candidate_id}")
        sequence_sha = str(stats.get("canonical_sequence_sha256") or "")
        if len(sequence_sha) != 64:
            raise ValueError(f"invalid canonical sequence SHA-256: {candidate_id}")
        payloads[candidate_id] = payload
    if len(implementations) > 1:
        raise ValueError("mixed implementation hashes in archive genome results")

    groups: dict[str, list[str]] = defaultdict(list)
    for candidate_id, payload in payloads.items():
        groups[payload["stats"]["canonical_sequence_sha256"]].append(candidate_id)
    duplicate_metadata: dict[str, tuple[str, int]] = {}
    duplicate_groups = 0
    duplicate_members = 0
    for sequence_sha, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        duplicate_groups += 1
        duplicate_members += len(ids)
        group_id = f"exact_{sequence_sha[:16]}"
        for candidate_id in ids:
            duplicate_metadata[candidate_id] = (group_id, len(ids))

    records: list[ArchiveGenomeAuditRecord] = []
    for candidate in candidates:
        common = asdict(candidate)
        payload = payloads.get(candidate.candidate_id)
        if payload is None:
            records.append(
                ArchiveGenomeAuditRecord(
                    **common,
                    sequence_count=None,
                    total_symbols=None,
                    acgt_count=None,
                    n_count=None,
                    gc_count=None,
                    lowercase_count=None,
                    iupac_ambiguous_count=None,
                    invalid_symbol_count=None,
                    duplicate_header_count=None,
                    empty_sequence_count=None,
                    min_sequence_length=None,
                    max_sequence_length=None,
                    n50=None,
                    member_file_sha256="",
                    canonical_sequence_sha256="",
                    duplicate_group="",
                    duplicate_group_size=0,
                    status="MISSING",
                )
            )
            continue
        stats = payload["stats"]
        group, group_size = duplicate_metadata.get(candidate.candidate_id, ("", 1))
        records.append(
            ArchiveGenomeAuditRecord(
                **common,
                sequence_count=int(stats["sequence_count"]),
                total_symbols=int(stats["total_symbols"]),
                acgt_count=int(stats["acgt_count"]),
                n_count=int(stats["n_count"]),
                gc_count=int(stats["gc_count"]),
                lowercase_count=int(stats["lowercase_count"]),
                iupac_ambiguous_count=int(stats["iupac_ambiguous_count"]),
                invalid_symbol_count=int(stats["invalid_symbol_count"]),
                duplicate_header_count=int(stats["duplicate_header_count"]),
                empty_sequence_count=int(stats["empty_sequence_count"]),
                min_sequence_length=int(stats["min_sequence_length"]),
                max_sequence_length=int(stats["max_sequence_length"]),
                n50=int(stats["n50"]),
                member_file_sha256=str(stats["file_sha256"]),
                canonical_sequence_sha256=str(stats["canonical_sequence_sha256"]),
                duplicate_group=group,
                duplicate_group_size=group_size,
                status="PASS",
            )
        )
    pass_records = [record for record in records if record.status == "PASS"]
    unique_groups = {
        record.canonical_sequence_sha256: record.total_symbols for record in pass_records
    }
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(records),
        "pass_count": len(pass_records),
        "missing_count": sum(record.status == "MISSING" for record in records),
        "implementation_sha256": next(iter(implementations), ""),
        "exact_duplicate_group_count": duplicate_groups,
        "exact_duplicate_member_count": duplicate_members,
        "unique_sequence_count": len(unique_groups),
        "total_symbols_in_pass_records": sum(record.total_symbols or 0 for record in pass_records),
        "unique_sequence_total_symbols": sum(int(value or 0) for value in unique_groups.values()),
    }
    return ArchiveGenomeAudit(tuple(records), summary)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_archive_genome_audit(
    audit: ArchiveGenomeAudit, output_dir: Path
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "archive_genome_qc.tsv"
    summary_path = output_dir / "archive_genome_qc.summary.json"
    stream = io.StringIO(newline="")
    field_names = [field.name for field in fields(ArchiveGenomeAuditRecord)]
    writer = csv.DictWriter(stream, fieldnames=field_names, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in audit.records:
        writer.writerow(asdict(record))
    manifest = stream.getvalue().encode("utf-8")
    summary = {**audit.summary, "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
    _atomic_write(manifest_path, manifest)
    _atomic_write(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest_path, summary_path

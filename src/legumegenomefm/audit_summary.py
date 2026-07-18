from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FastaAuditRecord:
    candidate_id: str
    relative_path: str
    source: str
    genus: str
    species: str
    assembly_label: str
    genome_role: str
    size_bytes: int
    mtime_ns: int
    sequence_count: int | None
    total_symbols: int | None
    acgt_count: int | None
    n_count: int | None
    gc_count: int | None
    lowercase_count: int | None
    ambiguous_iupac_count: int | None
    invalid_symbol_count: int | None
    duplicate_header_count: int | None
    empty_sequence_count: int | None
    min_sequence_length: int | None
    max_sequence_length: int | None
    n50: int | None
    compression: str
    file_sha256: str
    sequence_sha256: str
    duplicate_group: str
    duplicate_group_size: int
    error_type: str
    error: str
    status: str


@dataclass(frozen=True)
class FastaAudit:
    records: tuple[FastaAuditRecord, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class FastaAuditWriteResult:
    manifest_path: Path
    summary_path: Path


def _read_registry(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "candidate_id",
        "relative_path",
        "source",
        "genus",
        "species",
        "assembly_label",
        "genome_role",
        "size_bytes",
        "mtime_ns",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("candidate registry schema is invalid")
    ids = [row["candidate_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate registry contains duplicate IDs")
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return payload


def aggregate_fasta_audit(registry_path: Path, result_root: Path) -> FastaAudit:
    registry = _read_registry(Path(registry_path))
    expected = {row["candidate_id"]: row for row in registry}
    result_root = Path(result_root)
    passes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, str]] = {}
    implementation_hashes: set[str] = set()

    for path in sorted((result_root / "assemblies").glob("*.json")):
        payload = _read_json(path)
        if payload.get("state") != "PASS":
            raise ValueError(f"non-PASS assembly result: {path.name}")
        candidate_id = str(payload.get("candidate_id") or "")
        if candidate_id not in expected:
            raise ValueError(f"unexpected candidate result: {candidate_id}")
        if path.stem != candidate_id:
            raise ValueError(f"candidate filename mismatch: {candidate_id}")
        if candidate_id in passes:
            raise ValueError(f"duplicate candidate result: {candidate_id}")
        if payload.get("relative_path") != expected[candidate_id]["relative_path"]:
            raise ValueError(f"candidate path mismatch: {candidate_id}")
        if payload.get("input_size_bytes") != int(expected[candidate_id]["size_bytes"]):
            raise ValueError(f"candidate size mismatch: {candidate_id}")
        if payload.get("input_mtime_ns") != int(expected[candidate_id]["mtime_ns"]):
            raise ValueError(f"candidate mtime mismatch: {candidate_id}")
        implementation = str(payload.get("implementation_sha256") or "")
        if implementation:
            implementation_hashes.add(implementation)
        passes[candidate_id] = payload

    for path in sorted((result_root / "runs").glob("*.json")):
        payload = _read_json(path)
        implementation = str(payload.get("implementation_sha256") or "")
        if implementation:
            implementation_hashes.add(implementation)
        run_failures = payload.get("failures", [])
        if not isinstance(run_failures, list):
            raise ValueError(f"failures must be a list: {path.name}")
        for failure in run_failures:
            if not isinstance(failure, dict):
                raise ValueError(f"invalid failure record: {path.name}")
            candidate_id = str(failure.get("candidate_id") or "")
            if candidate_id not in expected:
                raise ValueError(f"unexpected failed candidate: {candidate_id}")
            if failure.get("relative_path") != expected[candidate_id]["relative_path"]:
                raise ValueError(f"failed candidate path mismatch: {candidate_id}")
            failures.setdefault(
                candidate_id,
                {
                    "error_type": str(failure.get("error_type") or "UnknownError"),
                    "error": str(failure.get("message") or failure.get("error") or ""),
                },
            )

    overlap = sorted(set(passes) & set(failures))
    if overlap:
        raise ValueError(f"candidates have both PASS and FAIL results: {','.join(overlap)}")

    if len(implementation_hashes) > 1:
        raise ValueError("mixed implementation hashes in FASTA audit results")

    sequence_groups: dict[str, list[str]] = defaultdict(list)
    for candidate_id, payload in passes.items():
        stats = payload.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"missing stats object: {candidate_id}")
        sequence_sha = str(stats.get("canonical_sequence_sha256") or "")
        if len(sequence_sha) != 64:
            raise ValueError(f"invalid sequence SHA-256: {candidate_id}")
        sequence_groups[sequence_sha].append(candidate_id)

    duplicate_metadata: dict[str, tuple[str, int]] = {}
    duplicate_groups = 0
    duplicate_members = 0
    for sequence_sha, candidate_ids in sorted(sequence_groups.items()):
        if len(candidate_ids) < 2:
            continue
        duplicate_groups += 1
        duplicate_members += len(candidate_ids)
        group_id = f"exact_{sequence_sha[:16]}"
        for candidate_id in candidate_ids:
            duplicate_metadata[candidate_id] = (group_id, len(candidate_ids))

    records: list[FastaAuditRecord] = []
    for row in registry:
        candidate_id = row["candidate_id"]
        common = {
            "candidate_id": candidate_id,
            "relative_path": row["relative_path"],
            "source": row["source"],
            "genus": row["genus"],
            "species": row["species"],
            "assembly_label": row["assembly_label"],
            "genome_role": row["genome_role"],
            "size_bytes": int(row["size_bytes"]),
            "mtime_ns": int(row["mtime_ns"]),
        }
        if candidate_id in passes:
            stats = passes[candidate_id]["stats"]
            duplicate_group, duplicate_group_size = duplicate_metadata.get(candidate_id, ("", 1))
            records.append(
                FastaAuditRecord(
                    **common,
                    sequence_count=int(stats["sequence_count"]),
                    total_symbols=int(stats["total_symbols"]),
                    acgt_count=int(stats["acgt_count"]),
                    n_count=int(stats["n_count"]),
                    gc_count=int(stats["gc_count"]),
                    lowercase_count=int(stats["lowercase_count"]),
                    ambiguous_iupac_count=int(stats["iupac_ambiguous_count"]),
                    invalid_symbol_count=int(stats["invalid_symbol_count"]),
                    duplicate_header_count=int(stats["duplicate_header_count"]),
                    empty_sequence_count=int(stats["empty_sequence_count"]),
                    min_sequence_length=int(stats["min_sequence_length"]),
                    max_sequence_length=int(stats["max_sequence_length"]),
                    n50=int(stats["n50"]),
                    compression=str(stats["compression"]),
                    file_sha256=str(stats["file_sha256"]),
                    sequence_sha256=str(stats["canonical_sequence_sha256"]),
                    duplicate_group=duplicate_group,
                    duplicate_group_size=duplicate_group_size,
                    error_type="",
                    error="",
                    status="PASS",
                )
            )
        elif candidate_id in failures:
            records.append(
                FastaAuditRecord(
                    **common,
                    sequence_count=None,
                    total_symbols=None,
                    acgt_count=None,
                    n_count=None,
                    gc_count=None,
                    lowercase_count=None,
                    ambiguous_iupac_count=None,
                    invalid_symbol_count=None,
                    duplicate_header_count=None,
                    empty_sequence_count=None,
                    min_sequence_length=None,
                    max_sequence_length=None,
                    n50=None,
                    compression="",
                    file_sha256="",
                    sequence_sha256="",
                    duplicate_group="",
                    duplicate_group_size=0,
                    error_type=failures[candidate_id]["error_type"],
                    error=failures[candidate_id]["error"],
                    status="FAIL",
                )
            )
        else:
            records.append(
                FastaAuditRecord(
                    **common,
                    sequence_count=None,
                    total_symbols=None,
                    acgt_count=None,
                    n_count=None,
                    gc_count=None,
                    lowercase_count=None,
                    ambiguous_iupac_count=None,
                    invalid_symbol_count=None,
                    duplicate_header_count=None,
                    empty_sequence_count=None,
                    min_sequence_length=None,
                    max_sequence_length=None,
                    n50=None,
                    compression="",
                    file_sha256="",
                    sequence_sha256="",
                    duplicate_group="",
                    duplicate_group_size=0,
                    error_type="",
                    error="",
                    status="MISSING",
                )
            )

    pass_records = [record for record in records if record.status == "PASS"]
    unique_sequence_symbols = 0
    seen_sequences: set[str] = set()
    for record in pass_records:
        if record.sequence_sha256 not in seen_sequences:
            seen_sequences.add(record.sequence_sha256)
            unique_sequence_symbols += int(record.total_symbols or 0)

    counts = Counter(record.status for record in records)
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "candidate_count": len(records),
        "pass_count": counts["PASS"],
        "fail_count": counts["FAIL"],
        "missing_count": counts["MISSING"],
        "implementation_sha256": next(iter(implementation_hashes), ""),
        "total_symbols_in_pass_records": sum(int(record.total_symbols or 0) for record in pass_records),
        "unique_sequence_total_symbols": unique_sequence_symbols,
        "unique_sequence_count": len(seen_sequences),
        "exact_duplicate_group_count": duplicate_groups,
        "exact_duplicate_member_count": duplicate_members,
        "invalid_symbol_record_count": sum(int(record.invalid_symbol_count or 0) > 0 for record in pass_records),
        "duplicate_header_record_count": sum(int(record.duplicate_header_count or 0) > 0 for record in pass_records),
        "empty_sequence_record_count": sum(int(record.empty_sequence_count or 0) > 0 for record in pass_records),
    }
    return FastaAudit(records=tuple(records), summary=summary)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_fasta_audit(audit: FastaAudit, output_dir: Path) -> FastaAuditWriteResult:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "fasta_qc.tsv"
    summary_path = output_dir / "fasta_qc.summary.json"
    fieldnames = [
        "candidate_id",
        "relative_path",
        "source",
        "genus",
        "species",
        "assembly_label",
        "genome_role",
        "size_bytes",
        "mtime_ns",
        "sequence_count",
        "total_symbols",
        "acgt_count",
        "n_count",
        "gc_count",
        "lowercase_count",
        "ambiguous_iupac_count",
        "invalid_symbol_count",
        "duplicate_header_count",
        "empty_sequence_count",
        "min_sequence_length",
        "max_sequence_length",
        "n50",
        "compression",
        "file_sha256",
        "sequence_sha256",
        "duplicate_group",
        "duplicate_group_size",
        "error_type",
        "error",
        "status",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in audit.records:
        row = asdict(record)
        writer.writerow({key: "" if row[key] is None else row[key] for key in fieldnames})
    manifest_bytes = stream.getvalue().encode("utf-8")
    _atomic_write(manifest_path, manifest_bytes)

    summary = dict(audit.summary)
    summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return FastaAuditWriteResult(manifest_path=manifest_path, summary_path=summary_path)

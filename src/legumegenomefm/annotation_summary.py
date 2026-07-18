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
class AnnotationAuditRecord:
    candidate_id: str
    relative_path: str
    source: str
    annotation_role: str
    is_primary_gene_model: bool
    assembly_key: str
    paired_assembly_ids: str
    size_bytes: int
    mtime_ns: int
    format: str
    gff_version: str
    feature_count: int | None
    gene_count: int | None
    transcript_count: int | None
    cds_count: int | None
    exon_count: int | None
    unique_gene_id_count: int | None
    duplicate_gene_id_count: int | None
    unique_transcript_id_count: int | None
    duplicate_transcript_id_count: int | None
    seqid_count: int | None
    malformed_line_count: int | None
    invalid_coordinate_count: int | None
    invalid_strand_count: int | None
    invalid_phase_count: int | None
    embedded_fasta: bool | None
    file_sha256: str
    duplicate_group: str
    duplicate_group_size: int
    error_type: str
    error: str
    status: str


@dataclass(frozen=True)
class AnnotationAudit:
    records: tuple[AnnotationAuditRecord, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class AnnotationAuditWriteResult:
    manifest_path: Path
    summary_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return payload


def aggregate_annotation_audit(registry_path: Path, result_root: Path) -> AnnotationAudit:
    with Path(registry_path).open(newline="", encoding="utf-8") as handle:
        registry = list(csv.DictReader(handle, delimiter="\t"))
    if not registry:
        raise ValueError("annotation registry is empty")
    expected = {row["candidate_id"]: row for row in registry}
    if len(expected) != len(registry):
        raise ValueError("annotation registry contains duplicate IDs")

    result_root = Path(result_root)
    passes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, str]] = {}
    implementation_hashes: set[str] = set()
    for path in sorted((result_root / "annotations").glob("*.json")):
        payload = _read_json(path)
        candidate = payload.get("candidate", {})
        candidate_id = str(candidate.get("candidate_id") or "")
        if payload.get("state") != "PASS" or candidate_id not in expected:
            raise ValueError(f"invalid annotation result: {path.name}")
        if candidate_id in passes:
            raise ValueError(f"duplicate annotation result: {candidate_id}")
        if candidate.get("relative_path") != expected[candidate_id]["relative_path"]:
            raise ValueError(f"annotation path mismatch: {candidate_id}")
        implementation = str(payload.get("implementation_sha256") or "")
        if implementation:
            implementation_hashes.add(implementation)
        passes[candidate_id] = payload
    for path in sorted((result_root / "runs").glob("*.json")):
        payload = _read_json(path)
        implementation = str(payload.get("implementation_sha256") or "")
        if implementation:
            implementation_hashes.add(implementation)
        for failure in payload.get("failures", []):
            candidate_id = str(failure.get("candidate_id") or "")
            if candidate_id not in expected:
                raise ValueError(f"unexpected failed annotation: {candidate_id}")
            failures.setdefault(
                candidate_id,
                {
                    "error_type": str(failure.get("error_type") or "UnknownError"),
                    "error": str(failure.get("error") or ""),
                },
            )
    if len(implementation_hashes) > 1:
        raise ValueError("mixed implementation hashes in annotation audit results")

    file_groups: dict[str, list[str]] = defaultdict(list)
    for candidate_id, payload in passes.items():
        stats = payload.get("stats")
        if not isinstance(stats, dict):
            raise ValueError(f"missing annotation stats: {candidate_id}")
        file_sha = str(stats.get("file_sha256") or "")
        if len(file_sha) != 64:
            raise ValueError(f"invalid annotation file hash: {candidate_id}")
        file_groups[file_sha].append(candidate_id)
    duplicate_metadata: dict[str, tuple[str, int]] = {}
    duplicate_group_count = 0
    duplicate_member_count = 0
    for file_sha, ids in sorted(file_groups.items()):
        if len(ids) < 2:
            continue
        duplicate_group_count += 1
        duplicate_member_count += len(ids)
        group = f"exact_{file_sha[:16]}"
        for candidate_id in ids:
            duplicate_metadata[candidate_id] = (group, len(ids))

    records: list[AnnotationAuditRecord] = []
    for row in registry:
        candidate_id = row["candidate_id"]
        common = {
            "candidate_id": candidate_id,
            "relative_path": row["relative_path"],
            "source": row["source"],
            "annotation_role": row["annotation_role"],
            "is_primary_gene_model": row["is_primary_gene_model"].lower() == "true",
            "assembly_key": row["assembly_key"],
            "paired_assembly_ids": row["paired_assembly_ids"],
            "size_bytes": int(row["size_bytes"]),
            "mtime_ns": int(row["mtime_ns"]),
        }
        if candidate_id in passes:
            stats = passes[candidate_id]["stats"]
            group, group_size = duplicate_metadata.get(candidate_id, ("", 1))
            records.append(
                AnnotationAuditRecord(
                    **common,
                    format=str(stats["format"]),
                    gff_version=str(stats["gff_version"]),
                    feature_count=int(stats["feature_count"]),
                    gene_count=int(stats["gene_count"]),
                    transcript_count=int(stats["transcript_count"]),
                    cds_count=int(stats["cds_count"]),
                    exon_count=int(stats["exon_count"]),
                    unique_gene_id_count=int(stats["unique_gene_id_count"]),
                    duplicate_gene_id_count=int(stats["duplicate_gene_id_count"]),
                    unique_transcript_id_count=int(stats["unique_transcript_id_count"]),
                    duplicate_transcript_id_count=int(stats["duplicate_transcript_id_count"]),
                    seqid_count=int(stats["seqid_count"]),
                    malformed_line_count=int(stats["malformed_line_count"]),
                    invalid_coordinate_count=int(stats["invalid_coordinate_count"]),
                    invalid_strand_count=int(stats["invalid_strand_count"]),
                    invalid_phase_count=int(stats["invalid_phase_count"]),
                    embedded_fasta=bool(stats["embedded_fasta"]),
                    file_sha256=str(stats["file_sha256"]),
                    duplicate_group=group,
                    duplicate_group_size=group_size,
                    error_type="",
                    error="",
                    status="PASS",
                )
            )
        else:
            failure = failures.get(candidate_id)
            records.append(
                AnnotationAuditRecord(
                    **common,
                    format="",
                    gff_version="",
                    feature_count=None,
                    gene_count=None,
                    transcript_count=None,
                    cds_count=None,
                    exon_count=None,
                    unique_gene_id_count=None,
                    duplicate_gene_id_count=None,
                    unique_transcript_id_count=None,
                    duplicate_transcript_id_count=None,
                    seqid_count=None,
                    malformed_line_count=None,
                    invalid_coordinate_count=None,
                    invalid_strand_count=None,
                    invalid_phase_count=None,
                    embedded_fasta=None,
                    file_sha256="",
                    duplicate_group="",
                    duplicate_group_size=0,
                    error_type=failure["error_type"] if failure else "",
                    error=failure["error"] if failure else "",
                    status="FAIL" if failure else "MISSING",
                )
            )

    counts = Counter(record.status for record in records)
    passed = [record for record in records if record.status == "PASS"]
    primary_pass = [record for record in passed if record.is_primary_gene_model]
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "candidate_count": len(records),
        "pass_count": counts["PASS"],
        "fail_count": counts["FAIL"],
        "missing_count": counts["MISSING"],
        "implementation_sha256": next(iter(implementation_hashes), ""),
        "paired_candidate_count": sum(bool(record.paired_assembly_ids) for record in records),
        "primary_gene_model_pass_count": len(primary_pass),
        "primary_gene_model_without_genes_count": sum((record.gene_count or 0) == 0 for record in primary_pass),
        "malformed_record_count": sum((record.malformed_line_count or 0) > 0 for record in passed),
        "invalid_coordinate_record_count": sum((record.invalid_coordinate_count or 0) > 0 for record in passed),
        "duplicate_gene_id_record_count": sum((record.duplicate_gene_id_count or 0) > 0 for record in passed),
        "exact_file_duplicate_group_count": duplicate_group_count,
        "exact_file_duplicate_member_count": duplicate_member_count,
        "total_gene_features_in_primary_models": sum(record.gene_count or 0 for record in primary_pass),
    }
    return AnnotationAudit(records=tuple(records), summary=summary)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_annotation_audit(audit: AnnotationAudit, output_dir: Path) -> AnnotationAuditWriteResult:
    output_dir = Path(output_dir)
    manifest_path = output_dir / "annotation_qc.tsv"
    summary_path = output_dir / "annotation_qc.summary.json"
    fieldnames = list(asdict(audit.records[0]))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in audit.records:
        row = asdict(record)
        writer.writerow({key: "" if value is None else value for key, value in row.items()})
    manifest_bytes = stream.getvalue().encode("utf-8")
    _atomic_write(manifest_path, manifest_bytes)
    summary = dict(audit.summary)
    summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return AnnotationAuditWriteResult(manifest_path=manifest_path, summary_path=summary_path)

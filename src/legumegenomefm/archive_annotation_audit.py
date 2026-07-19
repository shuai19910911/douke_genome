from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

from legumegenomefm.annotation_audit import AnnotationAuditResult, AnnotationStats, scan_annotation
from legumegenomefm.archive_audit import zip_member_is_safe


@dataclass(frozen=True)
class ArchiveAnnotationCandidate:
    candidate_id: str
    archive_relative_path: str
    member_name: str
    material: str
    archive_size_bytes: int
    archive_mtime_ns: int
    archive_sha256: str
    member_uncompressed_bytes: int
    member_crc32_hex: str
    paired_genome_ids: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveAnnotationRegistryResult:
    registry_path: Path
    summary_path: Path
    candidate_count: int


@dataclass(frozen=True)
class ArchiveAnnotationSummaryResult:
    manifest_path: Path
    summary_path: Path
    summary: dict[str, object]


def _rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _true(value: str) -> bool:
    return value.casefold() == "true"


def build_archive_annotation_candidates(
    inventory_path: Path,
    archive_qc_path: Path,
    archive_members_path: Path,
    archive_genome_registry_path: Path,
) -> list[ArchiveAnnotationCandidate]:
    inventory = {row["relative_path"]: row for row in _rows(inventory_path)}
    archive_qc = {
        row["relative_path"]: row
        for row in _rows(archive_qc_path)
        if row.get("status") == "PASS"
    }
    genome_index: dict[str, list[str]] = {}
    for row in _rows(archive_genome_registry_path):
        genome_index.setdefault(row["material"], []).append(row["candidate_id"])
    candidates: list[ArchiveAnnotationCandidate] = []
    for member in _rows(archive_members_path):
        archive_path = member["archive_relative_path"]
        if "/gff/" not in f"/{archive_path.casefold().strip('/')}/":
            continue
        if member.get("member_type") != "annotation" or member.get("status") != "PASS":
            continue
        if not _true(member.get("safe_path", "")) or not _true(member.get("crc_verified", "")):
            continue
        if not zip_member_is_safe(member["member_name"]):
            continue
        source = inventory.get(archive_path)
        archive = archive_qc.get(archive_path)
        if source is None or archive is None:
            continue
        if source.get("kind") != "file" or source.get("file_type") != "archive":
            continue
        if int(source["size_bytes"]) != int(archive["size_bytes"]):
            raise ValueError(f"archive size mismatch: {archive_path}")
        material = archive_path.split("/")[1]
        identity = f"zip-annotation:{archive_path}!{member['member_name']}"
        candidates.append(
            ArchiveAnnotationCandidate(
                candidate_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                archive_relative_path=archive_path,
                member_name=member["member_name"],
                material=material,
                archive_size_bytes=int(source["size_bytes"]),
                archive_mtime_ns=int(source["mtime_ns"]),
                archive_sha256=archive["file_sha256"],
                member_uncompressed_bytes=int(member["uncompressed_bytes"]),
                member_crc32_hex=member["crc32_hex"].casefold(),
                paired_genome_ids=tuple(sorted(genome_index.get(material, []))),
            )
        )
    candidates.sort(key=lambda item: (item.archive_relative_path, item.member_name))
    ids = [item.candidate_id for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("archive annotation candidate ID collision")
    return candidates


def _candidate_row(candidate: ArchiveAnnotationCandidate) -> dict[str, object]:
    row = asdict(candidate)
    row["paired_genome_ids"] = ";".join(candidate.paired_genome_ids) or "."
    return row


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


def write_archive_annotation_registry(
    candidates: Iterable[ArchiveAnnotationCandidate], output_dir: Path
) -> ArchiveAnnotationRegistryResult:
    ordered = sorted(candidates, key=lambda item: (item.archive_relative_path, item.member_name))
    names = [field.name for field in fields(ArchiveAnnotationCandidate)]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=names, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for candidate in ordered:
        writer.writerow(_candidate_row(candidate))
    payload = text.getvalue().encode("utf-8")
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(ordered),
        "paired_candidate_count": sum(bool(item.paired_genome_ids) for item in ordered),
        "total_uncompressed_bytes": sum(item.member_uncompressed_bytes for item in ordered),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output_dir = Path(output_dir)
    registry = output_dir / "archive_annotation_candidates.tsv"
    summary_path = output_dir / "archive_annotation_candidates.summary.json"
    _atomic_write(registry, payload)
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return ArchiveAnnotationRegistryResult(registry, summary_path, len(ordered))


def _candidate_from_row(row: dict[str, str]) -> ArchiveAnnotationCandidate:
    return ArchiveAnnotationCandidate(
        candidate_id=row["candidate_id"],
        archive_relative_path=row["archive_relative_path"],
        member_name=row["member_name"],
        material=row["material"],
        archive_size_bytes=int(row["archive_size_bytes"]),
        archive_mtime_ns=int(row["archive_mtime_ns"]),
        archive_sha256=row["archive_sha256"],
        member_uncompressed_bytes=int(row["member_uncompressed_bytes"]),
        member_crc32_hex=row["member_crc32_hex"],
        paired_genome_ids=(
            tuple(filter(None, row["paired_genome_ids"].split(";")))
            if row["paired_genome_ids"] != "."
            else ()
        ),
    )


def read_archive_annotation_registry(path: Path) -> list[ArchiveAnnotationCandidate]:
    return [_candidate_from_row(row) for row in _rows(path)]


def _confined(data_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("archive path is not confined to data root")
    root = Path(data_root).resolve(strict=True)
    target = (root / candidate).resolve(strict=True)
    if os.path.commonpath((str(root), str(target))) != str(root) or not target.is_file():
        raise ValueError("archive path is not confined to data root")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_archive_annotation(
    data_root: Path,
    candidate: ArchiveAnnotationCandidate,
    output_dir: Path,
    implementation_sha256: str,
    *,
    temporary_dir: Path,
) -> AnnotationAuditResult:
    if len(implementation_sha256) != 64:
        raise ValueError("invalid implementation SHA-256")
    archive_path = _confined(data_root, candidate.archive_relative_path)
    stat = archive_path.stat()
    if stat.st_size != candidate.archive_size_bytes or stat.st_mtime_ns != candidate.archive_mtime_ns:
        raise RuntimeError("archive metadata drift")
    result_path = Path(output_dir) / f"{candidate.candidate_id}.json"
    expected_candidate = asdict(candidate)
    if result_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("state") == "PASS"
            and existing.get("implementation_sha256") == implementation_sha256
            and existing.get("candidate") == json.loads(json.dumps(expected_candidate))
        ):
            return AnnotationAuditResult(result_path, True)
    if _sha256(archive_path) != candidate.archive_sha256:
        raise RuntimeError("archive SHA-256 mismatch")
    Path(temporary_dir).mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(candidate.member_name)
            if info.file_size != candidate.member_uncompressed_bytes or f"{info.CRC:08x}" != candidate.member_crc32_hex:
                raise RuntimeError("ZIP member identity mismatch")
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f"lgfm-gff-{candidate.candidate_id}-", suffix=".gff", dir=temporary_dir, delete=False
            ) as destination:
                temporary_path = Path(destination.name)
                with archive.open(info) as source:
                    shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
        if temporary_path.stat().st_size != candidate.member_uncompressed_bytes:
            raise RuntimeError("extracted ZIP member size mismatch")
        stats = scan_annotation(temporary_path)
        payload = {
            "schema_version": "1.0",
            "state": "PASS",
            "implementation_sha256": implementation_sha256,
            "candidate": expected_candidate,
            "stats": asdict(stats),
        }
        _atomic_write(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        return AnnotationAuditResult(result_path, False)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def aggregate_archive_annotation_audit(
    registry_path: Path, result_dir: Path, output_dir: Path | None = None
) -> ArchiveAnnotationSummaryResult:
    candidates = read_archive_annotation_registry(registry_path)
    result_dir = Path(result_dir)
    expected_ids = {candidate.candidate_id for candidate in candidates}
    observed = {path.stem for path in result_dir.glob("*.json")}
    extras = sorted(observed - expected_ids)
    if extras:
        raise ValueError(f"unexpected archive annotation results: {','.join(extras)}")
    rows: list[dict[str, object]] = []
    implementations: set[str] = set()
    for candidate in candidates:
        path = result_dir / f"{candidate.candidate_id}.json"
        if not path.is_file():
            raise ValueError(f"missing archive annotation result: {candidate.candidate_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("state") != "PASS":
            raise ValueError(f"non-PASS archive annotation result: {candidate.candidate_id}")
        expected = json.loads(json.dumps(asdict(candidate)))
        if payload.get("candidate") != expected:
            raise ValueError(f"archive annotation candidate identity mismatch: {candidate.candidate_id}")
        implementation = payload.get("implementation_sha256", "")
        if not isinstance(implementation, str) or len(implementation) != 64:
            raise ValueError("invalid archive annotation implementation hash")
        implementations.add(implementation)
        stats = payload.get("stats")
        if not isinstance(stats, dict) or set(stats) != {field.name for field in fields(AnnotationStats)}:
            raise ValueError(f"invalid archive annotation stats: {candidate.candidate_id}")
        row: dict[str, object] = _candidate_row(candidate)
        row.update(stats)
        row["feature_counts"] = json.dumps(stats["feature_counts"], sort_keys=True, separators=(",", ":"))
        row["seqid_max_end"] = json.dumps(stats["seqid_max_end"], sort_keys=True, separators=(",", ":"))
        row["implementation_sha256"] = implementation
        row["status"] = "PASS"
        rows.append(row)
    if len(implementations) != 1:
        raise ValueError("mixed archive annotation implementation hashes")
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "candidate_count": len(candidates),
        "pass_count": len(rows),
        "paired_count": sum(bool(candidate.paired_genome_ids) for candidate in candidates),
        "gene_count": sum(int(row["gene_count"]) for row in rows),
        "feature_count": sum(int(row["feature_count"]) for row in rows),
        "malformed_line_count": sum(int(row["malformed_line_count"]) for row in rows),
        "invalid_coordinate_count": sum(int(row["invalid_coordinate_count"]) for row in rows),
        "implementation_sha256": next(iter(implementations), ""),
    }
    target = Path(output_dir) if output_dir is not None else Path(registry_path).parent
    fieldnames = list(rows[0]) if rows else [field.name for field in fields(ArchiveAnnotationCandidate)]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest = target / "archive_annotation_qc.tsv"
    summary_path = target / "archive_annotation_qc.summary.json"
    _atomic_write(manifest, text.getvalue().encode("utf-8"))
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return ArchiveAnnotationSummaryResult(manifest, summary_path, summary)

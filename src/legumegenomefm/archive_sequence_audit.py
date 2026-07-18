from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from legumegenomefm.archive_audit import zip_member_is_safe
from legumegenomefm.assembly_audit import AuditResult, scan_fasta


@dataclass(frozen=True)
class ArchiveGenomeCandidate:
    candidate_id: str
    archive_relative_path: str
    member_name: str
    material: str
    archive_size_bytes: int
    archive_mtime_ns: int
    archive_sha256: str
    member_uncompressed_bytes: int
    member_crc32_hex: str


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _true(value: str) -> bool:
    return value.casefold() == "true"


def build_archive_genome_candidates(
    inventory_path: Path,
    archive_qc_path: Path,
    archive_members_path: Path,
) -> list[ArchiveGenomeCandidate]:
    inventory = {row["relative_path"]: row for row in _read_tsv(inventory_path)}
    archive_qc = {
        row["relative_path"]: row
        for row in _read_tsv(archive_qc_path)
        if row.get("status") == "PASS"
    }
    candidates: list[ArchiveGenomeCandidate] = []
    for member in _read_tsv(archive_members_path):
        archive_path = member["archive_relative_path"]
        normalized = "/" + archive_path.casefold().strip("/") + "/"
        if "/genome/" not in normalized:
            continue
        if member.get("member_type") != "fasta" or member.get("status") != "PASS":
            continue
        if not _true(member.get("safe_path", "")) or not _true(member.get("crc_verified", "")):
            continue
        if not zip_member_is_safe(member["member_name"]):
            continue
        source = inventory.get(archive_path)
        qc = archive_qc.get(archive_path)
        if source is None or qc is None:
            continue
        if source.get("kind") != "file" or source.get("file_type") != "archive":
            continue
        archive_size = int(source["size_bytes"])
        if archive_size != int(qc["size_bytes"]):
            raise ValueError(f"archive size mismatch between manifests: {archive_path}")
        archive_sha = qc["file_sha256"]
        crc32_hex = member["crc32_hex"].casefold()
        if len(archive_sha) != 64 or any(value not in "0123456789abcdef" for value in archive_sha):
            raise ValueError(f"invalid archive SHA-256: {archive_path}")
        if len(crc32_hex) != 8 or any(value not in "0123456789abcdef" for value in crc32_hex):
            raise ValueError(f"invalid member CRC32: {archive_path}")
        parts = archive_path.split("/")
        material = parts[1] if len(parts) > 1 else ""
        identity = f"zip:{archive_path}!{member['member_name']}"
        candidates.append(
            ArchiveGenomeCandidate(
                candidate_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                archive_relative_path=archive_path,
                member_name=member["member_name"],
                material=material,
                archive_size_bytes=archive_size,
                archive_mtime_ns=int(source["mtime_ns"]),
                archive_sha256=archive_sha,
                member_uncompressed_bytes=int(member["uncompressed_bytes"]),
                member_crc32_hex=crc32_hex,
            )
        )
    candidates.sort(key=lambda candidate: (candidate.archive_relative_path, candidate.member_name))
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("archive genome candidate ID collision")
    return candidates


def _confined_archive(data_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("archive path is not confined to data root")
    root = Path(data_root).resolve(strict=True)
    target = (root / candidate).resolve(strict=True)
    if os.path.commonpath((str(root), str(target))) != str(root) or not target.is_file():
        raise ValueError("archive path is not confined to data root")
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_archive_genome(
    data_root: Path,
    candidate: ArchiveGenomeCandidate,
    output_dir: Path,
    implementation_sha256: str,
    *,
    temporary_dir: Path,
) -> AuditResult:
    if len(implementation_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in implementation_sha256
    ):
        raise ValueError("implementation_sha256 must contain 64 hexadecimal characters")
    if not zip_member_is_safe(candidate.member_name):
        raise ValueError("unsafe ZIP member path")
    archive_path = _confined_archive(data_root, candidate.archive_relative_path)
    stat_result = archive_path.stat()
    if (
        stat_result.st_size != candidate.archive_size_bytes
        or stat_result.st_mtime_ns != candidate.archive_mtime_ns
    ):
        raise RuntimeError("archive metadata drift")
    output_dir = Path(output_dir)
    result_json = output_dir / f"{candidate.candidate_id}.json"
    identity = {
        "candidate_id": candidate.candidate_id,
        "archive_relative_path": candidate.archive_relative_path,
        "member_name": candidate.member_name,
        "archive_size_bytes": candidate.archive_size_bytes,
        "archive_mtime_ns": candidate.archive_mtime_ns,
        "archive_sha256": candidate.archive_sha256,
        "member_uncompressed_bytes": candidate.member_uncompressed_bytes,
        "member_crc32_hex": candidate.member_crc32_hex,
        "implementation_sha256": implementation_sha256,
    }
    if result_json.is_file():
        try:
            existing = json.loads(result_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("state") == "PASS" and all(
            existing.get(key) == value for key, value in identity.items()
        ):
            return AuditResult(result_json, True)
    if _sha256_file(archive_path) != candidate.archive_sha256:
        raise RuntimeError("archive SHA-256 mismatch")

    Path(temporary_dir).mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with zipfile.ZipFile(archive_path) as archive:
            info = archive.getinfo(candidate.member_name)
            if info.file_size != candidate.member_uncompressed_bytes:
                raise RuntimeError("ZIP member size mismatch")
            if f"{info.CRC:08x}" != candidate.member_crc32_hex:
                raise RuntimeError("ZIP member CRC32 mismatch")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f"lgfm-{candidate.candidate_id}-",
                suffix=".fna",
                dir=temporary_dir,
                delete=False,
            ) as temporary_handle:
                temporary_path = Path(temporary_handle.name)
                with archive.open(info, "r") as member_handle:
                    shutil.copyfileobj(member_handle, temporary_handle, length=8 * 1024 * 1024)
        if temporary_path.stat().st_size != candidate.member_uncompressed_bytes:
            raise RuntimeError("extracted ZIP member size mismatch")
        stats = scan_fasta(temporary_path)
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "state": "PASS",
            **identity,
            "source": "soyod_zip",
            "material": candidate.material,
            "stats": asdict(stats),
        }
        _atomic_json(result_json, payload)
        return AuditResult(result_json, False)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

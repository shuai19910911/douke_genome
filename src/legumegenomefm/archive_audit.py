from __future__ import annotations

import hashlib
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fas")
_PROTEIN_SUFFIXES = (".faa", ".pep")
_ANNOTATION_SUFFIXES = (".gff", ".gff3", ".gtf")
_TABLE_SUFFIXES = (".tsv", ".csv", ".xlsx", ".xls")


@dataclass(frozen=True)
class ZipMemberAudit:
    member_name: str
    member_type: str
    uncompressed_bytes: int
    compressed_bytes: int
    crc32_hex: str
    compression_method: int
    encrypted: bool
    safe_path: bool
    compression_ratio: float
    crc_verified: bool
    error_type: str
    error_message: str


@dataclass(frozen=True)
class ZipArchiveAudit:
    file_size_bytes: int
    file_sha256: str
    member_count: int
    file_member_count: int
    total_uncompressed_bytes: int
    total_compressed_member_bytes: int
    encrypted_member_count: int
    unsafe_member_count: int
    high_compression_ratio_count: int
    duplicate_member_name_count: int
    crc_verified_count: int
    crc_failure_count: int
    counts_by_member_type: dict[str, int]
    members: tuple[ZipMemberAudit, ...]


def _strip_gzip_suffix(value: str) -> str:
    return value[:-3] if value.casefold().endswith(".gz") else value


def classify_member(member_name: str) -> str:
    lower = _strip_gzip_suffix(member_name.casefold())
    if lower.endswith(_PROTEIN_SUFFIXES):
        return "protein_fasta"
    if lower.endswith(_ANNOTATION_SUFFIXES):
        return "annotation"
    if lower.endswith(_FASTA_SUFFIXES):
        return "fasta"
    if lower.endswith(_TABLE_SUFFIXES):
        return "table"
    return "other"


def zip_member_is_safe(member_name: str) -> bool:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return False
    if path.parts and re.match(r"^[A-Za-z]:$", path.parts[0]):
        return False
    return bool(path.parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scan_zip_archive(path: Path, *, verify_crc: bool) -> ZipArchiveAudit:
    path = Path(path)
    file_sha256 = _sha256_file(path)
    members: list[ZipMemberAudit] = []
    with zipfile.ZipFile(path) as archive:
        infos = sorted(archive.infolist(), key=lambda info: info.filename)
        for info in infos:
            if info.is_dir():
                continue
            encrypted = bool(info.flag_bits & 0x1)
            crc_verified = False
            error_type = ""
            error_message = ""
            if verify_crc and not encrypted:
                try:
                    with archive.open(info, "r") as member:
                        while member.read(4 * 1024 * 1024):
                            pass
                    crc_verified = True
                except (zipfile.BadZipFile, RuntimeError, EOFError, OSError) as error:
                    error_type = type(error).__name__
                    error_message = str(error)
            ratio = info.file_size / max(info.compress_size, 1)
            members.append(
                ZipMemberAudit(
                    member_name=info.filename,
                    member_type=classify_member(info.filename),
                    uncompressed_bytes=info.file_size,
                    compressed_bytes=info.compress_size,
                    crc32_hex=f"{info.CRC:08x}",
                    compression_method=info.compress_type,
                    encrypted=encrypted,
                    safe_path=zip_member_is_safe(info.filename),
                    compression_ratio=round(ratio, 6),
                    crc_verified=crc_verified,
                    error_type=error_type,
                    error_message=error_message,
                )
            )
    name_counts = Counter(member.member_name for member in members)
    return ZipArchiveAudit(
        file_size_bytes=path.stat().st_size,
        file_sha256=file_sha256,
        member_count=len(infos),
        file_member_count=len(members),
        total_uncompressed_bytes=sum(member.uncompressed_bytes for member in members),
        total_compressed_member_bytes=sum(member.compressed_bytes for member in members),
        encrypted_member_count=sum(member.encrypted for member in members),
        unsafe_member_count=sum(not member.safe_path for member in members),
        high_compression_ratio_count=sum(member.compression_ratio > 1000 for member in members),
        duplicate_member_name_count=sum(count - 1 for count in name_counts.values() if count > 1),
        crc_verified_count=sum(member.crc_verified for member in members),
        crc_failure_count=sum(bool(member.error_type) for member in members),
        counts_by_member_type=dict(sorted(Counter(member.member_type for member in members).items())),
        members=tuple(members),
    )

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_FASTA_SUFFIXES = (".fa", ".fasta", ".fna", ".fas")
_ANNOTATION_SUFFIXES = (".gff", ".gff3", ".gtf")
_VARIANT_SUFFIXES = (".vcf", ".bcf")
_INTERVAL_SUFFIXES = (".bed", ".bedgraph", ".bigwig", ".bw")
_CHECKSUM_SUFFIXES = (".md5", ".sha1", ".sha256", ".checksums")
_ARCHIVE_SUFFIXES = (".tar", ".zip", ".tgz", ".tbz2", ".xz", ".7z")
_COMPRESSION_SUFFIXES = (".gz", ".bgz", ".bz2", ".zst")


@dataclass(frozen=True)
class FileRecord:
    relative_path: str
    kind: str
    file_type: str
    size_bytes: int
    mtime_ns: int
    mode_octal: str
    link_target: str


@dataclass(frozen=True)
class InventoryResult:
    inventory_tsv: Path
    summary_json: Path
    inventory_sha256: str


def classify_path(path: Path) -> str:
    name = path.name.lower()
    uncompressed = name
    for suffix in _COMPRESSION_SUFFIXES:
        if uncompressed.endswith(suffix):
            uncompressed = uncompressed[: -len(suffix)]
            break
    if uncompressed.endswith(_FASTA_SUFFIXES):
        return "fasta"
    if uncompressed.endswith(_ANNOTATION_SUFFIXES):
        return "annotation"
    if uncompressed.endswith(_VARIANT_SUFFIXES):
        return "variant"
    if uncompressed.endswith(_INTERVAL_SUFFIXES):
        return "interval"
    if name.endswith(_CHECKSUM_SUFFIXES):
        return "checksum"
    if uncompressed.endswith(_ARCHIVE_SUFFIXES) or name.endswith((".tar.gz", ".tar.bz2", ".tar.zst")):
        return "archive"
    return "other"


def _safe_link_target(path: Path) -> str:
    target = os.readlink(path)
    if os.path.isabs(target):
        digest = hashlib.sha256(target.encode("utf-8", "surrogateescape")).hexdigest()
        return f"<absolute:path-sha256:{digest}>"
    return target


def scan_tree(root: Path) -> list[FileRecord]:
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"data root must be a real directory: {root}")
    records: list[FileRecord] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise RuntimeError(f"cannot scan directory: {directory}") from exc
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
                relative = path.relative_to(root).as_posix()
                mode = stat.filemode(info.st_mode)
                if entry.is_symlink():
                    records.append(
                        FileRecord(relative, "symlink", classify_path(path), 0, info.st_mtime_ns, mode, _safe_link_target(path))
                    )
                elif entry.is_dir(follow_symlinks=False):
                    visit(path)
                elif entry.is_file(follow_symlinks=False):
                    records.append(
                        FileRecord(relative, "file", classify_path(path), info.st_size, info.st_mtime_ns, mode, "")
                    )
                else:
                    records.append(FileRecord(relative, "special", "other", 0, info.st_mtime_ns, mode, ""))
            except OSError as exc:
                raise RuntimeError(f"cannot inspect path: {path}") from exc

    visit(root)
    return records


def _render_tsv(records: Iterable[FileRecord]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["relative_path", "kind", "file_type", "size_bytes", "mtime_ns", "link_target", "mode_octal"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        writer.writerow(asdict(record))
    return buffer.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
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


def write_inventory(data_root: Path, output_dir: Path) -> InventoryResult:
    records = scan_tree(Path(data_root))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_tsv = output_dir / "raw_inventory.tsv"
    summary_json = output_dir / "raw_inventory.summary.json"
    tsv_payload = _render_tsv(records)
    inventory_sha256 = hashlib.sha256(tsv_payload).hexdigest()
    regular_files = [record for record in records if record.kind == "file"]
    counts = Counter(record.file_type for record in regular_files)
    summary = {
        "schema_version": "1.0",
        "inventory_sha256": inventory_sha256,
        "record_count": len(records),
        "file_count": len(regular_files),
        "symlink_count": sum(record.kind == "symlink" for record in records),
        "special_count": sum(record.kind == "special" for record in records),
        "total_bytes": sum(record.size_bytes for record in regular_files),
        "counts_by_type": dict(sorted(counts.items())),
    }
    summary_payload = (json.dumps(summary, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_write(inventory_tsv, tsv_payload)
    _atomic_write(summary_json, summary_payload)
    directory_fd = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return InventoryResult(inventory_tsv, summary_json, inventory_sha256)

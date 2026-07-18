from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator


@dataclass(frozen=True)
class AssemblyCandidate:
    candidate_id: str
    relative_path: str
    source: str
    genus: str
    species: str
    assembly_label: str
    genome_role: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class RegistryResult:
    registry_tsv: Path
    summary_json: Path
    registry_sha256: str
    candidate_count: int


@dataclass(frozen=True)
class ShardManifestResult:
    shard_paths: tuple[Path, ...]
    summary_json: Path
    shard_count: int
    candidate_count: int


@dataclass(frozen=True)
class AuditResult:
    result_json: Path
    reused: bool


@dataclass(frozen=True)
class FastaStats:
    compression: str
    file_size_bytes: int
    file_sha256: str
    canonical_sequence_sha256: str
    sequence_count: int
    duplicate_header_count: int
    empty_sequence_count: int
    total_symbols: int
    min_sequence_length: int
    max_sequence_length: int
    n50: int
    acgt_count: int
    gc_count: int
    n_count: int
    iupac_ambiguous_count: int
    invalid_symbol_count: int
    lowercase_count: int
    symbol_counts: dict[str, int]


def _genome_role(relative_path: str) -> str:
    value = relative_path.casefold()
    if "softmask" in value:
        return "softmasked"
    if "genome_main" in value:
        return "main"
    if "_genomic.fna" in value:
        return "ncbi_genomic"
    return "generic_genome"


def _source_metadata(relative_path: str) -> tuple[str, str, str, str]:
    parts = relative_path.split("/")
    if parts[:2] == ["legume_family", "legumeinfo"] and len(parts) >= 5:
        genus, epithet, assembly = parts[2], parts[3], parts[4]
        return "legume_family_legumeinfo", genus, f"{genus} {epithet.replace('_', ' ')}", assembly
    if parts[:2] == ["legume_family", "ncbi"] and len(parts) >= 4:
        species_token, assembly = parts[2], parts[3]
        words = species_token.replace("_", " ").split()
        genus = words[0] if words else ""
        return "legume_family_ncbi", genus, " ".join(words), assembly
    if parts and parts[0] == "legumeinfo":
        return "legumeinfo", "", "", parts[1] if len(parts) > 1 else ""
    if parts and parts[0] == "soyomics":
        return "soyomics", "", "", parts[1] if len(parts) > 1 else ""
    if parts and parts[0] == "soyod":
        return "soyod", "", "", parts[1] if len(parts) > 1 else ""
    return parts[0] if parts else "unknown", "", "", ""


def _is_genome_fasta(row: dict[str, str]) -> bool:
    if row.get("kind") != "file" or row.get("file_type") != "fasta":
        return False
    normalized = "/" + row.get("relative_path", "").casefold().strip("/") + "/"
    return "/genome/" in normalized


def build_assembly_candidates(inventory_tsv: Path) -> list[AssemblyCandidate]:
    candidates: list[AssemblyCandidate] = []
    with Path(inventory_tsv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if not _is_genome_fasta(row):
                continue
            relative_path = row["relative_path"]
            source, genus, species, assembly_label = _source_metadata(relative_path)
            candidate_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
            candidates.append(
                AssemblyCandidate(
                    candidate_id=candidate_id,
                    relative_path=relative_path,
                    source=source,
                    genus=genus,
                    species=species,
                    assembly_label=assembly_label,
                    genome_role=_genome_role(relative_path),
                    size_bytes=int(row["size_bytes"]),
                    mtime_ns=int(row["mtime_ns"]),
                )
            )
    candidates.sort(key=lambda value: value.relative_path)
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("candidate ID collision")
    return candidates


def lpt_shards(candidates: Iterable[AssemblyCandidate], shard_count: int) -> list[list[AssemblyCandidate]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[AssemblyCandidate]] = [[] for _ in range(shard_count)]
    totals = [0 for _ in range(shard_count)]
    for candidate in sorted(candidates, key=lambda value: (-value.size_bytes, value.candidate_id)):
        index = min(range(shard_count), key=lambda value: (totals[value], value))
        shards[index].append(candidate)
        totals[index] += candidate.size_bytes
    for shard in shards:
        shard.sort(key=lambda value: value.relative_path)
    return shards


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


def write_assembly_registry(candidates: Iterable[AssemblyCandidate], output_dir: Path) -> RegistryResult:
    values = sorted(candidates, key=lambda value: value.relative_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_tsv = output_dir / "assembly_candidates.tsv"
    summary_json = output_dir / "assembly_candidates.summary.json"
    fields = [
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
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for value in values:
        writer.writerow(asdict(value))
    payload = buffer.getvalue().encode("utf-8")
    registry_sha256 = hashlib.sha256(payload).hexdigest()
    role_counts = Counter(value.genome_role for value in values)
    source_counts = Counter(value.source for value in values)
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(values),
        "total_compressed_bytes": sum(value.size_bytes for value in values),
        "registry_sha256": registry_sha256,
        "counts_by_role": dict(sorted(role_counts.items())),
        "counts_by_source": dict(sorted(source_counts.items())),
    }
    _atomic_write(registry_tsv, payload)
    _atomic_write(summary_json, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return RegistryResult(registry_tsv, summary_json, registry_sha256, len(values))


def write_shard_manifests(
    candidates: Iterable[AssemblyCandidate], output_dir: Path, shard_count: int
) -> ShardManifestResult:
    values = list(candidates)
    shards = lpt_shards(values, shard_count)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
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
    paths: list[Path] = []
    summary_rows: list[dict[str, object]] = []
    observed_ids: list[str] = []
    for index, shard in enumerate(shards):
        path = output_dir / f"shard_{index:02d}.tsv"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for candidate in shard:
            writer.writerow(asdict(candidate))
            observed_ids.append(candidate.candidate_id)
        payload = buffer.getvalue().encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_write(path, payload)
        paths.append(path)
        summary_rows.append(
            {
                "shard_id": index,
                "path": path.name,
                "candidate_count": len(shard),
                "compressed_bytes": sum(candidate.size_bytes for candidate in shard),
                "sha256": digest,
            }
        )
    expected_ids = sorted(candidate.candidate_id for candidate in values)
    if sorted(observed_ids) != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise RuntimeError("shard partition is not an exact candidate partition")
    summary_json = output_dir / "shards.summary.json"
    summary = {
        "schema_version": "1.0",
        "shard_count": shard_count,
        "candidate_count": len(values),
        "total_compressed_bytes": sum(candidate.size_bytes for candidate in values),
        "shards": summary_rows,
    }
    _atomic_write(summary_json, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return ShardManifestResult(tuple(paths), summary_json, shard_count, len(values))


def _confined_target(data_root: Path, relative_path: str) -> Path:
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise ValueError("candidate path is not confined to data root")
    root = Path(data_root).resolve(strict=True)
    target = (root / candidate_path).resolve(strict=True)
    if os.path.commonpath((str(root), str(target))) != str(root):
        raise ValueError("candidate path is not confined to data root")
    if not target.is_file():
        raise ValueError("candidate target is not a file")
    return target


def audit_candidate(
    data_root: Path,
    candidate: AssemblyCandidate,
    output_dir: Path,
    implementation_sha256: str,
) -> AuditResult:
    if len(implementation_sha256) != 64:
        raise ValueError("implementation_sha256 must contain 64 hexadecimal characters")
    target = _confined_target(Path(data_root), candidate.relative_path)
    info = target.stat()
    if info.st_size != candidate.size_bytes or info.st_mtime_ns != candidate.mtime_ns:
        raise RuntimeError(f"candidate metadata drift: {candidate.relative_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / f"{candidate.candidate_id}.json"
    identity = {
        "candidate_id": candidate.candidate_id,
        "relative_path": candidate.relative_path,
        "input_size_bytes": candidate.size_bytes,
        "input_mtime_ns": candidate.mtime_ns,
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
    stats = scan_fasta(target)
    payload = {
        "schema_version": "1.0",
        "state": "PASS",
        **identity,
        "source": candidate.source,
        "genus": candidate.genus,
        "species": candidate.species,
        "assembly_label": candidate.assembly_label,
        "genome_role": candidate.genome_role,
        "stats": asdict(stats),
    }
    _atomic_write(result_json, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return AuditResult(result_json, False)


class _HashingReader:
    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.digest.update(data)
        return data

    def readline(self, size: int = -1) -> bytes:
        data = self.raw.readline(size)
        self.digest.update(data)
        return data

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        return self.raw.seek(offset, whence)

    def tell(self) -> int:
        return self.raw.tell()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return self.raw.seekable()

    def close(self) -> None:
        self.raw.close()


_DNA_STANDARD = set(b"ACGT")
_DNA_AMBIGUOUS = set(b"RYSWKMBDHV")
_DNA_VALID = _DNA_STANDARD | _DNA_AMBIGUOUS | {ord("N")}


def _n50(lengths: list[int], total: int) -> int:
    if not lengths or total == 0:
        return 0
    threshold = (total + 1) // 2
    cumulative = 0
    for length in sorted(lengths, reverse=True):
        cumulative += length
        if cumulative >= threshold:
            return length
    raise AssertionError("unreachable N50 state")


def scan_fasta(path: Path) -> FastaStats:
    path = Path(path)
    file_size = path.stat().st_size
    file_digest: str
    canonical = hashlib.sha256()
    lengths: list[int] = []
    headers: set[str] = set()
    duplicate_headers = 0
    symbol_counts: Counter[str] = Counter()
    lowercase_count = 0
    current_length: int | None = None

    with path.open("rb") as raw:
        magic = raw.read(2)
        raw.seek(0)
        hashing_reader = _HashingReader(raw)
        compression = "gzip" if magic == b"\x1f\x8b" else "none"
        stream: BinaryIO
        if compression == "gzip":
            stream = gzip.GzipFile(fileobj=hashing_reader, mode="rb")
        else:
            stream = hashing_reader  # type: ignore[assignment]
        try:
            for line_number, raw_line in enumerate(stream, start=1):
                if raw_line.startswith(b">"):
                    if current_length is not None:
                        lengths.append(current_length)
                    header_text = raw_line[1:].strip().decode("utf-8", "replace")
                    header_id = header_text.split(None, 1)[0] if header_text else ""
                    if header_id in headers:
                        duplicate_headers += 1
                    headers.add(header_id)
                    current_length = 0
                    canonical.update(b"\x00")
                    continue
                sequence = b"".join(raw_line.split())
                if not sequence:
                    continue
                if current_length is None:
                    raise ValueError(f"sequence before first header at line {line_number}")
                lowercase_count += sum(97 <= value <= 122 for value in sequence)
                upper = sequence.upper()
                canonical.update(upper)
                current_length += len(upper)
                symbol_counts.update(chr(value) for value in upper)
            if current_length is not None:
                lengths.append(current_length)
        finally:
            if compression == "gzip":
                stream.close()
        file_digest = hashing_reader.digest.hexdigest()

    if not lengths:
        raise ValueError("FASTA contains no headers")
    total = sum(lengths)
    acgt_count = sum(symbol_counts.get(chr(value), 0) for value in _DNA_STANDARD)
    ambiguous_count = sum(symbol_counts.get(chr(value), 0) for value in _DNA_AMBIGUOUS)
    n_count = symbol_counts.get("N", 0)
    valid_count = acgt_count + ambiguous_count + n_count
    return FastaStats(
        compression=compression,
        file_size_bytes=file_size,
        file_sha256=file_digest,
        canonical_sequence_sha256=canonical.hexdigest(),
        sequence_count=len(lengths),
        duplicate_header_count=duplicate_headers,
        empty_sequence_count=sum(length == 0 for length in lengths),
        total_symbols=total,
        min_sequence_length=min(lengths),
        max_sequence_length=max(lengths),
        n50=_n50(lengths, total),
        acgt_count=acgt_count,
        gc_count=symbol_counts.get("G", 0) + symbol_counts.get("C", 0),
        n_count=n_count,
        iupac_ambiguous_count=ambiguous_count,
        invalid_symbol_count=total - valid_count,
        lowercase_count=lowercase_count,
        symbol_counts=dict(sorted(symbol_counts.items())),
    )

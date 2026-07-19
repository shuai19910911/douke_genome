from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import BinaryIO, Iterator

from sourmash import MinHash, SourmashSignature
from sourmash.save_load import SaveSignaturesToLocation

from legumegenomefm.sequence_store import PackedSequenceStore, PackedSequenceStoreWriter


@dataclass(frozen=True)
class GenomeSketchCandidate:
    candidate_id: str
    source_kind: str
    relative_path: str
    member_name: str
    container_size_bytes: int
    container_sha256: str
    payload_sha256: str
    sequence_sha256: str
    species: str
    material_key: str


@dataclass(frozen=True)
class GenomeSketchResult:
    result_path: Path
    signature_path: Path
    reused: bool


@dataclass(frozen=True)
class GenomeSketchRegistryResult:
    registry_path: Path
    summary_path: Path
    candidate_count: int


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

    def __iter__(self) -> _HashingReader:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if line:
            return line
        raise StopIteration

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def _rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_sketch_candidates(catalog_path: Path) -> list[GenomeSketchCandidate]:
    candidates: list[GenomeSketchCandidate] = []
    for row in _rows(catalog_path):
        if row.get("status") != "PASS" or row.get("exact_representative") != "True":
            continue
        if "hardmasked" in row["relative_path"].casefold():
            continue
        total_symbols = int(row["total_symbols"])
        if total_symbols <= 0 or int(row["n_count"]) / total_symbols > 0.20:
            continue
        candidates.append(
            GenomeSketchCandidate(
                candidate_id=row["candidate_id"],
                source_kind=row["source_kind"],
                relative_path=row["relative_path"],
                member_name=row["member_name"],
                container_size_bytes=int(row["container_size_bytes"]),
                container_sha256=row["container_sha256"],
                payload_sha256=row["payload_sha256"],
                sequence_sha256=row["sequence_sha256"],
                species=row["species"],
                material_key=row["material_key"],
            )
        )
    candidates.sort(key=lambda item: (item.relative_path, item.member_name, item.candidate_id))
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate genome sketch candidate ID")
    return candidates


def write_sketch_registry(
    candidates: list[GenomeSketchCandidate], output_dir: Path
) -> GenomeSketchRegistryResult:
    ordered = sorted(candidates, key=lambda item: (item.relative_path, item.member_name, item.candidate_id))
    text = io.StringIO(newline="")
    writer = csv.DictWriter(
        text,
        fieldnames=[field.name for field in fields(GenomeSketchCandidate)],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for candidate in ordered:
        writer.writerow(asdict(candidate))
    payload = text.getvalue().encode("utf-8")
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(ordered),
        "container_bytes": sum(candidate.container_size_bytes for candidate in ordered),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output_dir = Path(output_dir)
    registry_path = output_dir / "genome_sketch_candidates.tsv"
    summary_path = output_dir / "genome_sketch_candidates.summary.json"
    _atomic_bytes(registry_path, payload)
    _atomic_bytes(
        summary_path,
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return GenomeSketchRegistryResult(registry_path, summary_path, len(ordered))


def read_sketch_registry(path: Path) -> list[GenomeSketchCandidate]:
    return [
        GenomeSketchCandidate(
            candidate_id=row["candidate_id"],
            source_kind=row["source_kind"],
            relative_path=row["relative_path"],
            member_name=row["member_name"],
            container_size_bytes=int(row["container_size_bytes"]),
            container_sha256=row["container_sha256"],
            payload_sha256=row["payload_sha256"],
            sequence_sha256=row["sequence_sha256"],
            species=row["species"],
            material_key=row["material_key"],
        )
        for row in _rows(path)
    ]


def _confined(data_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("source path is not confined to data root")
    root = Path(data_root).resolve(strict=True)
    target = (root / candidate).resolve(strict=True)
    if os.path.commonpath((str(root), str(target))) != str(root) or not target.is_file():
        raise ValueError("source path is not confined to data root")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feed_fasta(
    lines: Iterator[bytes],
    minhash: MinHash,
    *,
    ksize: int,
    block_size: int = 4 * 1024 * 1024,
    store_writer: PackedSequenceStoreWriter | None = None,
) -> tuple[int, int]:
    tail = b""
    buffer = bytearray()
    in_sequence = False
    sequence_count = 0
    total_symbols = 0

    def feed_buffer(final: bool) -> None:
        nonlocal tail
        while len(buffer) >= block_size or (final and buffer):
            take = len(buffer) if final and len(buffer) < block_size else block_size
            chunk = bytes(buffer[:take])
            del buffer[:take]
            combined = tail + chunk
            if len(combined) >= ksize:
                minhash.add_sequence(combined.decode("ascii"), force=True)
            tail = combined[-(ksize - 1) :] if ksize > 1 else b""

    for raw_line in lines:
        if raw_line.startswith(b">"):
            if in_sequence:
                feed_buffer(True)
            name = raw_line[1:].strip().split(None, 1)[0].decode("utf-8", errors="replace")
            if store_writer is not None:
                store_writer.start_contig(name)
            sequence_count += 1
            in_sequence = True
            tail = b""
            buffer.clear()
            continue
        sequence = b"".join(raw_line.split()).upper()
        if not sequence:
            continue
        if not in_sequence:
            raise ValueError("FASTA sequence encountered before first header")
        total_symbols += len(sequence)
        if store_writer is not None:
            store_writer.add_bases(sequence)
        buffer.extend(sequence)
        feed_buffer(False)
    if in_sequence:
        feed_buffer(True)
    if sequence_count == 0:
        raise ValueError("FASTA contains no sequence records")
    return sequence_count, total_symbols


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def audit_genome_sketch(
    data_root: Path,
    candidate: GenomeSketchCandidate,
    output_dir: Path,
    implementation_sha256: str,
    *,
    ksize: int = 31,
    scaled: int = 10_000,
    store_root: Path | None = None,
) -> GenomeSketchResult:
    if ksize < 1 or scaled < 1:
        raise ValueError("ksize and scaled must be positive")
    source = _confined(data_root, candidate.relative_path)
    stat = source.stat()
    if stat.st_size != candidate.container_size_bytes:
        raise RuntimeError("genome sketch source size drift")
    output_dir = Path(output_dir)
    result_path = output_dir / f"{candidate.candidate_id}.json"
    signature_path = output_dir / f"{candidate.candidate_id}.sig.gz"
    identity = {
        **asdict(candidate),
        "implementation_sha256": implementation_sha256,
        "ksize": ksize,
        "scaled": scaled,
        "store_enabled": store_root is not None,
    }
    store_target = Path(store_root) / candidate.candidate_id if store_root is not None else None
    store_valid = False
    if store_target is not None and store_target.is_dir():
        try:
            store = PackedSequenceStore(store_target)
            store_valid = store.manifest.get("identity") == identity
        except (OSError, ValueError, json.JSONDecodeError):
            store_valid = False
    if result_path.is_file() and signature_path.is_file():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("state") == "PASS"
            and all(existing.get(key) == value for key, value in identity.items())
            and existing.get("signature_sha256") == _sha256(signature_path)
            and (store_target is None or store_valid)
        ):
            return GenomeSketchResult(result_path, signature_path, True)

    minhash = MinHash(n=0, ksize=ksize, scaled=scaled, track_abundance=False)
    store_writer = None
    if store_target is not None:
        if store_target.exists():
            shutil.rmtree(store_target)
        store_writer = PackedSequenceStoreWriter(store_target, identity)
    observed_payload_sha = ""
    try:
        if candidate.source_kind == "file":
            with source.open("rb") as raw:
                magic = raw.read(2)
                raw.seek(0)
                hashing_raw = _HashingReader(raw)
                if magic == b"\x1f\x8b":
                    stream = gzip.GzipFile(fileobj=hashing_raw, mode="rb")
                else:
                    stream = hashing_raw
                sequence_count, total_symbols = _feed_fasta(
                    iter(stream), minhash, ksize=ksize, store_writer=store_writer
                )
                observed_payload_sha = hashing_raw.digest.hexdigest()
            if observed_payload_sha != candidate.container_sha256 or observed_payload_sha != candidate.payload_sha256:
                raise RuntimeError("genome file SHA-256 mismatch")
        elif candidate.source_kind == "zip_member":
            if _sha256(source) != candidate.container_sha256:
                raise RuntimeError("genome archive SHA-256 mismatch")
            with zipfile.ZipFile(source) as archive:
                with archive.open(candidate.member_name) as member:
                    hashing_member = _HashingReader(member)
                    sequence_count, total_symbols = _feed_fasta(
                        iter(hashing_member), minhash, ksize=ksize, store_writer=store_writer
                    )
                    observed_payload_sha = hashing_member.digest.hexdigest()
            if observed_payload_sha != candidate.payload_sha256:
                raise RuntimeError("genome ZIP member SHA-256 mismatch")
        else:
            raise ValueError(f"unsupported genome source kind: {candidate.source_kind}")
        store_result = store_writer.finalize() if store_writer is not None else None
    except Exception:
        if store_writer is not None and not store_writer.closed:
            store_writer.abort()
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{candidate.candidate_id}.", suffix=".sig.gz", dir=output_dir)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        with SaveSignaturesToLocation(str(temporary_path)) as destination:
            destination.add(
                SourmashSignature(
                    minhash,
                    name=candidate.candidate_id,
                    filename=candidate.relative_path,
                )
            )
        os.replace(temporary_path, signature_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    signature_sha = _sha256(signature_path)
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "state": "PASS",
        **identity,
        "observed_payload_sha256": observed_payload_sha,
        "sequence_count": sequence_count,
        "total_symbols": total_symbols,
        "hash_count": len(minhash.hashes),
        "signature_sha256": signature_sha,
        "store_manifest_sha256": store_result.manifest_sha256 if store_result is not None else "",
        "store_packed_sha256": store_result.packed_sha256 if store_result is not None else "",
    }
    _atomic_json(result_path, payload)
    return GenomeSketchResult(result_path, signature_path, False)

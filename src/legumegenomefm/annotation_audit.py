from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


@dataclass(frozen=True)
class AnnotationCandidate:
    candidate_id: str
    relative_path: str
    source: str
    annotation_role: str
    is_primary_gene_model: bool
    assembly_key: str
    paired_assembly_ids: tuple[str, ...]
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class AnnotationRegistryResult:
    registry_path: Path
    summary_path: Path
    candidate_count: int


@dataclass(frozen=True)
class AnnotationShardResult:
    shard_paths: tuple[Path, ...]
    summary_path: Path
    shard_count: int
    candidate_count: int


@dataclass(frozen=True)
class AnnotationAuditResult:
    result_path: Path
    reused: bool


@dataclass(frozen=True)
class AnnotationStats:
    compression: str
    file_size_bytes: int
    file_sha256: str
    format: str
    gff_version: str
    feature_count: int
    comment_line_count: int
    malformed_line_count: int
    invalid_coordinate_count: int
    invalid_strand_count: int
    invalid_phase_count: int
    embedded_fasta: bool
    feature_counts: dict[str, int]
    gene_count: int
    transcript_count: int
    cds_count: int
    exon_count: int
    unique_gene_id_count: int
    duplicate_gene_id_count: int
    unique_transcript_id_count: int
    duplicate_transcript_id_count: int
    seqid_count: int
    seqid_max_end: dict[str, int]


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

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return True


def classify_annotation_role(path: str) -> tuple[str, bool]:
    low = path.casefold()
    base = Path(low).name
    if "gene_models_main" in base:
        return "gene_models_main", True
    if base.endswith(("_genomic.gff.gz", "_genomic.gff3.gz", "genomic.gff.gz", "genomic.gff3.gz")):
        return "ncbi_genomic", True
    if "/gff/" in f"/{low}" and (".gene.gff" in base or ".gene_models.gff" in base):
        return "soyomics_gene", True
    if "iprscan" in base:
        return "functional_iprscan", False
    if "repeat" in base:
        return "repeat", False
    if "noncoding" in base or "ncrna" in base.casefold():
        return "noncoding", False
    if "gene_models_low" in base or "low_quality_genes" in base:
        return "low_quality_gene_model", False
    if "gene_models_exons" in base:
        return "gene_model_exons", False
    if "gcv_genome" in base:
        return "gcv_genome", False
    if "gcv_genes" in base:
        return "gcv_genes", False
    return "other_annotation", False


def _assembly_stem(label: str) -> str:
    match = re.search(r"^(.+?\.gnm[^.]+)", label)
    return match.group(1) if match else label


def _assembly_pair_key(row: dict[str, str]) -> str:
    path_parts = row["relative_path"].split("/")
    source = row["source"]
    if source == "legume_family_legumeinfo":
        return f"{source}|{path_parts[2]}|{path_parts[3]}|{_assembly_stem(row['assembly_label'])}"
    if source == "legume_family_ncbi":
        return f"{source}|{path_parts[2]}|{row['assembly_label']}"
    if source == "legumeinfo":
        return f"{source}|{_assembly_stem(row['assembly_label'])}"
    if source == "soyomics":
        return f"{source}|{row['assembly_label']}"
    return f"{source}|{_assembly_stem(row['assembly_label'])}"


def _annotation_source_and_key(relative_path: str) -> tuple[str, str]:
    parts = relative_path.split("/")
    if parts[:2] == ["legume_family", "legumeinfo"] and len(parts) > 4:
        source = "legume_family_legumeinfo"
        return source, f"{source}|{parts[2]}|{parts[3]}|{_assembly_stem(parts[4])}"
    if parts[:2] == ["legume_family", "ncbi"] and len(parts) > 3:
        source = "legume_family_ncbi"
        return source, f"{source}|{parts[2]}|{parts[3]}"
    if parts[0] == "legumeinfo" and len(parts) > 1:
        source = "legumeinfo"
        return source, f"{source}|{_assembly_stem(parts[1])}"
    if parts[0] == "soyomics" and len(parts) > 1:
        source = "soyomics"
        return source, f"{source}|{parts[1]}"
    return parts[0], f"{parts[0]}|unknown"


def build_annotation_candidates(
    inventory_path: Path, assembly_registry_path: Path
) -> list[AnnotationCandidate]:
    with Path(assembly_registry_path).open(newline="", encoding="utf-8") as handle:
        assemblies = list(csv.DictReader(handle, delimiter="\t"))
    pair_index: dict[str, list[str]] = {}
    stem_index: dict[str, list[str]] = {}
    for row in assemblies:
        pair_index.setdefault(_assembly_pair_key(row), []).append(row["candidate_id"])
        stem_index.setdefault(_assembly_stem(row["assembly_label"]), []).append(row["candidate_id"])

    candidates: list[AnnotationCandidate] = []
    with Path(inventory_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("kind") != "file" or row.get("file_type") != "annotation":
                continue
            relative_path = row["relative_path"]
            source, pair_key = _annotation_source_and_key(relative_path)
            role, primary = classify_annotation_role(relative_path)
            paired_ids = pair_index.get(pair_key, [])
            if not paired_ids:
                paired_ids = stem_index.get(pair_key.rsplit("|", 1)[-1], [])
            candidates.append(
                AnnotationCandidate(
                    candidate_id=hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16],
                    relative_path=relative_path,
                    source=source,
                    annotation_role=role,
                    is_primary_gene_model=primary,
                    assembly_key=pair_key,
                    paired_assembly_ids=tuple(sorted(paired_ids)),
                    size_bytes=int(row["size_bytes"]),
                    mtime_ns=int(row["mtime_ns"]),
                )
            )
    return sorted(candidates, key=lambda item: item.relative_path)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_annotation_registry(
    candidates: Iterable[AnnotationCandidate], output_dir: Path
) -> AnnotationRegistryResult:
    ordered = sorted(candidates, key=lambda item: item.relative_path)
    output_dir = Path(output_dir)
    registry_path = output_dir / "annotation_candidates.tsv"
    summary_path = output_dir / "annotation_candidates.summary.json"
    fields = [
        "candidate_id",
        "relative_path",
        "source",
        "annotation_role",
        "is_primary_gene_model",
        "assembly_key",
        "paired_assembly_ids",
        "size_bytes",
        "mtime_ns",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for candidate in ordered:
        row = asdict(candidate)
        row["is_primary_gene_model"] = str(candidate.is_primary_gene_model).lower()
        row["paired_assembly_ids"] = ";".join(candidate.paired_assembly_ids)
        writer.writerow(row)
    payload = stream.getvalue().encode("utf-8")
    _atomic_write(registry_path, payload)
    role_counts = Counter(item.annotation_role for item in ordered)
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(ordered),
        "primary_gene_model_count": sum(item.is_primary_gene_model for item in ordered),
        "paired_candidate_count": sum(bool(item.paired_assembly_ids) for item in ordered),
        "unpaired_candidate_count": sum(not item.paired_assembly_ids for item in ordered),
        "compressed_bytes": sum(item.size_bytes for item in ordered),
        "counts_by_role": dict(sorted(role_counts.items())),
        "registry_sha256": hashlib.sha256(payload).hexdigest(),
    }
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return AnnotationRegistryResult(
        registry_path=registry_path,
        summary_path=summary_path,
        candidate_count=len(ordered),
    )


def lpt_annotation_shards(
    candidates: Iterable[AnnotationCandidate], shard_count: int
) -> list[list[AnnotationCandidate]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[AnnotationCandidate]] = [[] for _ in range(shard_count)]
    totals = [0] * shard_count
    for candidate in sorted(candidates, key=lambda item: (-item.size_bytes, item.candidate_id)):
        index = min(range(shard_count), key=lambda value: (totals[value], value))
        shards[index].append(candidate)
        totals[index] += candidate.size_bytes
    for shard in shards:
        shard.sort(key=lambda item: item.relative_path)
    return shards


def _candidate_row(candidate: AnnotationCandidate) -> dict[str, object]:
    return {
        "candidate_id": candidate.candidate_id,
        "relative_path": candidate.relative_path,
        "source": candidate.source,
        "annotation_role": candidate.annotation_role,
        "is_primary_gene_model": str(candidate.is_primary_gene_model).lower(),
        "assembly_key": candidate.assembly_key,
        "paired_assembly_ids": ";".join(candidate.paired_assembly_ids),
        "size_bytes": candidate.size_bytes,
        "mtime_ns": candidate.mtime_ns,
    }


def write_annotation_shard_manifests(
    candidates: Iterable[AnnotationCandidate], output_dir: Path, shard_count: int
) -> AnnotationShardResult:
    ordered = list(candidates)
    shards = lpt_annotation_shards(ordered, shard_count)
    output_dir = Path(output_dir)
    fields = [
        "candidate_id",
        "relative_path",
        "source",
        "annotation_role",
        "is_primary_gene_model",
        "assembly_key",
        "paired_assembly_ids",
        "size_bytes",
        "mtime_ns",
    ]
    paths: list[Path] = []
    entries: list[dict[str, object]] = []
    for index, shard in enumerate(shards):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for candidate in shard:
            writer.writerow(_candidate_row(candidate))
        payload = stream.getvalue().encode("utf-8")
        path = output_dir / f"shard_{index:02d}.tsv"
        _atomic_write(path, payload)
        paths.append(path)
        entries.append(
            {
                "shard_id": index,
                "path": path.name,
                "candidate_count": len(shard),
                "compressed_bytes": sum(item.size_bytes for item in shard),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    flattened = [item.candidate_id for shard in shards for item in shard]
    expected = [item.candidate_id for item in ordered]
    if len(flattened) != len(expected) or sorted(flattened) != sorted(expected):
        raise RuntimeError("annotation shard partition is not exhaustive and unique")
    summary_path = output_dir / "shards.summary.json"
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(ordered),
        "shard_count": shard_count,
        "total_compressed_bytes": sum(item.size_bytes for item in ordered),
        "shards": entries,
    }
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return AnnotationShardResult(
        shard_paths=tuple(paths),
        summary_path=summary_path,
        shard_count=shard_count,
        candidate_count=len(ordered),
    )


def _attribute_map(value: str) -> tuple[dict[str, str], str]:
    result: dict[str, str] = {}
    detected = "UNKNOWN"
    if re.search(r"(?:^|;)\s*\w+\s+\"[^\"]*\"", value):
        detected = "GTF"
        for key, entry in re.findall(r"(?:^|;)\s*([A-Za-z0-9_.:-]+)\s+\"([^\"]*)\"", value):
            result[key] = entry
    else:
        if "=" in value:
            detected = "GFF3"
        for field in value.split(";"):
            if "=" in field:
                key, entry = field.split("=", 1)
                result[key.strip()] = entry.strip()
    return result, detected


def scan_annotation(path: Path) -> AnnotationStats:
    path = Path(path)
    file_size = path.stat().st_size
    feature_counts: Counter[str] = Counter()
    gene_ids: Counter[str] = Counter()
    transcript_ids: Counter[str] = Counter()
    seqid_max_end: dict[str, int] = {}
    feature_count = 0
    comment_line_count = 0
    malformed_line_count = 0
    invalid_coordinate_count = 0
    invalid_strand_count = 0
    invalid_phase_count = 0
    embedded_fasta = False
    gff_version = ""
    detected_formats: Counter[str] = Counter()

    with path.open("rb") as raw:
        magic = raw.read(2)
        raw.seek(0)
        hashing_reader = _HashingReader(raw)
        if magic == b"\x1f\x8b":
            stream: Iterable[bytes] = gzip.GzipFile(fileobj=hashing_reader, mode="rb")
            compression = "gzip"
        else:
            stream = hashing_reader
            compression = "plain"
        in_embedded_fasta = False
        for raw_line in stream:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if in_embedded_fasta:
                continue
            if not line:
                continue
            if line.startswith("##FASTA"):
                embedded_fasta = True
                in_embedded_fasta = True
                continue
            if line.startswith("#"):
                comment_line_count += 1
                if line.startswith("##gff-version"):
                    parts = line.split()
                    if len(parts) >= 2:
                        gff_version = parts[-1]
                continue
            fields = line.split("\t")
            if len(fields) != 9:
                malformed_line_count += 1
                continue
            seqid, _, feature_type, start_text, end_text, _, strand, phase, attributes = fields
            feature_count += 1
            feature_counts[feature_type] += 1
            coordinate_valid = True
            try:
                start = int(start_text)
                end = int(end_text)
                if start < 1 or end < start:
                    coordinate_valid = False
            except ValueError:
                coordinate_valid = False
            if not coordinate_valid:
                invalid_coordinate_count += 1
            else:
                seqid_max_end[seqid] = max(seqid_max_end.get(seqid, 0), end)
            if strand not in {"+", "-", ".", "?"}:
                invalid_strand_count += 1
            if phase not in {"0", "1", "2", "."}:
                invalid_phase_count += 1
            parsed_attributes, detected = _attribute_map(attributes)
            if detected != "UNKNOWN":
                detected_formats[detected] += 1
            lower_type = feature_type.casefold()
            if lower_type == "gene":
                identifier = parsed_attributes.get("ID") or parsed_attributes.get("gene_id")
                if identifier:
                    gene_ids[identifier] += 1
            if lower_type in {"mrna", "transcript"}:
                identifier = parsed_attributes.get("ID") or parsed_attributes.get("transcript_id")
                if identifier:
                    transcript_ids[identifier] += 1
        file_sha = hashing_reader.digest.hexdigest()

    if gff_version:
        annotation_format = "GFF3"
    elif detected_formats:
        annotation_format = detected_formats.most_common(1)[0][0]
    else:
        annotation_format = "UNKNOWN"
    lower_counts = Counter({key.casefold(): value for key, value in feature_counts.items()})
    return AnnotationStats(
        compression=compression,
        file_size_bytes=file_size,
        file_sha256=file_sha,
        format=annotation_format,
        gff_version=gff_version,
        feature_count=feature_count,
        comment_line_count=comment_line_count,
        malformed_line_count=malformed_line_count,
        invalid_coordinate_count=invalid_coordinate_count,
        invalid_strand_count=invalid_strand_count,
        invalid_phase_count=invalid_phase_count,
        embedded_fasta=embedded_fasta,
        feature_counts=dict(sorted(feature_counts.items())),
        gene_count=lower_counts["gene"],
        transcript_count=lower_counts["mrna"] + lower_counts["transcript"],
        cds_count=lower_counts["cds"],
        exon_count=lower_counts["exon"],
        unique_gene_id_count=len(gene_ids),
        duplicate_gene_id_count=sum(value - 1 for value in gene_ids.values() if value > 1),
        unique_transcript_id_count=len(transcript_ids),
        duplicate_transcript_id_count=sum(
            value - 1 for value in transcript_ids.values() if value > 1
        ),
        seqid_count=len(seqid_max_end),
        seqid_max_end=dict(sorted(seqid_max_end.items())),
    )


def _confined_annotation_target(data_root: Path, relative_path: str) -> Path:
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute() or ".." in candidate_path.parts:
        raise ValueError("annotation path is not confined to data root")
    root = Path(data_root).resolve(strict=True)
    target = (root / candidate_path).resolve(strict=True)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("annotation path is not confined to data root") from exc
    if not target.is_file():
        raise ValueError("annotation candidate is not a regular file")
    return target


def audit_annotation_candidate(
    data_root: Path,
    candidate: AnnotationCandidate,
    output_dir: Path,
    implementation_sha256: str,
) -> AnnotationAuditResult:
    target = _confined_annotation_target(data_root, candidate.relative_path)
    stat_result = target.stat()
    if stat_result.st_size != candidate.size_bytes or stat_result.st_mtime_ns != candidate.mtime_ns:
        raise ValueError(f"annotation input changed after inventory: {candidate.candidate_id}")
    result_path = Path(output_dir) / "annotations" / f"{candidate.candidate_id}.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        existing_candidate = existing.get("candidate", {})
        if (
            existing.get("state") == "PASS"
            and existing.get("implementation_sha256") == implementation_sha256
            and existing_candidate.get("candidate_id") == candidate.candidate_id
            and existing_candidate.get("relative_path") == candidate.relative_path
            and existing_candidate.get("size_bytes") == candidate.size_bytes
            and existing_candidate.get("mtime_ns") == candidate.mtime_ns
        ):
            return AnnotationAuditResult(result_path=result_path, reused=True)
    stats = scan_annotation(target)
    payload = {
        "schema_version": "1.0",
        "state": "PASS",
        "implementation_sha256": implementation_sha256,
        "candidate": _candidate_row(candidate),
        "stats": asdict(stats),
    }
    _atomic_write(
        result_path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return AnnotationAuditResult(result_path=result_path, reused=False)

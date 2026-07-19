from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GenomeSourceRecord:
    candidate_id: str
    source_kind: str
    source: str
    relative_path: str
    member_name: str
    material_key: str
    material: str
    assembly_label: str
    genome_role: str
    genus: str
    species: str
    taxon_source: str
    status: str
    payload_size_bytes: int | None
    container_size_bytes: int | None
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
    container_sha256: str
    payload_sha256: str
    sequence_sha256: str
    exact_group_id: str
    exact_group_size: int
    exact_representative: bool
    error_type: str
    error: str


@dataclass(frozen=True)
class GenomeCatalogResult:
    catalog_path: Path
    summary_path: Path
    summary: dict[str, object]


def _rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _integer(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _material_key(value: str) -> str:
    normalized = re.sub(r"(?i)(?:[_-]?t2t(?:-?2)?|\.v\d+)$", "", value)
    return re.sub(r"[^a-z0-9]+", "", normalized.casefold())


def _validate_digest(value: str, field: str, *, allow_empty: bool = False) -> None:
    if allow_empty and not value:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"invalid {field}")


def _taxa_compatible(taxa: set[tuple[str, str]]) -> bool:
    if len(taxa) <= 1:
        return True
    genera = {genus for genus, _ in taxa}
    binomials = {" ".join(species.split()[:2]) for _, species in taxa}
    return len(genera) == 1 and len(binomials) == 1


def _regular_records(
    fasta_qc_path: Path, assembly_metadata_path: Path
) -> list[GenomeSourceRecord]:
    metadata_rows = _rows(assembly_metadata_path)
    metadata = {row["candidate_id"]: row for row in metadata_rows}
    if len(metadata) != len(metadata_rows):
        raise ValueError("duplicate candidate ID in assembly metadata")
    records: list[GenomeSourceRecord] = []
    for row in _rows(fasta_qc_path):
        candidate_id = row["candidate_id"]
        if candidate_id not in metadata:
            raise ValueError(f"missing assembly metadata: {candidate_id}")
        meta = metadata[candidate_id]
        if row["status"] not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid FASTA status: {candidate_id}")
        if row["status"] == "PASS":
            _validate_digest(row["file_sha256"], "FASTA payload SHA-256")
            _validate_digest(row["sequence_sha256"], "FASTA sequence SHA-256")
        records.append(
            GenomeSourceRecord(
                candidate_id=candidate_id,
                source_kind="file",
                source=row["source"],
                relative_path=row["relative_path"],
                member_name="",
                material_key=meta["material_key"],
                material=meta["material_key"],
                assembly_label=row["assembly_label"],
                genome_role=row["genome_role"],
                genus=meta["genus"],
                species=meta["species"],
                taxon_source=meta["taxon_source"],
                status=row["status"],
                payload_size_bytes=_integer(row["size_bytes"]),
                container_size_bytes=_integer(row["size_bytes"]),
                sequence_count=_integer(row.get("sequence_count")),
                total_symbols=_integer(row.get("total_symbols")),
                acgt_count=_integer(row.get("acgt_count")),
                n_count=_integer(row.get("n_count")),
                gc_count=_integer(row.get("gc_count")),
                lowercase_count=_integer(row.get("lowercase_count")),
                ambiguous_iupac_count=_integer(row.get("ambiguous_iupac_count")),
                invalid_symbol_count=_integer(row.get("invalid_symbol_count")),
                duplicate_header_count=_integer(row.get("duplicate_header_count")),
                empty_sequence_count=_integer(row.get("empty_sequence_count")),
                min_sequence_length=_integer(row.get("min_sequence_length")),
                max_sequence_length=_integer(row.get("max_sequence_length")),
                n50=_integer(row.get("n50")),
                container_sha256=row.get("file_sha256", ""),
                payload_sha256=row.get("file_sha256", ""),
                sequence_sha256=row.get("sequence_sha256", ""),
                exact_group_id="",
                exact_group_size=0,
                exact_representative=False,
                error_type=row.get("error_type", ""),
                error=row.get("error", ""),
            )
        )
    return records


def _archive_records(
    archive_qc_path: Path, archive_taxa_path: Path
) -> list[GenomeSourceRecord]:
    curated_rows = _rows(archive_taxa_path)
    curated = {row["material"]: row for row in curated_rows}
    if len(curated) != len(curated_rows):
        raise ValueError("duplicate curated archive material")
    records: list[GenomeSourceRecord] = []
    for row in _rows(archive_qc_path):
        if row["status"] != "PASS":
            raise ValueError(f"archive genome result is not PASS: {row['candidate_id']}")
        _validate_digest(row["archive_sha256"], "archive SHA-256")
        _validate_digest(row["member_file_sha256"], "member SHA-256")
        _validate_digest(row["canonical_sequence_sha256"], "member sequence SHA-256")
        taxon = curated.get(row["material"], {})
        records.append(
            GenomeSourceRecord(
                candidate_id=row["candidate_id"],
                source_kind="zip_member",
                source="soyod_zip",
                relative_path=row["archive_relative_path"],
                member_name=row["member_name"],
                material_key=taxon.get("material_key", _material_key(row["material"])),
                material=row["material"],
                assembly_label=row["material"],
                genome_role="main",
                genus=taxon.get("genus", ""),
                species=taxon.get("species", ""),
                taxon_source=taxon.get("taxon_source", ""),
                status="PASS",
                payload_size_bytes=_integer(row["member_uncompressed_bytes"]),
                container_size_bytes=_integer(row["archive_size_bytes"]),
                sequence_count=_integer(row["sequence_count"]),
                total_symbols=_integer(row["total_symbols"]),
                acgt_count=_integer(row["acgt_count"]),
                n_count=_integer(row["n_count"]),
                gc_count=_integer(row["gc_count"]),
                lowercase_count=_integer(row["lowercase_count"]),
                ambiguous_iupac_count=_integer(row["iupac_ambiguous_count"]),
                invalid_symbol_count=_integer(row["invalid_symbol_count"]),
                duplicate_header_count=_integer(row["duplicate_header_count"]),
                empty_sequence_count=_integer(row["empty_sequence_count"]),
                min_sequence_length=_integer(row["min_sequence_length"]),
                max_sequence_length=_integer(row["max_sequence_length"]),
                n50=_integer(row["n50"]),
                container_sha256=row["archive_sha256"],
                payload_sha256=row["member_file_sha256"],
                sequence_sha256=row["canonical_sequence_sha256"],
                exact_group_id="",
                exact_group_size=0,
                exact_representative=False,
                error_type="",
                error="",
            )
        )
    return records


def _representative_rank(record: GenomeSourceRecord) -> tuple[object, ...]:
    source_order = {
        "legumeinfo": 0,
        "legume_family_legumeinfo": 1,
        "legume_family_ncbi": 2,
        "soyomics": 3,
        "soyod_zip": 4,
    }
    role_order = {"main": 0, "primary": 0, "softmasked": 1}
    return (
        source_order.get(record.source, 9),
        role_order.get(record.genome_role, 5),
        record.relative_path,
        record.member_name,
    )


def build_unified_genome_catalog(
    fasta_qc_path: Path,
    assembly_metadata_path: Path,
    archive_genome_qc_path: Path,
    archive_taxa_path: Path,
) -> list[GenomeSourceRecord]:
    records = _regular_records(fasta_qc_path, assembly_metadata_path)
    records.extend(_archive_records(archive_genome_qc_path, archive_taxa_path))
    ids = [record.candidate_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ID across unified sources")

    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if record.status == "PASS":
            groups[record.sequence_sha256].append(index)
    updated = list(records)
    for sequence_sha, indices in groups.items():
        group_id = f"exact-{sequence_sha[:16]}"
        symbols = {records[index].total_symbols for index in indices}
        if len(symbols) != 1:
            raise ValueError(f"total symbol mismatch inside exact group: {group_id}")
        taxa = {
            (records[index].genus, records[index].species)
            for index in indices
            if records[index].genus and records[index].species
        }
        if not _taxa_compatible(taxa):
            raise ValueError(f"taxon conflict inside exact group: {group_id}")
        propagated_taxon = min(taxa, key=lambda value: (len(value[1].split()), value[1])) if taxa else ("", "")
        representative = min(indices, key=lambda value: _representative_rank(records[value]))
        for index in indices:
            record = records[index]
            genus, species = record.genus, record.species
            taxon_source = record.taxon_source
            if not species and propagated_taxon[1]:
                genus, species = propagated_taxon
                taxon_source = "exact_sequence_taxon_propagation"
            updated[index] = replace(
                record,
                genus=genus,
                species=species,
                taxon_source=taxon_source,
                exact_group_id=group_id,
                exact_group_size=len(indices),
                exact_representative=index == representative,
            )
    unresolved = [record.candidate_id for record in updated if record.status == "PASS" and not record.species]
    if unresolved:
        raise ValueError(f"unresolved taxon for PASS genome sources: {','.join(unresolved)}")
    return sorted(updated, key=lambda record: (record.relative_path, record.member_name, record.candidate_id))


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


def write_unified_genome_catalog(
    records: Iterable[GenomeSourceRecord], output_dir: Path
) -> GenomeCatalogResult:
    ordered = sorted(records, key=lambda record: (record.relative_path, record.member_name, record.candidate_id))
    fieldnames = [field.name for field in fields(GenomeSourceRecord)]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in ordered:
        row = asdict(record)
        if not row["error"]:
            row["error"] = "."
        writer.writerow(row)
    payload = text.getvalue().encode("utf-8")
    pass_records = [record for record in ordered if record.status == "PASS"]
    representatives = [record for record in pass_records if record.exact_representative]
    source_counts = Counter(record.source for record in ordered)
    species_counts = Counter(record.species for record in pass_records)
    duplicate_groups = {record.exact_group_id for record in pass_records if record.exact_group_size > 1}
    summary: dict[str, object] = {
        "schema_version": "1.0",
        "source_record_count": len(ordered),
        "pass_source_count": len(pass_records),
        "fail_source_count": len(ordered) - len(pass_records),
        "exact_unique_sequence_count": len(representatives),
        "exact_unique_symbols": sum(record.total_symbols or 0 for record in representatives),
        "exact_duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_source_count": sum(record.exact_group_size for record in pass_records if record.exact_representative and record.exact_group_size > 1),
        "species_count": len(species_counts),
        "source_counts": dict(sorted(source_counts.items())),
        "species_source_counts": dict(sorted(species_counts.items())),
        "catalog_sha256": hashlib.sha256(payload).hexdigest(),
    }
    output_dir = Path(output_dir)
    catalog_path = output_dir / "unified_genome_catalog.tsv"
    summary_path = output_dir / "unified_genome_catalog.summary.json"
    _atomic_write(catalog_path, payload)
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return GenomeCatalogResult(catalog_path, summary_path, summary)

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AssemblyMetadataRecord:
    candidate_id: str
    relative_path: str
    source: str
    assembly_label: str
    genome_role: str
    size_bytes: int
    mtime_ns: int
    material_key: str
    genus: str
    species: str
    material_type: str
    gwh_id: str
    official_assembly_code: str
    taxon_source: str


@dataclass(frozen=True)
class AssemblyMetadataWriteResult:
    manifest_path: Path
    summary_path: Path


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _material_name(label: str) -> str:
    return label.split(".gnm", 1)[0]


def _material_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _soyomics_code(relative_path: str) -> str:
    return Path(relative_path).name.split(".", 1)[0]


def _sequence_prefix(relative_path: str) -> str:
    return Path(relative_path).name.split(".", 1)[0].casefold()


def build_assembly_metadata(
    assembly_registry_path: Path,
    annotation_registry_path: Path,
    soyomics_metadata_path: Path,
) -> list[AssemblyMetadataRecord]:
    assemblies = _read_tsv(assembly_registry_path)
    annotations = _read_tsv(annotation_registry_path)
    official_rows = _read_tsv(soyomics_metadata_path)
    official = {row["assembly"]: row for row in official_rows}

    records: dict[str, AssemblyMetadataRecord] = {}
    for row in assemblies:
        if row["source"] == "soyomics":
            material_name = _soyomics_code(row["relative_path"])
        else:
            material_name = _material_name(row["assembly_label"])
        genus = row.get("genus", "")
        species = row.get("species", "")
        records[row["candidate_id"]] = AssemblyMetadataRecord(
            candidate_id=row["candidate_id"],
            relative_path=row["relative_path"],
            source=row["source"],
            assembly_label=row["assembly_label"],
            genome_role=row["genome_role"],
            size_bytes=int(row["size_bytes"]),
            mtime_ns=int(row["mtime_ns"]),
            material_key=_material_key(material_name),
            genus=genus,
            species=species,
            material_type="",
            gwh_id="",
            official_assembly_code="",
            taxon_source="assembly_path" if genus and species else "unresolved",
        )

    for row in annotations:
        parts = row.get("assembly_key", "").split("|")
        if len(parts) < 4 or parts[0] != "legume_family_legumeinfo":
            continue
        genus = parts[1]
        species = f"{genus} {parts[2].replace('_', ' ')}"
        for candidate_id in filter(None, row.get("paired_assembly_ids", "").split(";")):
            record = records.get(candidate_id)
            if record and not record.species:
                records[candidate_id] = replace(
                    record,
                    genus=genus,
                    species=species,
                    taxon_source="paired_legumeinfo_annotation",
                )

    prefix_taxa: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records.values():
        if record.genus and record.species:
            prefix_taxa[_sequence_prefix(record.relative_path)].add(
                (record.genus, record.species)
            )
    for candidate_id, record in list(records.items()):
        if record.species:
            continue
        taxa = prefix_taxa.get(_sequence_prefix(record.relative_path), set())
        if len(taxa) == 1:
            genus, species = next(iter(taxa))
            records[candidate_id] = replace(
                record,
                genus=genus,
                species=species,
                taxon_source="sequence_prefix_from_explicit_taxon",
            )

    for candidate_id, record in list(records.items()):
        if record.source != "soyomics":
            continue
        code = _soyomics_code(record.relative_path)
        row = official.get(code)
        if not row:
            continue
        species = row.get("scientific_name", "")
        genus = species.split(" ", 1)[0] if species else ""
        records[candidate_id] = replace(
            record,
            genus=genus,
            species=species,
            material_type=row.get("material_type", ""),
            gwh_id=row.get("gwh_id", ""),
            official_assembly_code=code,
            taxon_source="soyomics_official_api",
        )

    material_taxa: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for record in records.values():
        if record.genus and record.species:
            material_taxa[record.material_key].add((record.genus, record.species))
    for candidate_id, record in list(records.items()):
        if record.species:
            continue
        taxa = material_taxa.get(record.material_key, set())
        if len(taxa) == 1:
            genus, species = next(iter(taxa))
            records[candidate_id] = replace(
                record,
                genus=genus,
                species=species,
                taxon_source="cross_source_material",
            )

    return sorted(records.values(), key=lambda item: item.relative_path)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_assembly_metadata(
    records: Iterable[AssemblyMetadataRecord], output_dir: Path
) -> AssemblyMetadataWriteResult:
    ordered = sorted(records, key=lambda item: item.relative_path)
    if not ordered:
        raise ValueError("assembly metadata is empty")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "assembly_metadata.tsv"
    summary_path = output_dir / "assembly_metadata.summary.json"
    fieldnames = list(asdict(ordered[0]))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in ordered:
        writer.writerow(asdict(record))
    manifest = stream.getvalue().encode("utf-8")
    _atomic_write(manifest_path, manifest)
    summary = {
        "schema_version": "1.0",
        "record_count": len(ordered),
        "resolved_taxon_count": sum(bool(record.species) for record in ordered),
        "unresolved_taxon_count": sum(not record.species for record in ordered),
        "species_count": len({record.species for record in ordered if record.species}),
        "genus_count": len({record.genus for record in ordered if record.genus}),
        "counts_by_taxon_source": dict(sorted(Counter(record.taxon_source for record in ordered).items())),
        "counts_by_species": dict(sorted(Counter(record.species for record in ordered if record.species).items())),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return AssemblyMetadataWriteResult(manifest_path=manifest_path, summary_path=summary_path)

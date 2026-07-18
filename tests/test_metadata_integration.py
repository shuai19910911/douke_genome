from __future__ import annotations

import csv
from pathlib import Path

from legumegenomefm.metadata_integration import build_assembly_metadata, write_assembly_metadata


def write_tsv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def test_build_assembly_metadata_uses_explicit_official_and_propagated_taxa(tmp_path: Path) -> None:
    assemblies = tmp_path / "assemblies.tsv"
    annotations = tmp_path / "annotations.tsv"
    soyomics = tmp_path / "soyomics.tsv"
    write_tsv(
        assemblies,
        [
            "candidate_id",
            "relative_path",
            "source",
            "genus",
            "species",
            "assembly_label",
            "genome_role",
            "size_bytes",
            "mtime_ns",
        ],
        [
            ["a" * 16, "legume_family/legumeinfo/Glycine/soja/X.gnm1.A/genome/glyso.X.fna", "legume_family_legumeinfo", "Glycine", "Glycine soja", "X.gnm1.A", "main", 1, 1],
            ["b" * 16, "legumeinfo/W05.gnm1.S/genome/b.fna", "legumeinfo", "", "", "W05.gnm1.S", "main", 1, 1],
            ["c" * 16, "soyomics/PI/genome/SoyW01.fasta.gz", "soyomics", "", "", "PI", "generic", 1, 1],
            ["d" * 16, "soyomics/W05/genome/W05.a1.fasta.gz", "soyomics", "", "", "W05", "generic", 1, 1],
            ["f" * 16, "legumeinfo/Y.gnm1.A/genome/glyso.Y.genome_main.fna.gz", "legumeinfo", "", "", "Y.gnm1.A", "main", 1, 1],
        ],
    )
    write_tsv(
        annotations,
        [
            "candidate_id",
            "relative_path",
            "source",
            "annotation_role",
            "is_primary_gene_model",
            "assembly_key",
            "paired_assembly_ids",
            "size_bytes",
            "mtime_ns",
        ],
        [
            ["e" * 16, "x.gff", "legume_family_legumeinfo", "gene_models_main", "true", "legume_family_legumeinfo|Glycine|soja|W05.gnm1", "b" * 16, 1, 1]
        ],
    )
    write_tsv(
        soyomics,
        ["assembly", "scientific_name", "material_type", "gwh_id"],
        [["SoyW01", "Glycine soja", "Soja", "GWH1"]],
    )

    records = build_assembly_metadata(assemblies, annotations, soyomics)
    by_id = {record.candidate_id: record for record in records}

    assert by_id["a" * 16].species == "Glycine soja"
    assert by_id["a" * 16].taxon_source == "assembly_path"
    assert by_id["b" * 16].species == "Glycine soja"
    assert by_id["b" * 16].taxon_source == "paired_legumeinfo_annotation"
    assert by_id["c" * 16].species == "Glycine soja"
    assert by_id["c" * 16].taxon_source == "soyomics_official_api"
    assert by_id["c" * 16].gwh_id == "GWH1"
    assert by_id["d" * 16].species == "Glycine soja"
    assert by_id["d" * 16].taxon_source == "cross_source_material"
    assert by_id["f" * 16].species == "Glycine soja"
    assert by_id["f" * 16].taxon_source == "sequence_prefix_from_explicit_taxon"


def test_write_assembly_metadata_is_deterministic(tmp_path: Path) -> None:
    assemblies = tmp_path / "assemblies.tsv"
    annotations = tmp_path / "annotations.tsv"
    soyomics = tmp_path / "soyomics.tsv"
    write_tsv(
        assemblies,
        ["candidate_id", "relative_path", "source", "genus", "species", "assembly_label", "genome_role", "size_bytes", "mtime_ns"],
        [["a" * 16, "x/a.fna", "x", "A", "A one", "asm", "main", 1, 1]],
    )
    write_tsv(
        annotations,
        ["candidate_id", "relative_path", "source", "annotation_role", "is_primary_gene_model", "assembly_key", "paired_assembly_ids", "size_bytes", "mtime_ns"],
        [],
    )
    write_tsv(soyomics, ["assembly", "scientific_name", "material_type", "gwh_id"], [])
    records = build_assembly_metadata(assemblies, annotations, soyomics)

    first = write_assembly_metadata(records, tmp_path / "out")
    payload = first.manifest_path.read_bytes()
    second = write_assembly_metadata(records, tmp_path / "out")

    assert second.manifest_path.read_bytes() == payload
    assert all(not line.endswith((b"\t", b" ")) for line in payload.splitlines())

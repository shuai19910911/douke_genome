from __future__ import annotations

import csv
from pathlib import Path

from legumegenomefm.genome_catalog import (
    _taxa_compatible,
    build_unified_genome_catalog,
    write_unified_genome_catalog,
)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_catalog_groups_cross_container_duplicates_and_keeps_failures(tmp_path: Path) -> None:
    fasta = tmp_path / "fasta.tsv"
    metadata = tmp_path / "metadata.tsv"
    archive = tmp_path / "archive.tsv"
    curated = tmp_path / "curated.tsv"
    common = {
        "sequence_count": "2",
        "total_symbols": "100",
        "acgt_count": "98",
        "n_count": "2",
        "gc_count": "40",
        "lowercase_count": "0",
        "invalid_symbol_count": "0",
        "duplicate_header_count": "0",
        "empty_sequence_count": "0",
        "min_sequence_length": "40",
        "max_sequence_length": "60",
        "n50": "60",
    }
    _write_tsv(
        fasta,
        [
            {
                "candidate_id": "a" * 16,
                "relative_path": "legumeinfo/Cultivar.gnm1/genome/a.fna.gz",
                "source": "legumeinfo",
                "genus": "Glycine",
                "species": "Glycine max",
                "assembly_label": "Cultivar.gnm1",
                "genome_role": "main",
                "size_bytes": "50",
                "mtime_ns": "1",
                **common,
                "ambiguous_iupac_count": "0",
                "compression": "gzip",
                "file_sha256": "1" * 64,
                "sequence_sha256": "9" * 64,
                "error_type": "",
                "error": "",
                "status": "PASS",
            },
            {
                "candidate_id": "b" * 16,
                "relative_path": "legumeinfo/Broken/genome/b.fna.gz",
                "source": "legumeinfo",
                "genus": "Glycine",
                "species": "Glycine max",
                "assembly_label": "Broken.gnm1",
                "genome_role": "main",
                "size_bytes": "30",
                "mtime_ns": "2",
                **{key: "" for key in common},
                "ambiguous_iupac_count": "",
                "compression": "gzip",
                "file_sha256": "",
                "sequence_sha256": "",
                "error_type": "EOFError",
                "error": "truncated",
                "status": "FAIL",
            },
        ],
    )
    _write_tsv(
        metadata,
        [
            {
                "candidate_id": "a" * 16,
                "material_key": "cultivar",
                "genus": "Glycine",
                "species": "Glycine max",
                "taxon_source": "assembly_path",
            },
            {
                "candidate_id": "b" * 16,
                "material_key": "broken",
                "genus": "Glycine",
                "species": "Glycine max",
                "taxon_source": "assembly_path",
            },
        ],
    )
    _write_tsv(
        archive,
        [
            {
                "candidate_id": "c" * 16,
                "archive_relative_path": "soyod/Cultivar/genome/c.zip",
                "member_name": "c.fasta",
                "material": "Cultivar",
                "archive_size_bytes": "25",
                "archive_mtime_ns": "3",
                "archive_sha256": "2" * 64,
                "member_uncompressed_bytes": "100",
                "sequence_count": "2",
                "total_symbols": "100",
                "acgt_count": "98",
                "n_count": "2",
                "gc_count": "40",
                "lowercase_count": "0",
                "iupac_ambiguous_count": "0",
                "invalid_symbol_count": "0",
                "duplicate_header_count": "0",
                "empty_sequence_count": "0",
                "min_sequence_length": "40",
                "max_sequence_length": "60",
                "n50": "60",
                "member_file_sha256": "3" * 64,
                "canonical_sequence_sha256": "9" * 64,
                "status": "PASS",
            },
            {
                "candidate_id": "d" * 16,
                "archive_relative_path": "soyod/Wild_T2T/genome/d.zip",
                "member_name": "d.fasta",
                "material": "Wild_T2T",
                "archive_size_bytes": "26",
                "archive_mtime_ns": "4",
                "archive_sha256": "4" * 64,
                "member_uncompressed_bytes": "101",
                "sequence_count": "1",
                "total_symbols": "101",
                "acgt_count": "101",
                "n_count": "0",
                "gc_count": "41",
                "lowercase_count": "0",
                "iupac_ambiguous_count": "0",
                "invalid_symbol_count": "0",
                "duplicate_header_count": "0",
                "empty_sequence_count": "0",
                "min_sequence_length": "101",
                "max_sequence_length": "101",
                "n50": "101",
                "member_file_sha256": "5" * 64,
                "canonical_sequence_sha256": "8" * 64,
                "status": "PASS",
            },
        ],
    )
    _write_tsv(
        curated,
        [
            {
                "material": "Wild_T2T",
                "material_key": "wild",
                "genus": "Glycine",
                "species": "Glycine soja",
                "taxon_source": "curated_material_identity",
            }
        ],
    )

    records = build_unified_genome_catalog(fasta, metadata, archive, curated)
    assert len(records) == 4
    by_id = {record.candidate_id: record for record in records}
    assert by_id["a" * 16].exact_group_id == by_id["c" * 16].exact_group_id
    assert by_id["a" * 16].exact_group_size == 2
    assert by_id["a" * 16].exact_representative is True
    assert by_id["c" * 16].exact_representative is False
    assert by_id["c" * 16].species == "Glycine max"
    assert by_id["d" * 16].species == "Glycine soja"
    assert by_id["b" * 16].status == "FAIL"
    assert by_id["b" * 16].exact_group_id == ""


def test_catalog_write_is_deterministic_and_summarizes(tmp_path: Path) -> None:
    record_type = __import__("legumegenomefm.genome_catalog", fromlist=["GenomeSourceRecord"]).GenomeSourceRecord
    base = dict(
        source_kind="file", source="legumeinfo", relative_path="a", member_name="", material_key="a",
        material="a", assembly_label="a", genome_role="main", genus="Glycine", species="Glycine max",
        taxon_source="path", status="PASS", payload_size_bytes=10, container_size_bytes=10,
        sequence_count=1, total_symbols=10, acgt_count=10, n_count=0, gc_count=4, lowercase_count=0,
        ambiguous_iupac_count=0, invalid_symbol_count=0, duplicate_header_count=0, empty_sequence_count=0,
        min_sequence_length=10, max_sequence_length=10, n50=10, container_sha256="1" * 64,
        payload_sha256="1" * 64, sequence_sha256="2" * 64, exact_group_id="g", exact_group_size=1,
        exact_representative=True, error_type="", error="",
    )
    record = record_type(candidate_id="a" * 16, **base)
    first = write_unified_genome_catalog([record], tmp_path / "one")
    second = write_unified_genome_catalog([record], tmp_path / "two")
    assert first.catalog_path.read_bytes() == second.catalog_path.read_bytes()
    assert first.summary["pass_source_count"] == 1
    assert first.summary["exact_unique_sequence_count"] == 1
    assert first.summary["exact_unique_symbols"] == 10


def test_exact_group_taxa_allow_subspecies_but_reject_another_species() -> None:
    assert _taxa_compatible(
        {("Arachis", "Arachis hypogaea"), ("Arachis", "Arachis hypogaea subsp. hypogaea")}
    )
    assert not _taxa_compatible(
        {("Arachis", "Arachis hypogaea"), ("Arachis", "Arachis duranensis")}
    )

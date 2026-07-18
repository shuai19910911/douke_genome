from __future__ import annotations

import csv
import gzip
import hashlib
from pathlib import Path

import pytest

from legumegenomefm.annotation_audit import (
    AnnotationCandidate,
    audit_annotation_candidate,
    build_annotation_candidates,
    classify_annotation_role,
    lpt_annotation_shards,
    scan_annotation,
    write_annotation_registry,
    write_annotation_shard_manifests,
)


def test_classify_annotation_role_distinguishes_primary_and_auxiliary() -> None:
    assert classify_annotation_role("x.gene_models_main.gff3.gz") == ("gene_models_main", True)
    assert classify_annotation_role("GCF_1_genomic.gff.gz") == ("ncbi_genomic", True)
    assert classify_annotation_role("material/gff/SoyC01.gene.gff.gz") == ("soyomics_gene", True)
    assert classify_annotation_role("x.iprscan.gff3.gz") == ("functional_iprscan", False)
    assert classify_annotation_role("x.genome_repeatmask_coords.gff.gz") == ("repeat", False)
    assert classify_annotation_role("x.noncoding.gff3.gz") == ("noncoding", False)


def write_inventory(path: Path) -> None:
    rows = [
        [
            "legume_family/legumeinfo/Glycine/max/Wm82.gnm4.ann1.T8TQ/gff3/glyma.Wm82.gnm4.ann1.T8TQ.gene_models_main.gff3.gz",
            "file",
            "annotation",
            100,
            1,
            "",
            "0o644",
        ],
        [
            "legume_family/ncbi/Medicago_truncatula/MedtrA17_3.5/gff3/GCF_1_MedtrA17_3.5_genomic.gff.gz",
            "file",
            "annotation",
            200,
            2,
            "",
            "0o644",
        ],
        [
            "soyomics/PI_548362/gff/SoyC10.gene.gff.gz",
            "file",
            "annotation",
            300,
            3,
            "",
            "0o644",
        ],
        [
            "soyomics/PI_548362/ncrna/SoyC10.ncRNA.gff3.gz",
            "file",
            "annotation",
            50,
            4,
            "",
            "0o644",
        ],
        ["ignored.gff3.gz", "symlink", "annotation", 0, 5, "target", "0o777"],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "relative_path",
                "kind",
                "file_type",
                "size_bytes",
                "mtime_ns",
                "link_target",
                "mode_octal",
            ]
        )
        writer.writerows(rows)


def write_assembly_registry(path: Path) -> None:
    rows = [
        ["a" * 16, "legume_family/legumeinfo/Glycine/max/Wm82.gnm4.ABC/genome/a.fna.gz", "legume_family_legumeinfo", "Glycine", "Glycine max", "Wm82.gnm4.ABC", "main", 1, 1],
        ["b" * 16, "legume_family/ncbi/Medicago_truncatula/MedtrA17_3.5/genome/b.fna.gz", "legume_family_ncbi", "Medicago", "Medicago truncatula", "MedtrA17_3.5", "ncbi_genomic", 1, 1],
        ["c" * 16, "soyomics/PI_548362/genome/c.fna.gz", "soyomics", "", "", "PI_548362", "generic", 1, 1],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
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
            ]
        )
        writer.writerows(rows)


def test_build_annotation_candidates_pairs_by_source_and_assembly_stem(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.tsv"
    assemblies = tmp_path / "assemblies.tsv"
    write_inventory(inventory)
    write_assembly_registry(assemblies)

    candidates = build_annotation_candidates(inventory, assemblies)

    assert len(candidates) == 4
    assert candidates[0].paired_assembly_ids == ("a" * 16,)
    assert candidates[1].paired_assembly_ids == ("b" * 16,)
    assert candidates[2].paired_assembly_ids == ("c" * 16,)
    assert candidates[3].annotation_role == "noncoding"
    assert candidates[3].paired_assembly_ids == ("c" * 16,)


def test_build_annotation_candidates_supports_cross_source_stem_and_non_numeric_gnm(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "inventory.tsv"
    assemblies = tmp_path / "assemblies.tsv"
    with inventory.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "relative_path",
                "kind",
                "file_type",
                "size_bytes",
                "mtime_ns",
                "link_target",
                "mode_octal",
            ]
        )
        writer.writerow(
            [
                "legume_family/legumeinfo/Glycine/soja/W05.gnm1.ann1.T47J/gff3/x.gene_models_main.gff3.gz",
                "file",
                "annotation",
                10,
                1,
                "",
                "0o644",
            ]
        )
        writer.writerow(
            [
                "legume_family/legumeinfo/Medicago/truncatula/R108.gnmHiC_1.ann1.Y8NH/gff3/y.gene_models_main.gff3.gz",
                "file",
                "annotation",
                10,
                1,
                "",
                "0o644",
            ]
        )
    with assemblies.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
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
            ]
        )
        writer.writerow(
            [
                "d" * 16,
                "legumeinfo/W05.gnm1.SVL1/genome/x.fna.gz",
                "legumeinfo",
                "",
                "",
                "W05.gnm1.SVL1",
                "main",
                1,
                1,
            ]
        )
        writer.writerow(
            [
                "e" * 16,
                "legume_family/legumeinfo/Medicago/truncatula/R108.gnmHiC_1.7RWS/genome/y.fna.gz",
                "legume_family_legumeinfo",
                "Medicago",
                "Medicago truncatula",
                "R108.gnmHiC_1.7RWS",
                "main",
                1,
                1,
            ]
        )

    candidates = build_annotation_candidates(inventory, assemblies)

    paired = {candidate.assembly_key: candidate.paired_assembly_ids for candidate in candidates}
    assert paired["legume_family_legumeinfo|Glycine|soja|W05.gnm1"] == ("d" * 16,)
    assert paired["legume_family_legumeinfo|Medicago|truncatula|R108.gnmHiC_1"] == (
        "e" * 16,
    )


def test_write_annotation_registry_is_deterministic(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.tsv"
    assemblies = tmp_path / "assemblies.tsv"
    write_inventory(inventory)
    write_assembly_registry(assemblies)
    candidates = build_annotation_candidates(inventory, assemblies)

    result = write_annotation_registry(candidates, tmp_path / "out")
    first = result.registry_path.read_bytes()
    second = write_annotation_registry(candidates, tmp_path / "out")

    assert second.registry_path.read_bytes() == first
    assert all(not line.endswith((b"\t", b" ")) for line in first.splitlines())
    assert result.candidate_count == 4


def test_scan_annotation_counts_features_coordinates_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "genes.gff3"
    path.write_text(
        "##gff-version 3\n"
        "chr1\tsrc\tgene\t1\t100\t.\t+\t.\tID=g1\n"
        "chr1\tsrc\tmRNA\t1\t100\t.\t+\t.\tID=t1;Parent=g1\n"
        "chr1\tsrc\texon\t1\t50\t.\t+\t.\tParent=t1\n"
        "chr1\tsrc\tCDS\t10\t50\t.\t+\t0\tID=c1;Parent=t1\n"
        "chr2\tsrc\tgene\t5\t20\t.\t-\t.\tID=g1\n"
        "bad\tline\n"
    )

    stats = scan_annotation(path)

    assert stats.gff_version == "3"
    assert stats.feature_count == 5
    assert stats.feature_counts == {"CDS": 1, "exon": 1, "gene": 2, "mRNA": 1}
    assert stats.gene_count == 2
    assert stats.transcript_count == 1
    assert stats.cds_count == 1
    assert stats.exon_count == 1
    assert stats.unique_gene_id_count == 1
    assert stats.duplicate_gene_id_count == 1
    assert stats.seqid_count == 2
    assert stats.seqid_max_end == {"chr1": 100, "chr2": 20}
    assert stats.malformed_line_count == 1
    assert stats.invalid_coordinate_count == 0


def test_scan_annotation_supports_gzip_and_gtf_gene_ids(tmp_path: Path) -> None:
    path = tmp_path / "genes.gtf.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(
            "chr1\tsrc\tgene\t1\t9\t.\t+\t.\tgene_id \"g1\";\n"
            "chr1\tsrc\ttranscript\t1\t9\t.\t+\t.\tgene_id \"g1\"; transcript_id \"t1\";\n"
        )

    stats = scan_annotation(path)

    assert stats.compression == "gzip"
    assert stats.format == "GTF"
    assert stats.unique_gene_id_count == 1
    assert stats.unique_transcript_id_count == 1


def test_scan_annotation_records_invalid_coordinates_without_crashing(tmp_path: Path) -> None:
    path = tmp_path / "bad.gff3"
    path.write_text("chr1\tsrc\tgene\t0\t-1\t.\tX\t9\tID=g1\n")

    stats = scan_annotation(path)

    assert stats.feature_count == 1
    assert stats.invalid_coordinate_count == 1
    assert stats.invalid_strand_count == 1
    assert stats.invalid_phase_count == 1


def test_scan_annotation_consumes_embedded_fasta_for_complete_file_hash(tmp_path: Path) -> None:
    path = tmp_path / "embedded.gff3"
    payload = (
        b"##gff-version 3\n"
        b"chr1\tsrc\tgene\t1\t4\t.\t+\t.\tID=g1\n"
        b"##FASTA\n>chr1\nACGT\n"
    )
    path.write_bytes(payload)

    stats = scan_annotation(path)

    assert stats.embedded_fasta
    assert stats.feature_count == 1
    assert stats.file_sha256 == hashlib.sha256(payload).hexdigest()


def make_annotation_candidate(candidate_id: str, relative_path: str, size: int, mtime: int) -> AnnotationCandidate:
    return AnnotationCandidate(
        candidate_id=candidate_id,
        relative_path=relative_path,
        source="source",
        annotation_role="gene_models_main",
        is_primary_gene_model=True,
        assembly_key="source|asm",
        paired_assembly_ids=("a" * 16,),
        size_bytes=size,
        mtime_ns=mtime,
    )


def test_annotation_shards_are_deterministic_and_complete(tmp_path: Path) -> None:
    candidates = [
        make_annotation_candidate(f"{index:016x}", f"x/{index}.gff3.gz", size, index)
        for index, size in enumerate([9, 8, 7, 6, 5])
    ]
    first = lpt_annotation_shards(candidates, 2)
    second = lpt_annotation_shards(candidates, 2)

    assert first == second
    assert sorted(item.candidate_id for shard in first for item in shard) == sorted(
        item.candidate_id for item in candidates
    )
    result = write_annotation_shard_manifests(candidates, tmp_path / "shards", 2)
    assert result.shard_count == 2
    assert result.candidate_count == 5
    for path in result.shard_paths:
        assert all(not line.endswith((b"\t", b" ")) for line in path.read_bytes().splitlines())


def test_audit_annotation_candidate_is_resumable_and_confined(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    target = data_root / "source" / "genes.gff3"
    target.parent.mkdir(parents=True)
    target.write_text("##gff-version 3\nchr1\ts\tgene\t1\t4\t.\t+\t.\tID=g1\n")
    stat_result = target.stat()
    candidate = make_annotation_candidate(
        "a" * 16, "source/genes.gff3", stat_result.st_size, stat_result.st_mtime_ns
    )
    implementation = "f" * 64

    first = audit_annotation_candidate(data_root, candidate, tmp_path / "out", implementation)
    second = audit_annotation_candidate(data_root, candidate, tmp_path / "out", implementation)

    assert not first.reused
    assert second.reused
    assert first.result_path == second.result_path
    with pytest.raises(ValueError, match="confined"):
        audit_annotation_candidate(
            data_root,
            make_annotation_candidate("b" * 16, "../escape.gff3", 1, 1),
            tmp_path / "out",
            implementation,
        )

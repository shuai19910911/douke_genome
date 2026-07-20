from __future__ import annotations

import io
from pathlib import Path

import yaml

from scripts.audit_source_provenance import retrieve as retrieve_source_provenance

from legumegenomefm.data_refinement import (
    annotation_is_strict,
    audit_feature_coordinates,
    busco_gate_passes,
    canonical_material_key,
    eligible_tiara_chunks,
    classify_assembly,
    local_callable_intervals,
    legumeinfo_readme_url,
    metadata_license_allows_training,
    metadata_has_chromosome_evidence,
    metadata_provenance_passes,
    merge_half_open_intervals,
    merge_inclusive_intervals,
    merged_interval_bases,
    nonoverlapping_capacity,
    parse_gff_features,
    record_is_primary_nuclear,
    select_unique_candidates,
    subtract_half_open_intervals,
    taxon_name_matches,
    univec_hit_is_high_confidence,
)


def test_annotation_gate_rejects_invalid_or_ambiguous_primary_models() -> None:
    row = {
        "is_primary_gene_model": "True",
        "status": "PASS",
        "gene_count": "42",
        "duplicate_gene_id_count": "0",
        "duplicate_transcript_id_count": "0",
        "malformed_line_count": "0",
        "invalid_coordinate_count": "0",
        "invalid_strand_count": "0",
        "invalid_phase_count": "0",
    }
    assert annotation_is_strict(row)
    for field in (
        "duplicate_gene_id_count",
        "duplicate_transcript_id_count",
        "malformed_line_count",
        "invalid_coordinate_count",
        "invalid_strand_count",
        "invalid_phase_count",
    ):
        broken = {**row, field: "1"}
        assert not annotation_is_strict(broken)
    assert not annotation_is_strict({**row, "gene_count": "0"})
    assert not annotation_is_strict({**row, "is_primary_gene_model": "False"})


def test_coordinate_closure_requires_exact_seqids_and_in_bounds_features() -> None:
    contigs = {"Chr01": 1000, "Chr02": 500}
    passed = audit_feature_coordinates(
        contigs,
        [("Chr01", 1, 100, "gene"), ("Chr02", 400, 500, "CDS")],
    )
    assert passed["status"] == "PASS"
    assert passed["matched_seqid_count"] == 2
    unknown = audit_feature_coordinates(contigs, [("1", 1, 100, "gene")])
    assert unknown["status"] == "FAIL"
    assert unknown["unknown_seqids"] == ["1"]
    out_of_bounds = audit_feature_coordinates(contigs, [("Chr01", 900, 1001, "gene")])
    assert out_of_bounds["status"] == "FAIL"
    assert out_of_bounds["out_of_bounds_feature_count"] == 1
    no_genes = audit_feature_coordinates(contigs, [("Chr01", 1, 100, "CDS")])
    assert no_genes["status"] == "FAIL"


def test_gff_feature_parser_ignores_blank_and_comment_lines() -> None:
    text = "\n# comment\nChr01\tsource\tgene\t1\t10\t.\t+\t.\tID=g1\n\n"
    assert list(parse_gff_features(io.StringIO(text))) == [("Chr01", 1, 10, "gene")]


def test_legumeinfo_readme_urls_are_reconstructed_from_both_source_layouts() -> None:
    family = "legume_family/legumeinfo/Aeschynomene/evenia/CIAT22838.gnm1.XF73/genome/a.fna.gz"
    assert legumeinfo_readme_url(family, "Aeschynomene evenia").endswith(
        "/Aeschynomene/evenia/genomes/CIAT22838.gnm1.XF73/README.CIAT22838.gnm1.XF73.yml"
    )
    soybean = "legumeinfo/Zh13.gnm2.LV9P/genome/glyma.fna.gz"
    assert legumeinfo_readme_url(soybean, "Glycine max").endswith(
        "/Glycine/max/genomes/Zh13.gnm2.LV9P/README.Zh13.gnm2.LV9P.yml"
    )


def test_metadata_requires_explicit_chromosome_evidence() -> None:
    assert metadata_has_chromosome_evidence({"chromosome_prefix": "Chr"})
    assert metadata_has_chromosome_evidence({"description": "telomere-to-telomere assembly"})
    assert not metadata_has_chromosome_evidence({"description": "whole genome shotgun assembly"})


def test_metadata_provenance_accepts_repository_or_publication_but_not_neither() -> None:
    base = {"scientific_name": "Glycine max", "chromosome_prefix": "Chr"}
    assert metadata_provenance_passes({**base, "source": "https://example.org"}, "Glycine max")
    assert metadata_provenance_passes({**base, "publication_doi": "10.1/example"}, "Glycine max")
    assert not metadata_provenance_passes(base, "Glycine max")
    assert not metadata_provenance_passes({**base, "source": "https://example.org"}, "Glycine soja")


def test_legumeinfo_license_gate_requires_explicit_public_open_terms() -> None:
    allowed_access = ["public"]
    allowed_licenses = ["open"]
    assert metadata_license_allows_training("Public", "Open", allowed_access, allowed_licenses)
    assert not metadata_license_allows_training(
        "public", "Open, with usage agreement", allowed_access, allowed_licenses
    )
    assert not metadata_license_allows_training(
        "public, restricted", "Open", allowed_access, allowed_licenses
    )
    assert not metadata_license_allows_training(".", ".", allowed_access, allowed_licenses)


def test_ncbi_assembly_report_proves_level_but_does_not_grant_training_license(tmp_path: Path) -> None:
    report = tmp_path / "assembly_report.txt"
    report.write_text("# Assembly level: Chromosome\n")
    row = {
        "candidate_id": "candidate",
        "species": "Glycine max",
        "material_key": "sample",
        "assembly_evidence": "chromosome_official",
        "official_assembly_report": report.name,
        "assembly_accession": "GCA_TEST",
    }
    result = retrieve_source_provenance((row, 1, str(tmp_path), [], ["public"], ["open"]))
    assert result["evidence_type"] == "NCBI_ASSEMBLY_REPORT"
    assert result["status"] == "LICENSE_REVIEW_REQUIRED"
    assert result["license"] == "NCBI_SUBMITTER_RIGHTS_UNRESOLVED"


def test_taxon_matching_normalizes_punctuation_and_allows_explicit_source_aliases() -> None:
    assert taxon_name_matches("Glycine D3-tomentella", "Glycine D3 tomentella", [])
    assert taxon_name_matches("Glycine stenophita", "Glycine stenophiyta", ["Glycine stenophiyta"])
    assert not taxon_name_matches("Glycine max", "Glycine soja", [])


def test_material_aliases_are_explicit_and_do_not_use_fuzzy_substrings() -> None:
    aliases = {
        "Glycine max": {
            "zh13": ["zh13", "whfsgmzh1310", "zh13iga1005", "gmaxzh13", "gmaxzh13v20"]
        }
    }
    assert canonical_material_key("Glycine max", "whfsgmzh1310", aliases) == "zh13"
    assert canonical_material_key("Glycine max", "zh13iga1005", aliases) == "zh13"
    assert canonical_material_key("Medicago sativa", "zhongmuno1", aliases) == "zhongmuno1"


def test_refinement_config_keeps_strict_primary_and_contamination_thresholds() -> None:
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / "configs/data_refinement.yaml").read_text())
    assert config["assembly"]["minimum_primary_nuclear_fraction"] == 0.90
    assert config["contamination"]["minimum_primary_tiara_evaluated_fraction"] == 0.95
    assert config["contamination"]["maximum_primary_prokaryotic_fraction"] == 0.005
    assert config["contamination"]["maximum_primary_vector_fraction"] == 0.0001


def test_busco_gate_requires_both_genome_and_annotation_completeness() -> None:
    assert busco_gate_passes(95.0, 91.0, 90.0, 90.0)
    assert not busco_gate_passes(89.9, 99.0, 90.0, 90.0)
    assert not busco_gate_passes(99.0, 89.9, 90.0, 90.0)


def test_univec_rules_and_interval_merging_are_not_double_counted() -> None:
    rules = [
        {"minimum_alignment_length": 50, "minimum_percent_identity": 90.0},
        {"minimum_alignment_length": 30, "minimum_percent_identity": 95.0},
    ]
    assert univec_hit_is_high_confidence(90.0, 50, rules)
    assert univec_hit_is_high_confidence(95.0, 30, rules)
    assert not univec_hit_is_high_confidence(94.9, 30, rules)
    assert merge_inclusive_intervals([(10, 20), (15, 30), (40, 40)]) == [(10, 30), (40, 40)]
    assert merged_interval_bases([(10, 20), (15, 30), (40, 40)]) == 22


def test_primary_nuclear_record_policy_excludes_scaffolds_and_organelles() -> None:
    assert record_is_primary_nuclear("Chr01", 50_000_000, "assembled-molecule", "Chromosome", 10_000_000)
    assert not record_is_primary_nuclear("MT", 500_000, "assembled-molecule", "Mitochondrion", 10_000_000)
    assert record_is_primary_nuclear("glyma.Zh13.gnm2.Chr01", 50_000_000, ".", ".", 10_000_000)
    assert not record_is_primary_nuclear("scaffold_1", 2_000_000, ".", ".", 10_000_000)
    assert not record_is_primary_nuclear("chloroplast", 150_000, ".", ".", 10_000_000)


def test_tiara_chunks_skip_n_rich_sequence_and_preserve_coordinates() -> None:
    sequence = "ACGTACGT" + "N" * 8 + "ACGTACGT"
    assert eligible_tiara_chunks(sequence, 8, 4, 0.5) == [
        (0, 8, "ACGTACGT"),
        (16, 24, "ACGTACGT"),
    ]


def test_half_open_masks_merge_without_counting_touching_regions_twice() -> None:
    assert merge_half_open_intervals([(10, 20), (15, 30), (30, 35), (40, 41)]) == [
        (10, 35),
        (40, 41),
    ]


def test_callable_intervals_are_split_around_contamination_masks() -> None:
    assert subtract_half_open_intervals([(0, 100), (200, 260)], [(20, 30), (25, 50), (230, 300)]) == [
        (0, 20),
        (50, 100),
        (200, 230),
    ]


def test_callable_intervals_are_localized_before_record_masks_are_subtracted() -> None:
    manifest = {
        "contigs": [
            {"name": "chr1", "offset": 0, "length": 1000},
            {"name": "chr2", "offset": 1000, "length": 500},
        ],
        "callable_intervals": [
            {"contig_index": 1, "start": 1020, "length": 100},
        ],
    }
    offset, local = local_callable_intervals(manifest, 1)
    assert offset == 1000
    assert local == [(20, 120)]
    clean = subtract_half_open_intervals(local, [(30, 40)])
    assert clean == [(20, 30), (40, 120)]
    assert [(offset + start, end - start) for start, end in clean] == [(1020, 10), (1040, 80)]


def test_assembly_classification_separates_official_t2t_proxy_and_failure() -> None:
    assert classify_assembly("Complete Genome", False, 1, 0.0, 10_000_000, 0.8) == (
        "complete_genome",
        4,
        False,
    )
    assert classify_assembly("Chromosome", False, 1, 0.0, 10_000_000, 0.8) == (
        "chromosome_official",
        3,
        False,
    )
    assert classify_assembly("Chromosome", True, 1, 0.0, 10_000_000, 0.8) == (
        "chromosome_official_t2t",
        5,
        False,
    )
    assert classify_assembly("UNVERIFIED", True, 1, 0.0, 10_000_000, 0.8) == (
        "t2t_label",
        5,
        True,
    )
    assert classify_assembly("UNVERIFIED", False, 20_000_000, 0.9, 10_000_000, 0.8) == (
        "structural_proxy",
        2,
        True,
    )
    assert classify_assembly("UNVERIFIED", False, 9_999_999, 0.9, 10_000_000, 0.8) == (
        "insufficient",
        0,
        True,
    )


def test_nonoverlapping_capacity_does_not_inflate_overlapping_windows() -> None:
    intervals = [{"length": 262_144}, {"length": 524_287}, {"length": 524_288}]
    assert nonoverlapping_capacity(intervals, 262_144) == 4


def test_selection_prefers_quality_within_orientation_then_one_per_material() -> None:
    rows = [
        {
            "candidate_id": "a",
            "species": "Glycine max",
            "material_key": "zh13",
            "orientation_group_id": "orientation-1",
            "hard_gate_pass": True,
            "assembly_tier": 3,
            "long_callable_fraction": 0.99,
            "n_fraction": 0.01,
            "large_sequence_fraction": 0.95,
            "n50": 50_000_000,
            "contig_count": 20,
            "base_count": 1_000_000_000,
        },
        {
            "candidate_id": "b",
            "species": "Glycine max",
            "material_key": "zh13",
            "orientation_group_id": "orientation-1",
            "hard_gate_pass": True,
            "assembly_tier": 5,
            "long_callable_fraction": 1.0,
            "n_fraction": 0.0,
            "large_sequence_fraction": 1.0,
            "n50": 51_000_000,
            "contig_count": 20,
            "base_count": 1_010_000_000,
        },
        {
            "candidate_id": "c",
            "species": "Glycine max",
            "material_key": "zh13",
            "orientation_group_id": "orientation-2",
            "hard_gate_pass": True,
            "assembly_tier": 2,
            "long_callable_fraction": 0.9,
            "n_fraction": 0.02,
            "large_sequence_fraction": 0.9,
            "n50": 40_000_000,
            "contig_count": 40,
            "base_count": 990_000_000,
        },
        {
            "candidate_id": "d",
            "species": "Glycine soja",
            "material_key": "pi1",
            "orientation_group_id": "orientation-3",
            "hard_gate_pass": False,
            "assembly_tier": 0,
            "long_callable_fraction": 0.0,
            "n_fraction": 0.2,
            "large_sequence_fraction": 0.0,
            "n50": 1,
            "contig_count": 1000,
            "base_count": 10,
        },
    ]
    assert select_unique_candidates(rows) == {
        "a": "REJECTED_ORIENTATION_ALTERNATIVE",
        "b": "SELECTED",
        "c": "REJECTED_MATERIAL_ALTERNATIVE",
        "d": "REJECTED_HARD_GATE",
    }

from __future__ import annotations

import csv
import json
from pathlib import Path
from zipfile import ZipFile

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "LEGUMEGENOMEFM_RESEARCH_DESIGN.md"
DOCX = ROOT / "docs" / "LegumeGenomeFM_Research_Design_and_Benchmark_20260722.docx"


def test_research_design_reports_current_performance_honestly() -> None:
    text = REPORT.read_text(encoding="utf-8")
    assert "CANDIDATE_SET_READY_DATA_NOT_YET_PACKAGED" in text
    assert "正式预训练和全部下游任务均未运行" in text
    assert "91,922,061,939" in text
    assert "314,669,504" in text
    assert "AgroNT-1B" in text
    assert "PlantCAD2-Large" in text
    assert "Nucleotide Transformer v2 500M" in text
    assert "Evo-2-7B" in text
    assert "目前所有任务性能" in text
    assert "10.1016/j.xplc.2024.100984" in text
    assert "10.1016/j.xplc.2024.100961" not in text
    assert "10.1038/s41588-025-02190-w" not in text
    assert "10.1186/s12915-025-02377-x" not in text


def test_proposed_cold_genus_split_closes_exactly() -> None:
    evidence = json.loads((ROOT / "research/proposed_pretraining_split_evidence.json").read_text())
    assert evidence["status"] == "PROPOSED_NOT_FROZEN_NOT_A_TRAINING_RELEASE"
    capacities = evidence["capacity_by_split"]
    total = capacities["all_selected"]
    children = [
        capacities["pretraining_pool"],
        capacities["cold_development_phaseolus"],
        capacities["sealed_cold_test_arachis_vicia"],
    ]
    assert sum(item["sources"] for item in children) == total["sources"] == 74
    assert sum(item["species"] for item in children) == total["species"] == 19
    assert sum(item["trainable_bp"] for item in children) == total["trainable_bp"] == 91_922_061_939
    for context, expected in total["eligible_nonoverlap_windows"].items():
        assert sum(item["eligible_nonoverlap_windows"][context] for item in children) == expected

    matrix = yaml.safe_load((ROOT / "configs/evaluation_matrix.yaml").read_text())
    assert matrix["pretraining_cold_genera"] == ["Arachis", "Phaseolus", "Vicia"]
    assert matrix["cold_genus_roles"] == {
        "development": ["Phaseolus"],
        "sealed_test": ["Arachis", "Vicia"],
    }


def test_baseline_evidence_has_required_models_and_verified_gpn_size() -> None:
    with (ROOT / "research/baseline_model_evidence.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_model = {row["model"]: row for row in rows}
    required = {
        "AgroNT-1B",
        "PlantCAD2-Large",
        "Nucleotide-Transformer-v2-500M-multi-species",
        "Evo-2-7B",
    }
    assert required <= by_model.keys()
    assert by_model["GPN-Brassicales"]["parameter_count"] == "65,880,071"
    assert by_model["PlantCAD2-Large"]["publication_status"] == "preprint"
    assert all(row["paper_url"] and row["official_model_url"] for row in rows)


def test_docx_package_contains_current_evidence() -> None:
    assert DOCX.is_file()
    with ZipFile(DOCX) as archive:
        assert archive.testzip() is None
        document_xml = archive.read("word/document.xml").decode("utf-8")
        assert "314,669,504" in document_xml
        assert "91,922,061,939" in document_xml
        assert "10.1016/j.xplc.2024.100984" in document_xml
        assert "10.1038/s41588-025-02170-w" in document_xml
        assert "10.1186/s12870-025-07202-5" in document_xml
        assert "10.1016/j.xplc.2024.100961" not in document_xml

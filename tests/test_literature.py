from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from legumegenomefm.literature import (
    LiteratureRecord,
    build_crossref_url,
    crossref_item_to_record,
    deduplicate_records,
    is_nature_portfolio,
    normalize_doi,
    titles_equivalent,
    write_literature_manifest,
)


def test_normalize_doi_removes_resolver_and_prefix() -> None:
    assert normalize_doi("https://doi.org/10.1038/S41592-024-02523-Z") == "10.1038/s41592-024-02523-z"
    assert normalize_doi("doi: 10.1038/s41586-025-10014-0") == "10.1038/s41586-025-10014-0"


def test_titles_equivalent_normalizes_markup_and_unicode_punctuation() -> None:
    expected = "Nucleotide Transformer: building and evaluating robust foundation models"
    actual = "<i>Nucleotide Transformer</i>: Building and evaluating robust foundation models"
    assert titles_equivalent(expected, actual)
    assert not titles_equivalent(expected, "A different genomic foundation model")


def test_build_crossref_url_freezes_cutoff_and_fields() -> None:
    url = build_crossref_url("DNA genome language model", "2026-07-18", rows=75)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    assert parsed.netloc == "api.crossref.org"
    assert query["query.bibliographic"] == ["DNA genome language model"]
    assert query["rows"] == ["75"]
    assert query["filter"] == [
        "from-pub-date:2018-01-01,until-pub-date:2026-07-18,type:journal-article"
    ]
    assert "DOI" in query["select"][0]
    assert "author" in query["select"][0]


def test_crossref_item_to_record_uses_article_metadata() -> None:
    item = {
        "DOI": "10.1038/S41592-024-02523-Z",
        "title": ["Nucleotide Transformer: building and evaluating robust foundation models for human genomics"],
        "container-title": ["Nature Methods"],
        "publisher": "Springer Science and Business Media LLC",
        "published": {"date-parts": [[2025, 1, 6]]},
        "URL": "https://doi.org/10.1038/s41592-024-02523-z",
        "author": [{"family": "Dalla-Torre", "given": "Hugo"}, {"family": "Gonzalez", "given": "Liam"}],
        "type": "journal-article",
    }

    record = crossref_item_to_record(item, "nucleotide_transformer")

    assert record.doi == "10.1038/s41592-024-02523-z"
    assert record.year == 2025
    assert record.journal == "Nature Methods"
    assert record.authors == "Dalla-Torre, Hugo; Gonzalez, Liam"
    assert record.query_ids == ("nucleotide_transformer",)
    assert record.sources == ("crossref",)


def test_deduplicate_records_merges_queries_and_sources_by_doi() -> None:
    first = LiteratureRecord(
        doi="10.1038/example",
        title="A genome model",
        journal="Nature Methods",
        year=2025,
        publisher="Springer Nature",
        authors="A, One",
        url="https://doi.org/10.1038/example",
        publication_type="journal-article",
        query_ids=("q1",),
        sources=("crossref",),
    )
    second = LiteratureRecord(
        doi="10.1038/example",
        title="A genome model",
        journal="Nature Methods",
        year=2025,
        publisher="Springer Nature",
        authors="A, One; B, Two",
        url="https://www.nature.com/articles/example",
        publication_type="research-article",
        query_ids=("q2",),
        sources=("europe_pmc",),
    )

    merged = deduplicate_records([first, second])

    assert len(merged) == 1
    assert merged[0].authors == "A, One; B, Two"
    assert merged[0].query_ids == ("q1", "q2")
    assert merged[0].sources == ("crossref", "europe_pmc")


def test_nature_portfolio_requires_supported_journal_not_title_keyword() -> None:
    assert is_nature_portfolio("Nature Genetics", "Springer Nature")
    assert is_nature_portfolio("Communications Biology", "Springer Nature")
    assert is_nature_portfolio("Scientific Data", "Springer Nature")
    assert not is_nature_portfolio("Bioinformatics", "Oxford University Press")
    assert not is_nature_portfolio("Journal of Natural Language Processing", "Other")


def test_write_literature_manifest_emits_deduplicated_audit_files(tmp_path: Path) -> None:
    duplicate_a = LiteratureRecord(
        doi="10.1038/example",
        title="A genome model",
        journal="Nature Methods",
        year=2025,
        publisher="Springer Nature",
        authors="A, One",
        url="https://doi.org/10.1038/example",
        publication_type="journal-article",
        query_ids=("q1",),
        sources=("crossref",),
    )
    duplicate_b = LiteratureRecord(
        doi="10.1038/example",
        title="A genome model",
        journal="Nature Methods",
        year=2025,
        publisher="Springer Nature",
        authors="A, One; B, Two",
        url="https://doi.org/10.1038/example",
        publication_type="journal-article",
        query_ids=("q2",),
        sources=("crossref",),
    )

    result = write_literature_manifest(
        [duplicate_a, duplicate_b],
        query_ids=["q1", "q2", "q3"],
        cutoff_date="2026-07-18",
        output_dir=tmp_path,
    )

    summary = json.loads(result.summary_json.read_text())
    assert summary["schema_version"] == "1.0"
    assert summary["cutoff_date"] == "2026-07-18"
    assert summary["unique_candidate_count"] == 1
    assert summary["nature_portfolio_candidate_count"] == 1
    assert summary["query_count"] == 3
    assert summary["queries_with_candidates"] == ["q1", "q2"]
    lines = result.candidates_tsv.read_text().splitlines()
    assert len(lines) == 2
    assert "q1;q2" in lines[1]
    assert len(result.candidates_sha256) == 64

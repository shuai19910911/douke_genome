from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import urlencode


_NATURE_PORTFOLIO_EXACT = {
    "communications biology",
    "communications medicine",
    "communications psychology",
    "scientific data",
    "scientific reports",
}


@dataclass(frozen=True)
class LiteratureRecord:
    doi: str
    title: str
    journal: str
    year: int | None
    publisher: str
    authors: str
    url: str
    publication_type: str
    query_ids: tuple[str, ...]
    sources: tuple[str, ...]


def build_crossref_url(query: str, cutoff_date: str, rows: int = 100) -> str:
    if not 1 <= rows <= 1000:
        raise ValueError("Crossref rows must be in [1, 1000]")
    parameters = {
        "query.bibliographic": query,
        "filter": f"from-pub-date:2018-01-01,until-pub-date:{cutoff_date},type:journal-article",
        "rows": str(rows),
        "select": "DOI,title,author,published,container-title,publisher,URL,type",
    }
    return "https://api.crossref.org/works?" + urlencode(parameters)


def normalize_doi(value: str | None) -> str:
    if not value:
        return ""
    doi = value.strip().lower()
    doi = re.sub(r"^doi:\s*", "", doi)
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return doi.strip()


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _first(values: object) -> str:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
        return str(values[0]).strip()
    return ""


def _year(item: Mapping[str, object]) -> int | None:
    published = item.get("published")
    if not isinstance(published, Mapping):
        return None
    parts = published.get("date-parts")
    if not isinstance(parts, Sequence) or not parts or not isinstance(parts[0], Sequence) or not parts[0]:
        return None
    try:
        return int(parts[0][0])
    except (TypeError, ValueError):
        return None


def _authors(item: Mapping[str, object]) -> str:
    result: list[str] = []
    values = item.get("author")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ""
    for value in values:
        if not isinstance(value, Mapping):
            continue
        family = str(value.get("family", "")).strip()
        given = str(value.get("given", "")).strip()
        if family and given:
            result.append(f"{family}, {given}")
        elif family or given:
            result.append(family or given)
    return "; ".join(result)


def crossref_item_to_record(item: Mapping[str, object], query_id: str) -> LiteratureRecord:
    return LiteratureRecord(
        doi=normalize_doi(str(item.get("DOI", ""))),
        title=_first(item.get("title")),
        journal=_first(item.get("container-title")),
        year=_year(item),
        publisher=str(item.get("publisher", "")).strip(),
        authors=_authors(item),
        url=str(item.get("URL", "")).strip(),
        publication_type=str(item.get("type", "")).strip(),
        query_ids=(query_id,),
        sources=("crossref",),
    )


def _prefer_longer(first: str, second: str) -> str:
    return second if len(second) > len(first) else first


def deduplicate_records(records: Iterable[LiteratureRecord]) -> list[LiteratureRecord]:
    merged: dict[tuple[str, str, int | None], LiteratureRecord] = {}
    for record in records:
        key = ("doi", record.doi, None) if record.doi else ("title", normalize_title(record.title), record.year)
        existing = merged.get(key)
        if existing is None:
            merged[key] = record
            continue
        merged[key] = replace(
            existing,
            title=_prefer_longer(existing.title, record.title),
            journal=_prefer_longer(existing.journal, record.journal),
            publisher=_prefer_longer(existing.publisher, record.publisher),
            authors=_prefer_longer(existing.authors, record.authors),
            url=_prefer_longer(existing.url, record.url),
            publication_type=_prefer_longer(existing.publication_type, record.publication_type),
            query_ids=tuple(sorted(set(existing.query_ids) | set(record.query_ids))),
            sources=tuple(sorted(set(existing.sources) | set(record.sources))),
        )
    return sorted(merged.values(), key=lambda value: (value.year or 0, normalize_title(value.title), value.doi))


def is_nature_portfolio(journal: str, publisher: str = "") -> bool:
    normalized = journal.strip().casefold()
    if normalized == "nature" or normalized.startswith("nature "):
        return True
    if normalized in _NATURE_PORTFOLIO_EXACT:
        return True
    if normalized.startswith("communications ") and "springer" in publisher.casefold():
        return True
    return False


@dataclass(frozen=True)
class LiteratureManifestResult:
    candidates_tsv: Path
    summary_json: Path
    candidates_sha256: str


def _atomic_write(path: Path, payload: bytes) -> None:
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


def write_literature_manifest(
    records: Iterable[LiteratureRecord],
    query_ids: Iterable[str],
    cutoff_date: str,
    output_dir: Path,
) -> LiteratureManifestResult:
    unique = deduplicate_records(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_tsv = output_dir / "literature_candidates.tsv"
    summary_json = output_dir / "literature_search.summary.json"
    fields = [
        "doi",
        "title",
        "journal",
        "year",
        "publisher",
        "authors",
        "url",
        "publication_type",
        "nature_portfolio",
        "query_ids",
        "sources",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for record in unique:
        writer.writerow(
            {
                "doi": record.doi,
                "title": record.title,
                "journal": record.journal,
                "year": "" if record.year is None else record.year,
                "publisher": record.publisher,
                "authors": record.authors,
                "url": record.url,
                "publication_type": record.publication_type,
                "nature_portfolio": str(is_nature_portfolio(record.journal, record.publisher)).lower(),
                "query_ids": ";".join(record.query_ids),
                "sources": ";".join(record.sources),
            }
        )
    candidates_payload = buffer.getvalue().encode("utf-8")
    candidates_sha256 = hashlib.sha256(candidates_payload).hexdigest()
    declared_queries = sorted(set(query_ids))
    seen_queries = sorted({query_id for record in unique for query_id in record.query_ids})
    summary = {
        "schema_version": "1.0",
        "cutoff_date": cutoff_date,
        "query_count": len(declared_queries),
        "queries_with_candidates": seen_queries,
        "queries_without_candidates": sorted(set(declared_queries) - set(seen_queries)),
        "unique_candidate_count": len(unique),
        "nature_portfolio_candidate_count": sum(
            is_nature_portfolio(record.journal, record.publisher) for record in unique
        ),
        "candidates_sha256": candidates_sha256,
    }
    summary_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write(candidates_tsv, candidates_payload)
    _atomic_write(summary_json, summary_payload)
    return LiteratureManifestResult(candidates_tsv, summary_json, candidates_sha256)

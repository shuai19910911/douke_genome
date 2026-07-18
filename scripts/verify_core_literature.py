#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from legumegenomefm.literature import normalize_doi, titles_equivalent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a frozen core-literature allowlist")
    parser.add_argument("--allowlist", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def fetch(url: str, *, accept: str = "application/json") -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "LegumeGenomeFM-literature-audit/1.0 (mailto:literature-audit@example.invalid)",
        },
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), response.geturl()
        except Exception as exc:  # network boundary
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to retrieve source after retries: {url}") from last_error


def crossref_record(doi: str) -> dict[str, object]:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    payload, _ = fetch(url)
    return json.loads(payload)["message"]


def date_string(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, dict):
        return ""
    parts = value.get("date-parts")
    if not isinstance(parts, list) or not parts or not isinstance(parts[0], list):
        return ""
    return "-".join(str(part) for part in parts[0])


def first_text(item: dict[str, object], key: str) -> str:
    value = item.get(key)
    if isinstance(value, list) and value:
        return str(value[0])
    return str(value or "")


def meta_value(html: str, name: str) -> str:
    escaped = re.escape(name)
    patterns = [
        rf'<meta[^>]+name=["\']{escaped}["\'][^>]+content=["\']([^"\']+)',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']{escaped}["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    config = json.loads(args.allowlist.read_text(encoding="utf-8"))
    records = config["records"]
    output_rows: list[dict[str, object]] = []
    failures: list[str] = []

    for index, record in enumerate(records, 1):
        identifier = str(record["identifier"])
        identifier_type = str(record["identifier_type"])
        expected_title = str(record["expected_title"])
        observed_title = ""
        journal = ""
        publisher = ""
        published_online = ""
        published_print = ""
        metadata_type = ""
        article_type = ""
        update_count = 0
        resolved_url = ""
        status = "PASS"

        try:
            if identifier_type == "doi":
                doi = normalize_doi(identifier)
                item = crossref_record(doi)
                observed_title = first_text(item, "title")
                journal = first_text(item, "container-title")
                publisher = str(item.get("publisher") or "")
                published_online = date_string(item, "published-online")
                published_print = date_string(item, "published-print")
                metadata_type = str(item.get("type") or "")
                update_to = item.get("update-to")
                update_count = len(update_to) if isinstance(update_to, list) else 0
                resolved_url = str(item.get("URL") or f"https://doi.org/{doi}")
                if doi.startswith("10.1038/"):
                    page, resolved_url = fetch(
                        f"https://www.nature.com/articles/{doi.split('/', 1)[1]}",
                        accept="text/html,application/xhtml+xml",
                    )
                    html = page.decode("utf-8", errors="replace")
                    article_type = meta_value(html, "citation_article_type")
                    page_title = meta_value(html, "citation_title")
                    if page_title and not titles_equivalent(expected_title, page_title):
                        failures.append(f"{identifier}: Nature page title mismatch")
                        status = "FAIL"
                    expected_type = str(record.get("expected_article_type") or "")
                    if expected_type and article_type != expected_type:
                        failures.append(
                            f"{identifier}: article type {article_type!r} != {expected_type!r}"
                        )
                        status = "FAIL"
            elif identifier_type == "url":
                page, resolved_url = fetch(identifier, accept="text/html,application/xhtml+xml")
                html = page.decode("utf-8", errors="replace")
                observed_title = meta_value(html, "citation_title")
                if not observed_title:
                    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                    observed_title = re.sub(r"\s+", " ", match.group(1)).strip() if match else ""
                metadata_type = "url"
            else:
                raise ValueError(f"unsupported identifier_type={identifier_type}")

            if not titles_equivalent(expected_title, observed_title):
                failures.append(f"{identifier}: metadata title mismatch: {observed_title!r}")
                status = "FAIL"
        except Exception as exc:
            failures.append(f"{identifier}: {type(exc).__name__}: {exc}")
            status = "FAIL"

        output_rows.append(
            {
                "identifier": identifier,
                "identifier_type": identifier_type,
                "title": observed_title,
                "journal": journal,
                "publisher": publisher,
                "published_online": published_online,
                "published_print": published_print,
                "metadata_type": metadata_type,
                "article_type": article_type,
                "decision": record["decision"],
                "evidence_role": record["evidence_role"],
                "discovery": record["discovery"],
                "crossref_update_count": update_count,
                "resolved_url": resolved_url,
                "verification_status": status,
            }
        )
        print(f"verified={index}/{len(records)} status={status} id={identifier}", file=sys.stderr)
        time.sleep(0.1)

    fields = list(output_rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    manifest_path = args.output_dir / "core_literature_verified.tsv"
    atomic_write(manifest_path, stream.getvalue())

    summary = {
        "schema_version": "1.0",
        "cutoff_date": config["cutoff_date"],
        "record_count": len(output_rows),
        "pass_count": sum(row["verification_status"] == "PASS" for row in output_rows),
        "fail_count": sum(row["verification_status"] == "FAIL" for row in output_rows),
        "decision_counts": {
            decision: sum(row["decision"] == decision for row in output_rows)
            for decision in sorted({str(row["decision"]) for row in output_rows})
        },
        "failures": failures,
    }
    atomic_write(
        args.output_dir / "core_literature_verified.summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

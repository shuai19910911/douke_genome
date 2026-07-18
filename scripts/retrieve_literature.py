#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from legumegenomefm.literature import (
    build_crossref_url,
    crossref_item_to_record,
    write_literature_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve an auditable Crossref literature candidate universe.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fetch_json(url: str, attempts: int = 4) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "LegumeGenomeFM/0.1 (systematic evidence retrieval)"},
    )
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Crossref request failed after {attempts} attempts") from last_error


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    cutoff_date = str(config["cutoff_date"])
    rows = int(config["rows_per_query"])
    queries = config["queries"]
    query_ids = [str(item["id"]) for item in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("literature query IDs must be unique")
    records = []
    for index, item in enumerate(queries, start=1):
        query_id = str(item["id"])
        query = str(item["query"])
        print(f"query {index}/{len(queries)}: {query_id}", file=sys.stderr, flush=True)
        payload = fetch_json(build_crossref_url(query, cutoff_date, rows))
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("items"), list):
            raise RuntimeError(f"invalid Crossref response for {query_id}")
        records.extend(crossref_item_to_record(value, query_id) for value in message["items"])
        time.sleep(0.15)
    result = write_literature_manifest(records, query_ids, cutoff_date, args.output_dir)
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    print(json.dumps({"state": "PASS", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

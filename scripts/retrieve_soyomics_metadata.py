#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


BASE_URL = "https://ngdc.cncb.ac.cn/soyomics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve official SoyOmics assembly metadata")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def fetch_json(url: str, *, post: bool = False) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=b"" if post else None,
        method="POST" if post else "GET",
        headers={"User-Agent": "LegumeGenomeFM-data-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("SoyOmics API did not return an object")
    return payload


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    index = fetch_json(f"{BASE_URL}/assembly/accession_all")
    results = index.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("SoyOmics accession index is empty")
    assemblies = sorted({str(item["value"]) for item in results if isinstance(item, dict)})
    rows: list[dict[str, str]] = []
    for assembly in assemblies:
        url = f"{BASE_URL}/assembly/summary?" + urllib.parse.urlencode({"assembly": assembly})
        payload = fetch_json(url, post=True)
        if payload.get("exit") != 1 or not isinstance(payload.get("data"), dict):
            raise RuntimeError(f"SoyOmics summary unavailable: {assembly}")
        data = payload["data"]
        rows.append(
            {
                "assembly": assembly,
                "accession_id": str(data.get("accessionId") or ""),
                "scientific_name": str(data.get("scientificName") or ""),
                "common_name": str(data.get("commonName") or ""),
                "material_type": str(data.get("type") or ""),
                "country": str(data.get("country") or ""),
                "assembly_level": str(data.get("assemblyLevel") or ""),
                "gwh_id": str(data.get("gwhId") or ""),
                "genome_size_text": str(data.get("size") or ""),
                "n50_text": str(data.get("n50") or ""),
                "protein_count_text": str(data.get("protein") or ""),
                "publication": str(data.get("pub") or ""),
            }
        )
        time.sleep(0.05)
    fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    manifest = stream.getvalue().encode("utf-8")
    manifest_path = args.output_dir / "soyomics_assemblies.tsv"
    atomic_write(manifest_path, manifest)
    summary = {
        "schema_version": "1.0",
        "source_url": f"{BASE_URL}/genome_assembly",
        "source_publication_doi": "10.1016/j.molp.2023.03.011",
        "assembly_count": len(rows),
        "species_counts": dict(sorted(Counter(row["scientific_name"] for row in rows).items())),
        "material_type_counts": dict(sorted(Counter(row["material_type"] for row in rows).items())),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
    }
    atomic_write(
        args.output_dir / "soyomics_assemblies.summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

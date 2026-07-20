#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from legumegenomefm.data_refinement import (
    legumeinfo_readme_url,
    metadata_license_allows_training,
    metadata_has_chromosome_evidence,
    metadata_provenance_passes,
    taxon_name_matches,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def scalar(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key, ".")
    if isinstance(value, list):
        return ";".join(str(item) for item in value) or "."
    text = str(value).strip()
    return text or "."


def retrieve(task: tuple[dict[str, str], int, str, list[str], list[str], list[str]]) -> dict[str, str]:
    row, timeout, project_root_text, accepted_aliases, allowed_access, allowed_licenses = task
    project_root = Path(project_root_text)
    result = {
        "candidate_id": row["candidate_id"],
        "species": row["species"],
        "material_key": row["material_key"],
        "assembly_evidence": row["assembly_evidence"],
        "evidence_type": ".",
        "evidence_location": ".",
        "evidence_sha256": ".",
        "metadata_identifier": ".",
        "metadata_scientific_name": ".",
        "chromosome_prefix": ".",
        "source_repository": ".",
        "genbank_accession": ".",
        "bioproject": ".",
        "publication_doi": ".",
        "publication_title": ".",
        "public_access_level": ".",
        "license": ".",
        "error": ".",
        "status": "PENDING",
    }
    if row["assembly_evidence"] in {"chromosome_official", "chromosome_official_t2t", "complete_genome"}:
        result.update(
            {
                "evidence_type": "NCBI_ASSEMBLY_REPORT",
                "evidence_location": row["official_assembly_report"],
                "evidence_sha256": sha256(project_root / row["official_assembly_report"]),
                "genbank_accession": row["assembly_accession"],
                "public_access_level": "public_repository",
                "license": "NCBI_SUBMITTER_RIGHTS_UNRESOLVED",
                "error": "license_review_required_ncbi_submitter_rights",
                "status": "LICENSE_REVIEW_REQUIRED",
            }
        )
        return result
    if row["source"] not in {"legumeinfo", "legume_family_legumeinfo"}:
        result["error"] = "no auditable source metadata resolver for this source"
        result["status"] = "PENDING_NO_RESOLVER"
        return result
    try:
        url = legumeinfo_readme_url(row["relative_path"], row["species"])
        request = urllib.request.Request(url, headers={"User-Agent": "soygenome-data-audit/1.0"})
        payload: bytes | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(2**attempt)
        if payload is None:
            assert last_error is not None
            raise last_error
        metadata = yaml.safe_load(payload.decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("README YAML is not a mapping")
        result.update(
            {
                "evidence_type": "LEGUMEINFO_README",
                "evidence_location": url,
                "evidence_sha256": hashlib.sha256(payload).hexdigest(),
                "metadata_identifier": scalar(metadata, "identifier"),
                "metadata_scientific_name": scalar(metadata, "scientific_name"),
                "chromosome_prefix": scalar(metadata, "chromosome_prefix"),
                "source_repository": scalar(metadata, "source"),
                "genbank_accession": scalar(metadata, "genbank_accession"),
                "bioproject": scalar(metadata, "bioproject"),
                "publication_doi": scalar(metadata, "publication_doi"),
                "publication_title": scalar(metadata, "publication_title"),
                "public_access_level": scalar(metadata, "public_access_level"),
                "license": scalar(metadata, "license"),
            }
        )
        taxon_matches = taxon_name_matches(row["species"], result["metadata_scientific_name"], accepted_aliases)
        source_present = result["source_repository"] != "." or result["genbank_accession"] != "."
        publication_present = result["publication_doi"] != "." or result["publication_title"] != "."
        chromosome_present = metadata_has_chromosome_evidence(metadata)
        if metadata_provenance_passes(metadata, row["species"], accepted_aliases):
            if metadata_license_allows_training(
                result["public_access_level"],
                result["license"],
                allowed_access,
                allowed_licenses,
            ):
                result["status"] = "PASS"
            else:
                result["error"] = "license_review_required"
                result["status"] = "LICENSE_REVIEW_REQUIRED"
        else:
            missing = []
            if not taxon_matches:
                missing.append("taxon_mismatch")
            if not source_present:
                missing.append("source_missing")
            if not publication_present:
                missing.append("publication_missing")
            if not chromosome_present:
                missing.append("chromosome_evidence_missing")
            result["error"] = ";".join(missing)
            result["status"] = "INCOMPLETE_METADATA"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "ERROR"
    return result


def render_tsv(rows: list[dict[str, str]]) -> bytes:
    fieldnames = list(rows[0])
    if fieldnames[-1] != "status":
        raise ValueError("status must be the final field")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve auditable chromosome-scale source metadata")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    if args.workers < 1 or args.timeout < 1:
        raise ValueError("workers and timeout must be positive")
    candidates_path = args.candidates.resolve()
    project_root = args.project_root.resolve()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    provenance_config = config.get("source_provenance", {})
    aliases_by_species = provenance_config.get("accepted_taxon_aliases", {})
    allowed_access = list(provenance_config.get("allowed_public_access_levels", []))
    allowed_licenses = list(provenance_config.get("allowed_licenses", []))
    if not allowed_access or not allowed_licenses:
        raise ValueError("source provenance license allowlists must be non-empty")
    with candidates_path.open(newline="", encoding="utf-8") as handle:
        selected = [row for row in csv.DictReader(handle, delimiter="\t") if row["hard_gate_pass"] == "True"]
    if not selected:
        raise ValueError("candidate table contains no selected rows")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(
            pool.map(
                retrieve,
                (
                    (
                        row,
                        args.timeout,
                        str(project_root),
                        list(aliases_by_species.get(row["species"], [])),
                        allowed_access,
                        allowed_licenses,
                    )
                    for row in selected
                ),
            )
        )
    rows.sort(key=lambda row: row["candidate_id"])
    tsv = render_tsv(rows)
    output_path = args.output.resolve()
    atomic_write(output_path, tsv)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary = {
        "schema_version": "1.0",
        "state": "PASS" if counts.get("PASS", 0) == len(rows) else "INCOMPLETE",
        "candidate_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "config_sha256": sha256(args.config.resolve()),
        "candidate_tsv_sha256": sha256(candidates_path),
        "provenance_tsv_sha256": hashlib.sha256(tsv).hexdigest(),
    }
    atomic_write(
        output_path.with_suffix(".summary.json"),
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a unified ordinary-file and ZIP-member annotation catalog")
    parser.add_argument("--ordinary", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    unified: list[dict[str, object]] = []
    for row in rows(args.ordinary):
        paired = row["paired_assembly_ids"] or "."
        unified.append(
            {
                "annotation_id": row["candidate_id"],
                "source_kind": "file",
                "relative_path": row["relative_path"],
                "member_name": ".",
                "source": row["source"],
                "material": row["assembly_key"],
                "annotation_role": row["annotation_role"],
                "is_primary_gene_model": row["is_primary_gene_model"],
                "paired_genome_ids": paired,
                "format": row["format"],
                "feature_count": row["feature_count"],
                "gene_count": row["gene_count"],
                "transcript_count": row["transcript_count"],
                "cds_count": row["cds_count"],
                "exon_count": row["exon_count"],
                "duplicate_gene_id_count": row["duplicate_gene_id_count"],
                "malformed_line_count": row["malformed_line_count"],
                "invalid_coordinate_count": row["invalid_coordinate_count"],
                "file_sha256": row["file_sha256"],
                "status": row["status"],
            }
        )
    for row in rows(args.archive):
        paired = row["paired_genome_ids"] or "."
        unified.append(
            {
                "annotation_id": row["candidate_id"],
                "source_kind": "zip_member",
                "relative_path": row["archive_relative_path"],
                "member_name": row["member_name"],
                "source": "soyod",
                "material": row["material"],
                "annotation_role": "soyod_gene_models",
                "is_primary_gene_model": "True",
                "paired_genome_ids": paired,
                "format": row["format"],
                "feature_count": row["feature_count"],
                "gene_count": row["gene_count"],
                "transcript_count": row["transcript_count"],
                "cds_count": row["cds_count"],
                "exon_count": row["exon_count"],
                "duplicate_gene_id_count": row["duplicate_gene_id_count"],
                "malformed_line_count": row["malformed_line_count"],
                "invalid_coordinate_count": row["invalid_coordinate_count"],
                "file_sha256": row["file_sha256"],
                "status": row["status"],
            }
        )
    unified.sort(key=lambda item: str(item["annotation_id"]))
    if len({str(item["annotation_id"]) for item in unified}) != len(unified):
        raise ValueError("duplicate unified annotation IDs")
    fieldnames = list(unified[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(unified)
    tsv = buffer.getvalue().encode("utf-8")
    output = args.output_dir / "unified_annotation_catalog.tsv"
    atomic(output, tsv)
    summary = {
        "schema_version": "1.0",
        "source_count": len(unified),
        "ordinary_source_count": sum(item["source_kind"] == "file" for item in unified),
        "archive_source_count": sum(item["source_kind"] == "zip_member" for item in unified),
        "pass_count": sum(item["status"] == "PASS" for item in unified),
        "primary_gene_model_count": sum(str(item["is_primary_gene_model"]).lower() == "true" for item in unified),
        "paired_source_count": sum(item["paired_genome_ids"] != "." for item in unified),
        "unpaired_source_count": sum(item["paired_genome_ids"] == "." for item in unified),
        "gene_feature_count": sum(int(item["gene_count"]) for item in unified),
        "malformed_source_count": sum(int(item["malformed_line_count"]) > 0 for item in unified),
        "invalid_coordinate_source_count": sum(int(item["invalid_coordinate_count"]) > 0 for item in unified),
        "tsv_sha256": hashlib.sha256(tsv).hexdigest(),
    }
    atomic(args.output_dir / "unified_annotation_catalog.summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

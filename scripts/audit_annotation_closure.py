#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

from legumegenomefm.data_refinement import audit_feature_coordinates, parse_gff_features


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


@contextmanager
def annotation_text(path: Path, member_name: str) -> Iterator[TextIO]:
    if member_name == ".":
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="strict", newline="") as handle:
                yield handle
        else:
            with path.open(encoding="utf-8", errors="strict", newline="") as handle:
                yield handle
        return
    with zipfile.ZipFile(path) as archive:
        with archive.open(member_name) as raw:
            with io.TextIOWrapper(raw, encoding="utf-8", errors="strict", newline="") as handle:
                yield handle


def audit_one(task: tuple[str, dict[str, str]]) -> dict[str, object]:
    project_root_text, row = task
    project_root = Path(project_root_text)
    candidate_id = row["candidate_id"]
    result: dict[str, object] = {
        "candidate_id": candidate_id,
        "annotation_id": row["annotation_id"],
        "annotation_path": row["annotation_path"],
        "annotation_member_name": row["annotation_member_name"],
        "genome_contig_count": 0,
        "feature_count": 0,
        "gene_count": 0,
        "matched_seqid_count": 0,
        "unknown_seqid_count": 0,
        "unknown_seqids_preview": ".",
        "unknown_seqids_sha256": hashlib.sha256(b"").hexdigest(),
        "out_of_bounds_feature_count": 0,
        "error": ".",
        "status": "ERROR",
    }
    try:
        manifest = json.loads(
            (
                project_root
                / "data/processed/sequence_store"
                / candidate_id
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        contigs = {str(contig["name"]): int(contig["length"]) for contig in manifest["contigs"]}
        result["genome_contig_count"] = len(contigs)
        path = project_root / "data/raw" / row["annotation_path"]
        with annotation_text(path, row["annotation_member_name"]) as handle:
            closure = audit_feature_coordinates(contigs, parse_gff_features(handle))
        unknown = [str(value) for value in closure.pop("unknown_seqids")]
        unknown_bytes = ("\n".join(unknown) + ("\n" if unknown else "")).encode("utf-8")
        result.update(closure)
        result["unknown_seqids_preview"] = ";".join(unknown[:20]) if unknown else "."
        result["unknown_seqids_sha256"] = hashlib.sha256(unknown_bytes).hexdigest()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "ERROR"
    return result


def render_tsv(rows: list[dict[str, object]]) -> bytes:
    fieldnames = list(rows[0])
    if fieldnames[-1] != "status":
        raise ValueError("status must be the guaranteed non-empty final field")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit exact FASTA-GFF seqid and coordinate closure")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    project_root = args.project_root.resolve()
    candidates_path = args.candidates.resolve()
    with candidates_path.open(newline="", encoding="utf-8") as handle:
        selected = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["hard_gate_pass"] == "True"
        ]
    if not selected:
        raise ValueError("candidate table contains no selected rows")
    tasks = [(str(project_root), row) for row in selected]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(audit_one, tasks, chunksize=1))
    rows.sort(key=lambda row: str(row["candidate_id"]))
    if len(rows) != len(selected):
        raise ValueError("closure result count mismatch")
    tsv = render_tsv(rows)
    output_path = args.output.resolve()
    atomic_write(output_path, tsv)
    summary = {
        "schema_version": "1.0",
        "state": "PASS" if all(row["status"] == "PASS" for row in rows) else "INCOMPLETE",
        "candidate_count": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "error_count": sum(row["status"] == "ERROR" for row in rows),
        "unknown_seqid_candidate_count": sum(int(row["unknown_seqid_count"]) > 0 for row in rows),
        "out_of_bounds_candidate_count": sum(int(row["out_of_bounds_feature_count"]) > 0 for row in rows),
        "candidate_tsv_sha256": sha256(candidates_path),
        "closure_tsv_sha256": hashlib.sha256(tsv).hexdigest(),
    }
    atomic_write(
        output_path.with_suffix(".summary.json"),
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

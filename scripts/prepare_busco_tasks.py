#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare BUSCO tasks after source and coordinate gates")
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--closure", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidates_path = args.candidates.resolve()
    closure_path = args.closure.resolve()
    provenance_path = args.provenance.resolve()
    lineage_path = args.lineage.resolve()
    output_path = args.output.resolve()
    dataset_cfg = lineage_path / "dataset.cfg"
    if not dataset_cfg.is_file():
        raise FileNotFoundError(dataset_cfg)

    candidate_rows = [row for row in read_tsv(candidates_path) if row["hard_gate_pass"] == "True"]
    closure_rows = read_tsv(closure_path)
    provenance_rows = read_tsv(provenance_path)
    closure = {row["candidate_id"]: row for row in closure_rows}
    provenance = {row["candidate_id"]: row for row in provenance_rows}
    candidate_ids = {row["candidate_id"] for row in candidate_rows}
    if set(closure) != candidate_ids:
        raise ValueError("closure candidate set does not equal hard-gate candidate set")
    if set(provenance) != candidate_ids:
        raise ValueError("provenance candidate set does not equal hard-gate candidate set")

    tasks: list[dict[str, object]] = []
    exclusion_counts: dict[str, int] = {}
    excluded_ids: dict[str, list[str]] = {}
    for row in candidate_rows:
        candidate_id = row["candidate_id"]
        failures = []
        if closure[candidate_id]["status"] != "PASS":
            failures.append(f"annotation_closure_{closure[candidate_id]['status'].lower()}")
        if provenance[candidate_id]["status"] != "PASS":
            failures.append(f"source_provenance_{provenance[candidate_id]['status'].lower()}")
        if failures:
            for failure in failures:
                exclusion_counts[failure] = exclusion_counts.get(failure, 0) + 1
                excluded_ids.setdefault(failure, []).append(candidate_id)
            continue
        tasks.append(
            {
                "task_index": len(tasks) + 1,
                "candidate_id": candidate_id,
                "species": row["species"],
                "material_key": row["material_key"],
                "source": row["source"],
                "genome_relative_path": row["relative_path"],
                "genome_member_name": row["member_name"],
                "annotation_id": row["annotation_id"],
                "annotation_relative_path": row["annotation_path"],
                "annotation_member_name": row["annotation_member_name"],
                "base_count": row["base_count"],
                "task_state": "READY",
            }
        )
    tasks.sort(key=lambda row: str(row["candidate_id"]))
    for index, row in enumerate(tasks, start=1):
        row["task_index"] = index
    if not tasks:
        raise ValueError("no BUSCO tasks passed prerequisite gates")

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(tasks[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(tasks)
    tsv = buffer.getvalue().encode("utf-8")
    atomic_write(output_path, tsv)
    summary = {
        "schema_version": "1.0",
        "state": "READY",
        "hard_gate_candidate_count": len(candidate_rows),
        "task_count": len(tasks),
        "excluded_count": len(candidate_rows) - len(tasks),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "excluded_candidate_ids": {key: sorted(value) for key, value in sorted(excluded_ids.items())},
        "input_sha256": {
            "candidates": sha256(candidates_path),
            "closure": sha256(closure_path),
            "provenance": sha256(provenance_path),
            "lineage_dataset_cfg": sha256(dataset_cfg),
        },
        "task_tsv_sha256": hashlib.sha256(tsv).hexdigest(),
        "lineage_path": str(lineage_path.relative_to(Path.cwd().resolve())),
    }
    atomic_write(
        output_path.with_suffix(".summary.json"),
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

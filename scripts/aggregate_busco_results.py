#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import yaml

from legumegenomefm.data_refinement import busco_gate_passes


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


def metric(results: dict[str, object], key: str) -> float:
    value = results.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"BUSCO result {key!r} is not numeric: {value!r}")
    return float(value)


def count_metric(results: dict[str, object], key: str) -> int:
    value = results.get(key)
    if not isinstance(value, int):
        raise ValueError(f"BUSCO result {key!r} is not an integer: {value!r}")
    return value


def mode_values(shard: dict[str, object], mode: str) -> dict[str, object]:
    busco = shard.get("busco")
    if not isinstance(busco, dict) or mode not in busco or not isinstance(busco[mode], dict):
        raise ValueError(f"missing BUSCO mode {mode}")
    summary = busco[mode].get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), dict):
        raise ValueError(f"missing BUSCO {mode} summary results")
    results = summary["results"]
    return {
        "complete_percent": metric(results, "Complete percentage"),
        "single_percent": metric(results, "Single copy percentage"),
        "duplicated_percent": metric(results, "Multi copy percentage"),
        "fragmented_percent": metric(results, "Fragmented percentage"),
        "missing_percent": metric(results, "Missing percentage"),
        "complete_count": count_metric(results, "Complete BUSCOs"),
        "fragmented_count": count_metric(results, "Fragmented BUSCOs"),
        "missing_count": count_metric(results, "Missing BUSCOs"),
        "marker_count": count_metric(results, "n_markers"),
        "one_line_summary": str(results.get("one_line_summary", ".")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate fail-closed BUSCO shards")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    tasks_path = args.tasks.resolve()
    shard_dir = args.shard_dir.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    with tasks_path.open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    if not tasks:
        raise ValueError("BUSCO task manifest is empty")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    busco_config = config["busco"]
    minimum_annotation = float(busco_config["minimum_annotation_complete_percent"])
    minimum_genome = float(busco_config["minimum_genome_complete_percent"])

    expected_ids = {task["candidate_id"] for task in tasks}
    present = {path.stem for path in shard_dir.glob("*.json")}
    missing = sorted(expected_ids - present)
    extras = sorted(present - expected_ids)
    if missing or extras:
        raise ValueError(f"BUSCO shard set mismatch: missing={missing[:20]} extras={extras[:20]}")

    rows: list[dict[str, object]] = []
    for task in tasks:
        candidate_id = task["candidate_id"]
        shard_path = shard_dir / f"{candidate_id}.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "species": task["species"],
            "material_key": task["material_key"],
            "annotation_id": task["annotation_id"],
            "protein_record_count": ".",
            "annotation_complete_percent": ".",
            "annotation_single_percent": ".",
            "annotation_duplicated_percent": ".",
            "annotation_fragmented_percent": ".",
            "annotation_missing_percent": ".",
            "annotation_complete_count": ".",
            "annotation_fragmented_count": ".",
            "annotation_missing_count": ".",
            "annotation_marker_count": ".",
            "annotation_summary": ".",
            "genome_complete_percent": ".",
            "genome_single_percent": ".",
            "genome_duplicated_percent": ".",
            "genome_fragmented_percent": ".",
            "genome_missing_percent": ".",
            "genome_complete_count": ".",
            "genome_fragmented_count": ".",
            "genome_missing_count": ".",
            "genome_marker_count": ".",
            "genome_summary": ".",
            "elapsed_seconds": shard.get("elapsed_seconds", "."),
            "error": ".",
            "status": "ERROR",
        }
        try:
            if shard.get("candidate_id") != candidate_id:
                raise ValueError("candidate ID mismatch")
            if shard.get("status") != "PASS":
                raise ValueError(str(shard.get("error", "worker did not pass")))
            if set(shard.get("requested_modes", [])) != {"proteins", "genome"}:
                raise ValueError(f"required BUSCO modes missing: {shard.get('requested_modes')}")
            staging = shard.get("staging")
            if not isinstance(staging, dict):
                raise ValueError("staging summary missing")
            annotation = mode_values(shard, "proteins")
            genome = mode_values(shard, "genome")
            row["protein_record_count"] = staging.get("protein_record_count", ".")
            for prefix, values in (("annotation", annotation), ("genome", genome)):
                row[f"{prefix}_complete_percent"] = values["complete_percent"]
                row[f"{prefix}_single_percent"] = values["single_percent"]
                row[f"{prefix}_duplicated_percent"] = values["duplicated_percent"]
                row[f"{prefix}_fragmented_percent"] = values["fragmented_percent"]
                row[f"{prefix}_missing_percent"] = values["missing_percent"]
                row[f"{prefix}_complete_count"] = values["complete_count"]
                row[f"{prefix}_fragmented_count"] = values["fragmented_count"]
                row[f"{prefix}_missing_count"] = values["missing_count"]
                row[f"{prefix}_marker_count"] = values["marker_count"]
                row[f"{prefix}_summary"] = values["one_line_summary"]
            row["status"] = (
                "PASS"
                if busco_gate_passes(
                    float(annotation["complete_percent"]),
                    float(genome["complete_percent"]),
                    minimum_annotation,
                    minimum_genome,
                )
                else "FAIL"
            )
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    rows.sort(key=lambda row: str(row["candidate_id"]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    tsv = buffer.getvalue().encode("utf-8")
    atomic_write(output_path, tsv)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    summary = {
        "schema_version": "1.0",
        "state": "COMPLETE" if counts.get("ERROR", 0) == 0 else "INCOMPLETE",
        "candidate_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "lineage": str(busco_config["lineage"]),
        "minimum_annotation_complete_percent": minimum_annotation,
        "minimum_genome_complete_percent": minimum_genome,
        "input_sha256": {
            "tasks": sha256(tasks_path),
            "config": sha256(config_path),
        },
        "busco_tsv_sha256": hashlib.sha256(tsv).hexdigest(),
    }
    atomic_write(
        output_path.with_suffix(".summary.json"),
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

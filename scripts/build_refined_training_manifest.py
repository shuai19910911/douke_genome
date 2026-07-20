#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml

from legumegenomefm.training_data import build_refined_training_manifest


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
    parser = argparse.ArgumentParser(description="Build the interval-filtered schema-2 training release")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--selected", required=True, type=Path)
    parser.add_argument("--intervals", required=True, type=Path)
    parser.add_argument("--contexts", required=True, type=Path)
    parser.add_argument("--final-summary", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--store-root-reference", default="data/processed/sequence_store")
    parser.add_argument("--cold-genus", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    selected_path = args.selected.resolve()
    interval_path = args.intervals.resolve()
    context_path = args.contexts.resolve()
    final_summary_path = args.final_summary.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    ready_path = output_path.parent / "TRAINING_DATASET_READY"
    if ready_path.exists():
        raise FileExistsError(f"refusing to replace a READY training release: {ready_path}")

    final_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
    if final_summary.get("state") != "CANDIDATE_SET_READY_DATA_NOT_YET_PACKAGED":
        raise ValueError("final refinement summary is not ready for packaging")
    expected_outputs = final_summary.get("output_sha256", {})
    for key, path in (("selected", selected_path), ("intervals", interval_path), ("contexts", context_path)):
        if expected_outputs.get(key) != sha256(path):
            raise ValueError(f"final refinement output hash mismatch: {key}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contexts = tuple(int(value) for value in config["contexts"])
    cold_genera = set(args.cold_genus)
    provenance = {
        "data_refinement_config_sha256": sha256(config_path),
        "final_summary_sha256": sha256(final_summary_path),
        "selected_sha256": sha256(selected_path),
        "intervals_sha256": sha256(interval_path),
        "contexts_sha256": sha256(context_path),
    }
    result = build_refined_training_manifest(
        read_tsv(selected_path),
        read_tsv(interval_path),
        read_tsv(context_path),
        args.store_root.resolve(),
        output_path,
        store_root_reference=args.store_root_reference,
        cold_genera=cold_genera,
        contexts=contexts,
        provenance=provenance,
    )
    if result.summary["source_count"] != int(final_summary["selected_candidate_count"]):
        raise ValueError("release source count does not match final refinement selection")

    implementation = hashlib.sha256(b"refined-training-manifest-v2\0")
    implementation.update(Path(__file__).read_bytes())
    implementation.update(Path(__import__("legumegenomefm.training_data", fromlist=["x"]).__file__).read_bytes())
    receipt = {
        "schema_version": "2.0",
        "state": "READY",
        "dataset_manifest_sha256": sha256(result.manifest_path),
        "dataset_summary_sha256": sha256(result.summary_path),
        "implementation_sha256": implementation.hexdigest(),
        "inputs": provenance,
        "cold_genera": sorted(cold_genera),
        "summary": result.summary,
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    receipt_path = output_path.parent / "training_dataset.release.json"
    atomic_write(receipt_path, receipt_bytes)
    atomic_write(ready_path, (hashlib.sha256(receipt_bytes).hexdigest() + "\n").encode("ascii"))
    print(json.dumps({**result.summary, "release_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

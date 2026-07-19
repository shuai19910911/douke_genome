#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import yaml

from legumegenomefm.genome_sketch import read_sketch_registry
from legumegenomefm.training_data import build_training_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, payload: bytes) -> None:
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


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the formal pretraining source manifest")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sketch-registry", required=True, type=Path)
    parser.add_argument("--near-clusters", required=True, type=Path)
    parser.add_argument("--orientation-identity", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--store-root-reference", default="data/processed/sequence_store")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    clusters = {row["candidate_id"]: row for row in rows(args.near_clusters)}
    orientations = {row["candidate_id"]: row for row in rows(args.orientation_identity)}
    source_rows: list[dict[str, object]] = []
    candidates = read_sketch_registry(args.sketch_registry)
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if set(orientations) != candidate_ids:
        raise ValueError("orientation assignments do not match sketch registry")
    included = [candidate for candidate in candidates if orientations[candidate.candidate_id]["orientation_representative"] == "True"]
    material_groups: dict[tuple[str, str], list[str]] = {}
    for candidate in included:
        material_groups.setdefault((candidate.species, candidate.material_key), []).append(candidate.candidate_id)
    for candidate in candidates:
        orientation = orientations[candidate.candidate_id]
        if orientation["orientation_representative"] != "True":
            continue
        cluster = clusters.get(candidate.candidate_id)
        if cluster is None:
            raise ValueError(f"missing near-duplicate assignment: {candidate.candidate_id}")
        material_members = material_groups[(candidate.species, candidate.material_key)]
        material_digest = hashlib.sha256(f"{candidate.species}\0{candidate.material_key}".encode("utf-8")).hexdigest()[:16]
        source_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "genus": candidate.species.split()[0],
                "species": candidate.species,
                "material_key": candidate.material_key,
                "material_version_group_id": f"material-{material_digest}",
                "material_version_group_size": len(material_members),
                "near_duplicate_group_id": cluster["near_duplicate_group_id"],
                "near_duplicate_group_size": int(cluster["near_duplicate_group_size"]),
            }
        )
    if set(clusters) != candidate_ids:
        raise ValueError("near-duplicate assignments contain unexpected candidates")
    result = build_training_manifest(
        source_rows,
        args.store_root,
        args.output,
        store_root_reference=args.store_root_reference,
        cold_genera=set(config["cold_genera"]),
        max_context=int(config["max_context"]),
    )
    implementation = hashlib.sha256(b"training-manifest-producer-v1\0")
    implementation.update(Path(__file__).read_bytes())
    implementation.update(Path(__import__("legumegenomefm.training_data", fromlist=["x"]).__file__).read_bytes())
    receipt = {
        "schema_version": "1.0",
        "state": "READY",
        "dataset_manifest_sha256": sha256(result.manifest_path),
        "dataset_summary_sha256": sha256(result.summary_path),
        "implementation_sha256": implementation.hexdigest(),
        "inputs": {
            "data_freeze_config_sha256": sha256(args.config),
            "sketch_registry_sha256": sha256(args.sketch_registry),
            "near_duplicate_clusters_sha256": sha256(args.near_clusters),
            "orientation_identity_sha256": sha256(args.orientation_identity),
        },
        "summary": result.summary,
    }
    receipt_bytes = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    receipt_path = args.output.parent / "training_dataset.release.json"
    atomic(receipt_path, receipt_bytes)
    atomic(args.output.parent / "TRAINING_DATASET_READY", (hashlib.sha256(receipt_bytes).hexdigest() + "\n").encode("ascii"))
    print(json.dumps({**result.summary, "release_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

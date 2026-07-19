#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from legumegenomefm.genome_sketch import read_sketch_registry
from legumegenomefm.training_data import build_training_manifest


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
    for candidate in candidates:
        orientation = orientations[candidate.candidate_id]
        if orientation["orientation_representative"] != "True":
            continue
        cluster = clusters.get(candidate.candidate_id)
        if cluster is None:
            raise ValueError(f"missing near-duplicate assignment: {candidate.candidate_id}")
        source_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "genus": candidate.species.split()[0],
                "species": candidate.species,
                "material_key": candidate.material_key,
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
    print(json.dumps(result.summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

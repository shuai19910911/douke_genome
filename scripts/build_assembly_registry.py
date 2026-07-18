#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.assembly_audit import (
    build_assembly_candidates,
    write_assembly_registry,
    write_shard_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic genome assembly candidates and QC shards")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = build_assembly_candidates(args.inventory)
    registry = write_assembly_registry(candidates, args.output_dir)
    shards = write_shard_manifests(candidates, args.shard_dir, args.shard_count)
    print(
        json.dumps(
            {
                "state": "PASS",
                "candidate_count": registry.candidate_count,
                "registry": registry.registry_tsv.name,
                "registry_sha256": registry.registry_sha256,
                "shard_count": shards.shard_count,
                "shard_summary": shards.summary_json.name,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.annotation_audit import (
    build_annotation_candidates,
    write_annotation_registry,
    write_annotation_shard_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build annotation audit registry and shards")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--assembly-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = build_annotation_candidates(args.inventory, args.assembly_registry)
    registry = write_annotation_registry(candidates, args.output_dir)
    shards = write_annotation_shard_manifests(candidates, args.shard_dir, args.shard_count)
    print(
        json.dumps(
            {
                "candidate_count": registry.candidate_count,
                "registry": registry.registry_path.name,
                "shard_count": shards.shard_count,
                "shard_summary": shards.summary_path.name,
                "state": "COMPLETE",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

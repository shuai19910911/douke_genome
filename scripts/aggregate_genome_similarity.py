#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.genome_similarity import aggregate_genome_sketches


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate genome MinHash signatures into near-duplicate groups")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--related-threshold", type=float, default=0.80)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.95)
    args = parser.parse_args()
    result = aggregate_genome_sketches(
        args.registry,
        args.result_dir,
        args.output_dir,
        related_threshold=args.related_threshold,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    print(json.dumps(result.summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.genome_sketch import build_sketch_candidates, write_sketch_registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Build exact-representative genome sketch registry")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = write_sketch_registry(build_sketch_candidates(args.catalog), args.output_dir)
    print(json.dumps({"candidate_count": result.candidate_count, "registry": str(result.registry_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

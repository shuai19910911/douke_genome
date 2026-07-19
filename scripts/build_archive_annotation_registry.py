#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.archive_annotation_audit import (
    build_archive_annotation_candidates,
    write_archive_annotation_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ZIP-backed annotation candidate registry")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--archive-qc", required=True, type=Path)
    parser.add_argument("--archive-members", required=True, type=Path)
    parser.add_argument("--archive-genomes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    candidates = build_archive_annotation_candidates(
        args.inventory, args.archive_qc, args.archive_members, args.archive_genomes
    )
    result = write_archive_annotation_registry(candidates, args.output_dir)
    print(json.dumps({"candidate_count": result.candidate_count, "registry": str(result.registry_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

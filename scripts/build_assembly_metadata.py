#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.metadata_integration import build_assembly_metadata, write_assembly_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate assembly taxon and source metadata")
    parser.add_argument("--assembly-registry", required=True, type=Path)
    parser.add_argument("--annotation-registry", required=True, type=Path)
    parser.add_argument("--soyomics-metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_assembly_metadata(
        args.assembly_registry, args.annotation_registry, args.soyomics_metadata
    )
    result = write_assembly_metadata(records, args.output_dir)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

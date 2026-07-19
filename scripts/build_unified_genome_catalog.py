#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.genome_catalog import build_unified_genome_catalog, write_unified_genome_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified source-level genome QC catalog")
    parser.add_argument("--fasta-qc", required=True, type=Path)
    parser.add_argument("--assembly-metadata", required=True, type=Path)
    parser.add_argument("--archive-genome-qc", required=True, type=Path)
    parser.add_argument("--archive-taxa", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = build_unified_genome_catalog(
        args.fasta_qc,
        args.assembly_metadata,
        args.archive_genome_qc,
        args.archive_taxa,
    )
    result = write_unified_genome_catalog(records, args.output_dir)
    print(json.dumps(result.summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

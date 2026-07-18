#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.archive_sequence_summary import (
    aggregate_archive_genome_audit,
    write_archive_genome_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SoyOD ZIP genome member QC results")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = aggregate_archive_genome_audit(args.registry, args.result_dir)
    if args.require_complete and audit.summary["missing_count"]:
        raise RuntimeError(
            f"archive genome audit is incomplete: {audit.summary['missing_count']} results missing"
        )
    manifest, summary = write_archive_genome_audit(audit, args.output_dir)
    print(
        json.dumps(
            {
                **audit.summary,
                "manifest": manifest.name,
                "summary": summary.name,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

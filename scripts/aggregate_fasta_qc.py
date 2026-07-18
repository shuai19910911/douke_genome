#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.audit_summary import aggregate_fasta_audit, write_fasta_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-assembly FASTA audit results")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = aggregate_fasta_audit(args.registry, args.result_root)
    if args.require_complete and audit.summary["missing_count"]:
        raise RuntimeError(
            f"audit is incomplete: {audit.summary['missing_count']} candidate results are missing"
        )
    result = write_fasta_audit(audit, args.output_dir)
    print(
        json.dumps(
            {
                **audit.summary,
                "manifest": result.manifest_path.name,
                "summary": result.summary_path.name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

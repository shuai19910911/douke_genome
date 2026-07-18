#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.data_inventory import write_inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a deterministic raw-data inventory.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_inventory(args.data_root, args.output_dir)
    summary = json.loads(result.summary_json.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "state": "PASS",
                "file_count": summary["file_count"],
                "record_count": summary["record_count"],
                "total_bytes": summary["total_bytes"],
                "inventory_sha256": result.inventory_sha256,
                "inventory_tsv": result.inventory_tsv.name,
                "summary_json": result.summary_json.name,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

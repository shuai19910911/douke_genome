#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.reference_integrity import validate_busco_lineage


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a BUSCO lineage against its immutable receipt")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    receipt_sha = validate_busco_lineage(
        args.project_root.resolve(),
        args.lineage.resolve(),
        full=args.full,
    )
    print(
        json.dumps(
            {
                "state": "PASS",
                "lineage_id": args.lineage.resolve().name,
                "lineage_receipt_sha256": receipt_sha,
                "verification": "full_tree" if args.full else "path_size_inventory",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

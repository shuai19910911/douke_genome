#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.reference_integrity import (
    load_contamination_legacy_bindings,
    validate_contamination_references,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen Tiara, UniVec, BLAST, and legacy contamination bindings")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tiara-image", required=True, type=Path)
    parser.add_argument("--univec-db", required=True, type=Path)
    parser.add_argument("--qc-env", required=True, type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    reference_sha = validate_contamination_references(
        project_root,
        args.tiara_image.resolve(),
        args.univec_db.resolve(),
        args.qc_env.resolve() / "bin/blastn",
        full=args.full,
    )
    legacy = load_contamination_legacy_bindings(project_root, reference_sha)
    print(
        json.dumps(
            {
                "contamination_reference_receipt_sha256": reference_sha,
                "legacy_bound_shard_count": len(legacy),
                "state": "PASS",
                "verification": "full" if args.full else "inventory",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

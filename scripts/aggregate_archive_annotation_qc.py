#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from legumegenomefm.archive_annotation_audit import aggregate_archive_annotation_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly aggregate ZIP-backed annotation audit results")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate_archive_annotation_audit(args.registry, args.result_dir, args.output_dir)
    print(json.dumps(result.summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

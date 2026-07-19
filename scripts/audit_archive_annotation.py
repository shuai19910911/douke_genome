#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import legumegenomefm.annotation_audit as annotation_module
import legumegenomefm.archive_annotation_audit as archive_annotation_module
from legumegenomefm.archive_annotation_audit import audit_archive_annotation, read_archive_annotation_registry


def implementation_sha256() -> str:
    digest = hashlib.sha256(b"legumegenomefm-archive-annotation-audit-v1\0")
    digest.update(Path(annotation_module.__file__).read_bytes())
    digest.update(Path(archive_annotation_module.__file__).read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one ZIP-backed annotation candidate")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--temporary-dir", required=True, type=Path)
    args = parser.parse_args()
    candidates = read_archive_annotation_registry(args.registry)
    if args.index < 0 or args.index >= len(candidates):
        raise ValueError(f"candidate index out of range: {args.index}")
    candidate = candidates[args.index]
    result = audit_archive_annotation(
        args.data_root,
        candidate,
        args.output_dir,
        implementation_sha256(),
        temporary_dir=args.temporary_dir,
    )
    print(json.dumps({"candidate_id": candidate.candidate_id, "index": args.index, "reused": result.reused, "state": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import legumegenomefm.genome_sketch as sketch_module
import legumegenomefm.sequence_store as store_module
from legumegenomefm.genome_sketch import audit_genome_sketch, read_sketch_registry


def implementation_sha256() -> str:
    digest = hashlib.sha256(b"legumegenomefm-genome-sketch-v1\0")
    digest.update(Path(sketch_module.__file__).read_bytes())
    digest.update(Path(store_module.__file__).read_bytes())
    digest.update(Path(__file__).read_bytes())
    digest.update(version("sourmash").encode("ascii"))
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one RC-canonical genome MinHash signature")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--ksize", type=int, default=31)
    parser.add_argument("--scaled", type=int, default=10_000)
    args = parser.parse_args()
    candidates = read_sketch_registry(args.registry)
    if args.index < 0 or args.index >= len(candidates):
        raise ValueError(f"candidate index out of range: {args.index}")
    candidate = candidates[args.index]
    result = audit_genome_sketch(
        args.data_root,
        candidate,
        args.output_dir,
        implementation_sha256(),
        ksize=args.ksize,
        scaled=args.scaled,
        store_root=args.store_root,
    )
    print(json.dumps({"candidate_id": candidate.candidate_id, "index": args.index, "reused": result.reused, "state": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

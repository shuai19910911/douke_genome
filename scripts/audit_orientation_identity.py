#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import legumegenomefm.orientation_identity as identity_module
import legumegenomefm.sequence_store as store_module
from legumegenomefm.genome_sketch import read_sketch_registry
from legumegenomefm.orientation_identity import orientation_invariant_store_signature
from legumegenomefm.sequence_store import PackedSequenceStore


def implementation_hash() -> str:
    digest = hashlib.sha256(b"orientation-identity-worker-v1\0")
    digest.update(Path(identity_module.__file__).read_bytes())
    digest.update(Path(store_module.__file__).read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute one orientation/order-invariant genome signature")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    candidates = read_sketch_registry(args.registry)
    if args.index < 0 or args.index >= len(candidates):
        raise IndexError("candidate index is outside registry")
    candidate = candidates[args.index]
    expected_candidate = asdict(candidate)
    impl = implementation_hash()
    output = args.output_dir / f"{candidate.candidate_id}.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("state") == "PASS" and payload.get("candidate") == expected_candidate and payload.get("implementation_sha256") == impl:
            print(json.dumps({"candidate_id": candidate.candidate_id, "reused": True}, sort_keys=True))
            return 0
    store_dir = args.store_root / candidate.candidate_id
    store = PackedSequenceStore(store_dir)
    if store.manifest.get("identity", {}).get("candidate_id") != candidate.candidate_id:
        raise ValueError("sequence store candidate identity mismatch")
    result = orientation_invariant_store_signature(store_dir)
    manifest_sha = hashlib.sha256((store_dir / "manifest.json").read_bytes()).hexdigest()
    payload = {
        "schema_version": "1.0",
        "state": "PASS",
        "candidate": expected_candidate,
        "implementation_sha256": impl,
        "store_manifest_sha256": manifest_sha,
        "orientation_signature": result.assembly_signature,
        "total_symbols": result.total_symbols,
        "contig_count": result.contig_count,
    }
    atomic_json(output, payload)
    print(json.dumps({"candidate_id": candidate.candidate_id, "reused": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

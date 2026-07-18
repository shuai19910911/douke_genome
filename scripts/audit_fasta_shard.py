#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import fields
from pathlib import Path

import legumegenomefm.assembly_audit as assembly_audit
from legumegenomefm.assembly_audit import AssemblyCandidate, audit_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable streaming FASTA QC for one deterministic shard")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(assembly_audit.__file__).resolve(), Path(__file__).resolve()), key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def read_shard(path: Path) -> list[AssemblyCandidate]:
    expected = [field.name for field in fields(AssemblyCandidate)]
    candidates: list[AssemblyCandidate] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected:
            raise ValueError("shard schema mismatch")
        for row in reader:
            if not re.fullmatch(r"[0-9a-f]{16}", row["candidate_id"]):
                raise ValueError("invalid candidate ID")
            candidates.append(
                AssemblyCandidate(
                    candidate_id=row["candidate_id"],
                    relative_path=row["relative_path"],
                    source=row["source"],
                    genus=row["genus"],
                    species=row["species"],
                    assembly_label=row["assembly_label"],
                    genome_role=row["genome_role"],
                    size_bytes=int(row["size_bytes"]),
                    mtime_ns=int(row["mtime_ns"]),
                )
            )
    return candidates


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    candidates = read_shard(args.shard)
    implementation_sha256 = implementation_digest()
    passed = 0
    reused = 0
    failures: list[dict[str, str]] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        try:
            result = audit_candidate(args.data_root, candidate, args.output_dir / "assemblies", implementation_sha256)
            passed += 1
            reused += int(result.reused)
        except Exception as error:  # keep the shard auditable instead of losing later candidates
            failures.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "relative_path": candidate.relative_path,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
        if index == 1 or index == total or index % 10 == 0:
            print(f"progress={index}/{total} passed={passed} reused={reused} failed={len(failures)}", file=sys.stderr, flush=True)
    state = "PASS" if not failures else "FAIL"
    summary = {
        "schema_version": "1.0",
        "state": state,
        "shard": args.shard.name,
        "implementation_sha256": implementation_sha256,
        "candidate_count": total,
        "passed_count": passed,
        "reused_count": reused,
        "failed_count": len(failures),
        "failures": failures,
    }
    summary_path = args.output_dir / "runs" / f"{args.shard.stem}.json"
    atomic_json(summary_path, summary)
    print(json.dumps({key: summary[key] for key in ("state", "shard", "candidate_count", "passed_count", "reused_count", "failed_count")}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

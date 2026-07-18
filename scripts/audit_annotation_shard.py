#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import legumegenomefm.annotation_audit as annotation_module
from legumegenomefm.annotation_audit import AnnotationCandidate, audit_annotation_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one annotation candidate shard")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_candidates(path: Path) -> list[AnnotationCandidate]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    candidates: list[AnnotationCandidate] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        if not re.fullmatch(r"[0-9a-f]{16}", candidate_id):
            raise ValueError(f"invalid annotation candidate ID: {candidate_id}")
        candidates.append(
            AnnotationCandidate(
                candidate_id=candidate_id,
                relative_path=row["relative_path"],
                source=row["source"],
                annotation_role=row["annotation_role"],
                is_primary_gene_model=row["is_primary_gene_model"].lower() == "true",
                assembly_key=row["assembly_key"],
                paired_assembly_ids=tuple(filter(None, row["paired_assembly_ids"].split(";"))),
                size_bytes=int(row["size_bytes"]),
                mtime_ns=int(row["mtime_ns"]),
            )
        )
    return candidates


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    digest.update(b"legumegenomefm-annotation-audit-v1\0")
    digest.update(Path(annotation_module.__file__).read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    implementation = implementation_sha256()
    candidates = load_candidates(args.shard)
    passed = 0
    reused = 0
    failures: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates, 1):
        try:
            result = audit_annotation_candidate(
                args.data_root, candidate, args.output_dir, implementation
            )
            passed += 1
            reused += int(result.reused)
        except Exception as exc:
            failures.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "relative_path": candidate.relative_path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        if index == 1 or index % 20 == 0 or index == len(candidates):
            print(
                f"progress={index}/{len(candidates)} passed={passed} reused={reused} failed={len(failures)}",
                file=sys.stderr,
                flush=True,
            )
    state = "PASS" if not failures else "FAIL"
    summary = {
        "schema_version": "1.0",
        "state": state,
        "shard": args.shard.name,
        "implementation_sha256": implementation,
        "candidate_count": len(candidates),
        "passed_count": passed,
        "reused_count": reused,
        "failed_count": len(failures),
        "failures": failures,
    }
    atomic_write_json(args.output_dir / "runs" / f"{args.shard.stem}.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

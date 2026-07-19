#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from dataclasses import asdict
from pathlib import Path

from legumegenomefm.genome_sketch import read_sketch_registry
from legumegenomefm.orientation_identity import assign_orientation_groups


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Strictly aggregate orientation-invariant genome signatures")
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    candidates = read_sketch_registry(args.registry)
    expected = {candidate.candidate_id: candidate for candidate in candidates}
    files = {path.stem: path for path in args.result_dir.glob("*.json")}
    if set(files) != set(expected):
        missing = sorted(set(expected) - set(files))
        extra = sorted(set(files) - set(expected))
        raise ValueError(f"orientation result set mismatch: missing={missing[:5]} extra={extra[:5]}")
    items: list[tuple[str, str]] = []
    implementations: set[str] = set()
    for candidate_id in sorted(expected):
        payload = json.loads(files[candidate_id].read_text(encoding="utf-8"))
        if payload.get("state") != "PASS" or payload.get("candidate") != asdict(expected[candidate_id]):
            raise ValueError(f"invalid orientation result identity: {candidate_id}")
        signature = str(payload.get("orientation_signature", ""))
        if len(signature) != 64 or any(character not in "0123456789abcdef" for character in signature):
            raise ValueError(f"invalid orientation signature: {candidate_id}")
        implementations.add(str(payload.get("implementation_sha256", "")))
        items.append((candidate_id, signature))
    if len(implementations) != 1 or "" in implementations:
        raise ValueError("orientation results contain implementation drift")
    rows = assign_orientation_groups(items)
    fieldnames = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    tsv_bytes = buffer.getvalue().encode("utf-8")
    output_tsv = args.output_dir / "orientation_identity.tsv"
    atomic_bytes(output_tsv, tsv_bytes)
    group_sizes: dict[str, int] = {}
    for row in rows:
        group_sizes[str(row["orientation_group_id"])] = int(row["orientation_group_size"])
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(rows),
        "orientation_group_count": len(group_sizes),
        "duplicate_group_count": sum(size > 1 for size in group_sizes.values()),
        "duplicate_member_count": sum(size for size in group_sizes.values() if size > 1),
        "representative_count": sum(bool(row["orientation_representative"]) for row in rows),
        "implementation_sha256": next(iter(implementations)),
        "registry_sha256": hashlib.sha256(args.registry.read_bytes()).hexdigest(),
        "tsv_sha256": hashlib.sha256(tsv_bytes).hexdigest(),
    }
    atomic_bytes(args.output_dir / "orientation_identity.summary.json", (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

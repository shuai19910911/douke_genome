#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

from legumegenomefm.archive_sequence_audit import (
    ArchiveGenomeCandidate,
    build_archive_genome_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build verified SoyOD ZIP genome candidate registry")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--archive-qc", required=True, type=Path)
    parser.add_argument("--archive-members", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    candidates = build_archive_genome_candidates(
        args.inventory,
        args.archive_qc,
        args.archive_members,
    )
    if not candidates:
        raise RuntimeError("no verified archive genome candidates")
    field_names = [field.name for field in fields(ArchiveGenomeCandidate)]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=field_names, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for candidate in candidates:
        writer.writerow(asdict(candidate))
    payload = stream.getvalue().encode("utf-8")
    manifest_path = args.output_dir / "archive_genome_candidates.tsv"
    atomic_write(manifest_path, payload)
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(candidates),
        "material_count": len({candidate.material for candidate in candidates}),
        "total_uncompressed_bytes": sum(
            candidate.member_uncompressed_bytes for candidate in candidates
        ),
        "counts_by_material": dict(sorted(Counter(candidate.material for candidate in candidates).items())),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }
    atomic_write(
        args.output_dir / "archive_genome_candidates.summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import legumegenomefm.archive_sequence_audit as archive_sequence_audit
import legumegenomefm.assembly_audit as assembly_audit
from legumegenomefm.archive_sequence_audit import ArchiveGenomeCandidate, audit_archive_genome


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit one verified genome member inside a ZIP archive")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--temporary-dir", required=True, type=Path)
    return parser.parse_args()


def implementation_digest() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(archive_sequence_audit.__file__).resolve(),
        Path(assembly_audit.__file__).resolve(),
        Path(__file__).resolve(),
    )
    for path in sorted(paths, key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def read_candidate(path: Path, index: int) -> ArchiveGenomeCandidate:
    expected = [field.name for field in fields(ArchiveGenomeCandidate)]
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected:
            raise ValueError("archive genome registry schema mismatch")
        rows = list(reader)
    if index < 0 or index >= len(rows):
        raise IndexError(f"candidate index out of range: {index}")
    row = rows[index]
    return ArchiveGenomeCandidate(
        candidate_id=row["candidate_id"],
        archive_relative_path=row["archive_relative_path"],
        member_name=row["member_name"],
        material=row["material"],
        archive_size_bytes=int(row["archive_size_bytes"]),
        archive_mtime_ns=int(row["archive_mtime_ns"]),
        archive_sha256=row["archive_sha256"],
        member_uncompressed_bytes=int(row["member_uncompressed_bytes"]),
        member_crc32_hex=row["member_crc32_hex"],
    )


def main() -> None:
    args = parse_args()
    candidate = read_candidate(args.registry, args.index)
    result = audit_archive_genome(
        args.data_root,
        candidate,
        args.output_dir,
        implementation_digest(),
        temporary_dir=args.temporary_dir,
    )
    print(
        json.dumps(
            {
                "candidate_id": candidate.candidate_id,
                "index": args.index,
                "relative_path": candidate.archive_relative_path,
                "reused": result.reused,
                "state": "PASS",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

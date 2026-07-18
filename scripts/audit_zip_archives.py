#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import legumegenomefm.archive_audit as archive_module
from legumegenomefm.archive_audit import scan_zip_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ZIP archives with SHA-256 and member CRC checks")
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-archives", type=int)
    return parser.parse_args()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def implementation_sha256() -> str:
    digest = hashlib.sha256()
    for path in (Path(archive_module.__file__).resolve(), Path(__file__).resolve()):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_candidates(inventory: Path) -> list[dict[str, str]]:
    with inventory.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["kind"] == "file" and row["file_type"] == "archive"
        ]
    return sorted(rows, key=lambda row: row["relative_path"])


def confined_target(data_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("archive path is not confined to data root")
    root = data_root.resolve(strict=True)
    target = (root / candidate).resolve(strict=True)
    if os.path.commonpath((str(root), str(target))) != str(root):
        raise ValueError("archive path is not confined to data root")
    return target


def valid_cached(payload: dict[str, object], row: dict[str, str], implementation: str) -> bool:
    return (
        payload.get("state") == "PASS"
        and payload.get("relative_path") == row["relative_path"]
        and payload.get("size_bytes") == int(row["size_bytes"])
        and payload.get("mtime_ns") == int(row["mtime_ns"])
        and payload.get("implementation_sha256") == implementation
    )


def render_tsv(rows: list[dict[str, object]], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def main() -> int:
    args = parse_args()
    candidates = load_candidates(args.inventory)
    if args.max_archives is not None:
        candidates = candidates[: args.max_archives]
    if not candidates:
        raise RuntimeError("no archives selected")
    implementation = implementation_sha256()
    args.result_root.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    reused = 0
    for index, row in enumerate(candidates, start=1):
        candidate_id = hashlib.sha256(row["relative_path"].encode("utf-8")).hexdigest()[:16]
        result_path = args.result_root / f"{candidate_id}.json"
        cached: dict[str, object] | None = None
        if result_path.is_file():
            try:
                candidate_payload = json.loads(result_path.read_text(encoding="utf-8"))
                if valid_cached(candidate_payload, row, implementation):
                    cached = candidate_payload
            except (json.JSONDecodeError, OSError):
                cached = None
        if cached is not None:
            payloads.append(cached)
            reused += 1
        else:
            try:
                target = confined_target(args.data_root, row["relative_path"])
                stat_result = target.stat()
                if stat_result.st_size != int(row["size_bytes"]):
                    raise ValueError("archive size changed after inventory")
                if stat_result.st_mtime_ns != int(row["mtime_ns"]):
                    raise ValueError("archive mtime changed after inventory")
                audit = scan_zip_archive(target, verify_crc=True)
                payload = {
                    "schema_version": "1.0",
                    "state": "PASS",
                    "candidate_id": candidate_id,
                    "relative_path": row["relative_path"],
                    "size_bytes": int(row["size_bytes"]),
                    "mtime_ns": int(row["mtime_ns"]),
                    "implementation_sha256": implementation,
                    "audit": asdict(audit),
                }
                atomic_write(result_path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
                payloads.append(payload)
            except Exception as error:
                failures.append(
                    {
                        "candidate_id": candidate_id,
                        "relative_path": row["relative_path"],
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
        if index == 1 or index % 10 == 0 or index == len(candidates):
            print(
                f"progress={index}/{len(candidates)} passed={len(payloads)} reused={reused} failed={len(failures)}",
                file=sys.stderr,
                flush=True,
            )

    archive_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    counts_by_member_type: Counter[str] = Counter()
    for payload in sorted(payloads, key=lambda item: str(item["relative_path"])):
        audit = payload["audit"]
        members = audit["members"]
        archive_rows.append(
            {
                "candidate_id": payload["candidate_id"],
                "relative_path": payload["relative_path"],
                "size_bytes": payload["size_bytes"],
                "file_sha256": audit["file_sha256"],
                "file_member_count": audit["file_member_count"],
                "total_uncompressed_bytes": audit["total_uncompressed_bytes"],
                "crc_verified_count": audit["crc_verified_count"],
                "crc_failure_count": audit["crc_failure_count"],
                "encrypted_member_count": audit["encrypted_member_count"],
                "unsafe_member_count": audit["unsafe_member_count"],
                "high_compression_ratio_count": audit["high_compression_ratio_count"],
                "status": "PASS",
            }
        )
        for member in members:
            counts_by_member_type[member["member_type"]] += 1
            member_rows.append(
                {
                    "archive_candidate_id": payload["candidate_id"],
                    "archive_relative_path": payload["relative_path"],
                    "member_name": member["member_name"],
                    "member_type": member["member_type"],
                    "uncompressed_bytes": member["uncompressed_bytes"],
                    "compressed_bytes": member["compressed_bytes"],
                    "crc32_hex": member["crc32_hex"],
                    "encrypted": str(member["encrypted"]).lower(),
                    "safe_path": str(member["safe_path"]).lower(),
                    "crc_verified": str(member["crc_verified"]).lower(),
                    "error_type": member["error_type"],
                    "status": "PASS" if member["crc_verified"] else "NOT_VERIFIED",
                }
            )
    archive_fields = list(archive_rows[0])
    member_fields = list(member_rows[0]) if member_rows else []
    archive_manifest = render_tsv(archive_rows, archive_fields)
    member_manifest = render_tsv(member_rows, member_fields) if member_rows else b""
    atomic_write(args.output_dir / "archive_qc.tsv", archive_manifest)
    atomic_write(args.output_dir / "archive_members.tsv", member_manifest)
    summary = {
        "schema_version": "1.0",
        "candidate_count": len(candidates),
        "pass_count": len(payloads),
        "fail_count": len(failures),
        "reused_count": reused,
        "member_count": len(member_rows),
        "counts_by_member_type": dict(sorted(counts_by_member_type.items())),
        "crc_failure_archive_count": sum(int(row["crc_failure_count"]) > 0 for row in archive_rows),
        "unsafe_archive_count": sum(int(row["unsafe_member_count"]) > 0 for row in archive_rows),
        "encrypted_archive_count": sum(int(row["encrypted_member_count"]) > 0 for row in archive_rows),
        "implementation_sha256": implementation,
        "archive_manifest_sha256": hashlib.sha256(archive_manifest).hexdigest(),
        "member_manifest_sha256": hashlib.sha256(member_manifest).hexdigest(),
        "failures": failures,
    }
    atomic_write(
        args.output_dir / "archive_qc.summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(summary, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def valid_combined(value: dict[str, object] | None, candidate_id: str) -> bool:
    return bool(
        value
        and value.get("candidate_id") == candidate_id
        and value.get("status") == "PASS"
        and set(value.get("requested_modes", [])) == {"proteins", "genome"}
        and isinstance(value.get("busco"), dict)
        and set(value["busco"]) == {"proteins", "genome"}
    )


def valid_mode(value: dict[str, object] | None, candidate_id: str, mode: str) -> bool:
    return bool(
        value
        and value.get("candidate_id") == candidate_id
        and value.get("status") == "PASS"
        and value.get("requested_modes") == [mode]
        and isinstance(value.get("busco"), dict)
        and set(value["busco"]) == {mode}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge protein-only and genome-only BUSCO shards")
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--combined-dir", required=True, type=Path)
    parser.add_argument("--protein-dir", required=True, type=Path)
    parser.add_argument("--genome-dir", required=True, type=Path)
    args = parser.parse_args()
    with args.tasks.resolve().open(newline="", encoding="utf-8") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))
    combined_dir = args.combined_dir.resolve()
    protein_dir = args.protein_dir.resolve()
    genome_dir = args.genome_dir.resolve()
    combined_dir.mkdir(parents=True, exist_ok=True)
    merged_count = 0
    already_complete = 0
    incomplete: list[str] = []
    for task in tasks:
        candidate_id = task["candidate_id"]
        combined_path = combined_dir / f"{candidate_id}.json"
        if valid_combined(read_json(combined_path), candidate_id):
            already_complete += 1
            continue
        protein_path = protein_dir / f"{candidate_id}.json"
        genome_path = genome_dir / f"{candidate_id}.json"
        protein = read_json(protein_path)
        genome = read_json(genome_path)
        if not valid_mode(protein, candidate_id, "proteins") or not valid_mode(genome, candidate_id, "genome"):
            incomplete.append(candidate_id)
            continue
        assert protein is not None and genome is not None
        merged = dict(protein)
        merged["requested_modes"] = ["proteins", "genome"]
        merged["busco"] = {
            "proteins": protein["busco"]["proteins"],
            "genome": genome["busco"]["genome"],
        }
        merged["elapsed_seconds"] = round(
            float(protein.get("elapsed_seconds", 0.0)) + float(genome.get("elapsed_seconds", 0.0)), 3
        )
        merged["mode_shard_sha256"] = {
            "proteins": sha256(protein_path),
            "genome": sha256(genome_path),
        }
        staging = dict(protein.get("staging", {}))
        if isinstance(genome.get("staging"), dict):
            staging["genome_bytes"] = genome["staging"].get("genome_bytes", staging.get("genome_bytes"))
        merged["staging"] = staging
        merged["status"] = "PASS"
        atomic_json(combined_path, merged)
        merged_count += 1
    print(
        json.dumps(
            {
                "task_count": len(tasks),
                "already_complete": already_complete,
                "merged": merged_count,
                "incomplete": len(incomplete),
                "incomplete_candidate_ids": incomplete,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

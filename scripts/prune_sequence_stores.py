#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


CANDIDATE_ID = re.compile(r"^[0-9a-f]{16}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        temporary.unlink(missing_ok=True)


def directory_bytes(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError(f"sequence-store subtree contains a symlink: {Path(root) / name}")
        for name in files:
            item = Path(root) / name
            if item.is_symlink() or not item.is_file():
                raise ValueError(f"sequence-store subtree contains an unsafe file: {item}")
            total += item.stat().st_size
    return total


def validate_release(project_root: Path, manifest_path: Path) -> tuple[dict[str, object], Path]:
    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    receipt_path = manifest_path.parent / "training_dataset.release.json"
    ready_path = manifest_path.parent / "TRAINING_DATASET_READY"
    receipt_bytes = receipt_path.read_bytes()
    if ready_path.read_text(encoding="ascii").strip() != hashlib.sha256(receipt_bytes).hexdigest():
        raise ValueError("training release READY/hash mismatch")
    receipt = json.loads(receipt_bytes)
    if receipt.get("state") != "READY" or receipt.get("dataset_manifest_sha256") != sha256(manifest_path):
        raise ValueError("training release does not bind the manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "2.0" or manifest.get("state") != "READY":
        raise ValueError("only a READY schema-2 release can authorize store pruning")
    relative = Path(str(manifest.get("store_root", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe store_root in training manifest")
    store_root = (project_root / relative).resolve()
    expected = (project_root / "data/processed/sequence_store").resolve()
    if store_root != expected:
        raise ValueError(f"store pruning is confined to {expected}")
    return manifest, store_root


def build_gc_plan(project_root: Path, manifest_path: Path) -> dict[str, object]:
    manifest, store_root = validate_release(project_root, manifest_path)
    keep_ids = {str(source["candidate_id"]) for source in manifest.get("sources", [])}
    if not keep_ids or any(not CANDIDATE_ID.fullmatch(candidate_id) for candidate_id in keep_ids):
        raise ValueError("training manifest has invalid or empty candidate IDs")
    observed: dict[str, Path] = {}
    for item in store_root.iterdir():
        if item.is_symlink() or not item.is_dir() or not CANDIDATE_ID.fullmatch(item.name):
            raise ValueError(f"unexpected sequence-store entry: {item}")
        observed[item.name] = item
    missing = sorted(keep_ids - set(observed))
    if missing:
        raise ValueError(f"release-referenced stores are missing: {missing}")
    remove_ids = sorted(set(observed) - keep_ids)
    remove = []
    for candidate_id in remove_ids:
        directory = observed[candidate_id]
        ready = directory / "READY"
        manifest_file = directory / "manifest.json"
        if not ready.is_file() or ready.read_text(encoding="ascii").strip() != sha256(manifest_file):
            raise ValueError(f"unreferenced store is not internally READY: {candidate_id}")
        remove.append(
            {
                "candidate_id": candidate_id,
                "store_manifest_sha256": sha256(manifest_file),
                "bytes": directory_bytes(directory),
            }
        )
    return {
        "schema_version": "1.0",
        "state": "PLANNED",
        "training_manifest_sha256": sha256(manifest_path.resolve()),
        "store_root": "data/processed/sequence_store",
        "keep_candidate_ids": sorted(keep_ids),
        "remove": remove,
        "remove_store_count": len(remove),
        "remove_bytes": sum(int(row["bytes"]) for row in remove),
        "raw_scope": "OUT_OF_SCOPE_UNTOUCHED",
    }


def execute_gc(project_root: Path, manifest_path: Path, receipt_path: Path) -> dict[str, object]:
    plan = build_gc_plan(project_root, manifest_path)
    plan_bytes = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    planned_path = receipt_path.with_suffix(".plan.json")
    atomic_write(planned_path, plan_bytes)
    store_root = (project_root.resolve() / str(plan["store_root"])).resolve()
    for row in plan["remove"]:
        candidate_id = str(row["candidate_id"])
        target = (store_root / candidate_id).resolve()
        if target.parent != store_root or not CANDIDATE_ID.fullmatch(target.name):
            raise ValueError(f"unsafe deletion target: {target}")
        shutil.rmtree(target)
        if target.exists():
            raise RuntimeError(f"failed to remove unreferenced store: {target}")
    receipt = {
        **plan,
        "state": "COMPLETED",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "removed_candidate_ids": [str(row["candidate_id"]) for row in plan["remove"]],
    }
    atomic_write(receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune sequence stores not referenced by a READY schema-2 release")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        result = execute_gc(args.project_root, args.manifest, args.receipt)
    else:
        result = build_gc_plan(args.project_root, args.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

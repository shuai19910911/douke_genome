#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an AutoDL release staging directory")
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--deep", action="store_true", help="also rehash every packed sequence payload")
    args = parser.parse_args()
    root = args.release_root.resolve()
    manifest_path = root / "AUTODL_RELEASE_MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    if (root / "READY").read_text(encoding="ascii").strip() != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("release READY digest mismatch")
    payload = json.loads(manifest_bytes)
    for item in payload["project_files"]:
        path = root / item["path"]
        if path.stat().st_size != int(item["size_bytes"]) or sha256(path) != item["sha256"]:
            raise ValueError(f"project file mismatch: {item['path']}")
    dataset = root / payload["dataset_manifest_path"]
    if sha256(dataset) != payload["dataset_manifest_sha256"]:
        raise ValueError("dataset manifest hash mismatch")
    environment_archive = payload.get("environment_archive")
    if environment_archive is not None:
        archive = root / environment_archive["path"]
        if archive.stat().st_size != int(environment_archive["size_bytes"]):
            raise ValueError("environment archive size mismatch")
        if sha256(archive) != environment_archive["sha256"]:
            raise ValueError("environment archive hash mismatch")
    store_root = dataset.parent.parent / json.loads(dataset.read_text(encoding="utf-8"))["store_root"]
    for item in payload["stores"]:
        directory = store_root / item["candidate_id"]
        manifest = directory / "manifest.json"
        if sha256(manifest) != item["manifest_sha256"]:
            raise ValueError(f"store manifest mismatch: {item['candidate_id']}")
        if (directory / "READY").read_text(encoding="ascii").strip() != item["manifest_sha256"]:
            raise ValueError(f"store READY mismatch: {item['candidate_id']}")
        packed = directory / "sequence.2bit"
        if packed.stat().st_size != int(item["packed_size_bytes"]):
            raise ValueError(f"store size mismatch: {item['candidate_id']}")
        if args.deep and sha256(packed) != item["packed_sha256"]:
            raise ValueError(f"store payload mismatch: {item['candidate_id']}")
    print(json.dumps({"deep": args.deep, "source_count": payload["source_count"], "state": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

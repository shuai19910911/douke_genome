#!/usr/bin/env python3
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_exclusive(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=8 * 1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    shutil.copystat(source, destination, follow_symlinks=False)


def _reflink(source: Path, destination: Path) -> bool:
    ficlone = 0x40049409
    source_fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        try:
            fcntl.ioctl(destination_fd, ficlone, source_fd)
            os.fsync(destination_fd)
        except OSError:
            os.close(destination_fd)
            destination.unlink(missing_ok=True)
            return False
        else:
            os.close(destination_fd)
            shutil.copystat(source, destination, follow_symlinks=False)
            return True
    finally:
        os.close(source_fd)


def stage_file(source: Path, destination: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"release source must be a regular file: {source}")
    try:
        os.link(source, destination, follow_symlinks=False)
        mode = "hardlink"
    except OSError as error:
        if error.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
            raise
        if _reflink(source, destination):
            mode = "reflink"
        else:
            _copy_exclusive(source, destination)
            mode = "copy"
    if source.stat().st_size != destination.stat().st_size:
        raise ValueError(f"staged file size mismatch: {source}")
    return mode


def safe_extract_git_archive(project_root: Path, destination: Path) -> str:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=tar", commit], cwd=project_root)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        for member in handle.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile() and not member.isdir():
                raise ValueError(f"unsafe git archive member: {member.name}")
        handle.extractall(destination, filter="data")
    return commit


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a content-addressed AutoDL staging directory")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--environment-archive", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    dataset_manifest = args.dataset_manifest.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"release target already exists: {output}")
    output.mkdir(parents=True)
    release_project = output / "project"
    release_project.mkdir()
    commit = safe_extract_git_archive(project_root, release_project)
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    store_reference = Path(dataset["store_root"])
    source_store_root = project_root / store_reference
    release_store_root = release_project / store_reference
    release_store_root.mkdir(parents=True, exist_ok=True)
    stores: list[dict[str, object]] = []
    for source in dataset["sources"]:
        candidate_id = source["candidate_id"]
        source_dir = source_store_root / candidate_id
        target_dir = release_store_root / candidate_id
        target_dir.mkdir()
        manifest_bytes = (source_dir / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if (source_dir / "READY").read_text(encoding="ascii").strip() != manifest_sha:
            raise ValueError(f"invalid source store READY: {candidate_id}")
        store_manifest = json.loads(manifest_bytes)
        if source["store_manifest_sha256"] != manifest_sha:
            raise ValueError(f"dataset/store manifest mismatch: {candidate_id}")
        staging_modes = {
            name: stage_file(source_dir / name, target_dir / name)
            for name in ("manifest.json", "READY", "sequence.2bit")
        }
        stores.append(
            {
                "candidate_id": candidate_id,
                "manifest_sha256": manifest_sha,
                "packed_sha256": store_manifest["packed_sha256"],
                "packed_size_bytes": store_manifest["packed_size_bytes"],
                "staging_modes": staging_modes,
            }
        )
    release_dataset = release_project / "data_release" / "training_dataset.json"
    release_dataset.parent.mkdir(parents=True, exist_ok=True)
    release_dataset.write_bytes(dataset_manifest.read_bytes())
    environment_archive: dict[str, object] | None = None
    if args.environment_archive is not None:
        source_archive = args.environment_archive.resolve()
        target_archive = release_project / "environment" / "douke_genomemodel.tar.gz"
        target_archive.parent.mkdir(parents=True, exist_ok=True)
        staging_mode = stage_file(source_archive, target_archive)
        environment_archive = {
            "path": target_archive.relative_to(output).as_posix(),
            "sha256": sha256(target_archive),
            "size_bytes": target_archive.stat().st_size,
            "staging_mode": staging_mode,
        }
    project_files: list[dict[str, object]] = []
    for path in sorted(release_project.rglob("*")):
        if not path.is_file() or store_reference in path.relative_to(release_project).parents:
            continue
        relative = path.relative_to(output).as_posix()
        project_files.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "schema_version": "1.0",
        "state": "READY",
        "git_commit": commit,
        "dataset_manifest_path": "project/data_release/training_dataset.json",
        "dataset_manifest_sha256": sha256(release_dataset),
        "source_count": len(stores),
        "stores": stores,
        "project_files": project_files,
        "environment_archive": environment_archive,
        "store_staging_modes": sorted(
            {mode for store in stores for mode in store["staging_modes"].values()}
        ),
    }
    manifest_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output / "AUTODL_RELEASE_MANIFEST.json").write_bytes(manifest_bytes)
    (output / "READY").write_text(hashlib.sha256(manifest_bytes).hexdigest() + "\n", encoding="ascii")
    print(json.dumps({"commit": commit, "output": str(output), "source_count": len(stores)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from legumegenomefm.reference_integrity import validate_busco_lineage


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


def load_task(path: Path, task_index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if int(row["task_index"]) == task_index]
    if len(rows) != 1:
        raise ValueError(f"task index {task_index} resolves to {len(rows)} rows")
    if rows[0]["task_state"] != "READY":
        raise ValueError(f"task {task_index} is not READY")
    return rows[0]


def stage_member(source: Path, member_name: str, destination: Path) -> None:
    if member_name not in {"", "."}:
        with zipfile.ZipFile(source) as archive:
            info = archive.getinfo(member_name)
            with archive.open(info) as input_handle, destination.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
        return
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rb") as input_handle, destination.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)


def count_fasta_records(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.startswith(b">"):
                count += 1
    return count


def run_command(
    command: list[str],
    env: dict[str, str],
    timeout: int | None = None,
    cwd: Path | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )
    elapsed = time.monotonic() - started
    record = {
        "command": [Path(command[0]).name, *command[1:]],
        "exit_code": completed.returncode,
        "elapsed_seconds": round(elapsed, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-8000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, sort_keys=True))
    return record


def find_summary(run_root: Path, mode_name: str) -> dict[str, object]:
    matches = sorted(run_root.rglob("short_summary.specific.*.json"))
    if len(matches) != 1:
        raise ValueError(f"{mode_name} BUSCO produced {len(matches)} specific JSON summaries")
    summary = json.loads(matches[0].read_text(encoding="utf-8"))
    results = summary.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{mode_name} BUSCO summary lacks results mapping")
    return {
        "summary_relative_path": str(matches[0].relative_to(run_root)),
        "parameters": summary.get("parameters", {}),
        "lineage_dataset": summary.get("lineage_dataset", {}),
        "versions": summary.get("versions", {}),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run annotation and genome BUSCO for one audited candidate")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--lineage", required=True, type=Path)
    parser.add_argument("--qc-env", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--modes", default="proteins,genome")
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    task = load_task(args.tasks.resolve(), args.task_index)
    lineage = args.lineage.resolve()
    qc_env = args.qc_env.resolve()
    output = args.output_dir.resolve() / f"{task['candidate_id']}.json"
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not modes or any(mode not in {"proteins", "genome"} for mode in modes):
        raise ValueError(f"invalid modes: {modes}")
    if args.cpus < 1:
        raise ValueError("cpus must be positive")
    if not (lineage / "dataset.cfg").is_file():
        raise FileNotFoundError(lineage / "dataset.cfg")
    for executable in ("python", "gffread", "busco"):
        if not (qc_env / "bin" / executable).is_file():
            raise FileNotFoundError(qc_env / "bin" / executable)

    scratch_parent = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    scratch_parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "schema_version": "1.0",
        "candidate_id": task["candidate_id"],
        "task_index": args.task_index,
        "species": task["species"],
        "material_key": task["material_key"],
        "annotation_id": task["annotation_id"],
        "base_count": int(task["base_count"]),
        "requested_modes": modes,
        "status": "ERROR",
    }
    started = time.monotonic()
    try:
        result["lineage_receipt_sha256"] = validate_busco_lineage(project_root, lineage)
        with tempfile.TemporaryDirectory(prefix=f"soybusco-{task['candidate_id']}-", dir=scratch_parent) as scratch_text:
            scratch = Path(scratch_text)
            genome = scratch / "genome.fa"
            annotation = scratch / "annotation.gff3"
            proteins = scratch / "annotation.proteins.fa"
            stage_member(
                project_root / "data/raw" / task["genome_relative_path"],
                task["genome_member_name"],
                genome,
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{qc_env / 'bin'}:{environment.get('PATH', '')}"
            environment["PYTHONNOUSERSITE"] = "1"
            staging: dict[str, object] = {"genome_bytes": genome.stat().st_size}
            if "proteins" in modes:
                stage_member(
                    project_root / "data/raw" / task["annotation_relative_path"],
                    task["annotation_member_name"],
                    annotation,
                )
                gffread_record = run_command(
                    [str(qc_env / "bin/gffread"), str(annotation), "-g", str(genome), "-y", str(proteins)],
                    environment,
                    cwd=scratch,
                )
                protein_count = count_fasta_records(proteins)
                if protein_count < 1:
                    raise ValueError("gffread produced no protein sequences")
                staging.update(
                    {
                        "annotation_bytes": annotation.stat().st_size,
                        "protein_bytes": proteins.stat().st_size,
                        "protein_record_count": protein_count,
                        "gffread": gffread_record,
                    }
                )
            result["staging"] = staging
            busco_results: dict[str, object] = {}
            for mode in modes:
                input_path = proteins if mode == "proteins" else genome
                run_root = scratch / f"busco_{mode}"
                command = [
                    str(qc_env / "bin/busco"),
                    "-i",
                    str(input_path),
                    "-l",
                    str(lineage),
                    "-o",
                    mode,
                    "--out_path",
                    str(run_root),
                    "-m",
                    mode,
                    "-c",
                    str(args.cpus),
                    "--offline",
                    "--force",
                ]
                command_record = run_command(command, environment, cwd=scratch)
                busco_results[mode] = {
                    "command": command_record,
                    "summary": find_summary(run_root, mode),
                }
            result["busco"] = busco_results
            result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    atomic_json(output, result)
    print(json.dumps({"candidate_id": task["candidate_id"], "status": result["status"], "output": str(output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path


def read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("task manifest is empty")
    return rows


def read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def busco_valid(path: Path, candidate_id: str) -> bool:
    value = read_json(path)
    return bool(
        value
        and value.get("candidate_id") == candidate_id
        and value.get("status") == "PASS"
        and set(value.get("requested_modes", [])) == {"proteins", "genome"}
    )


def contamination_valid(path: Path, candidate_id: str) -> bool:
    value = read_json(path)
    if not value or value.get("candidate_id") != candidate_id or value.get("status") != "PASS":
        return False
    tiara = value.get("tiara")
    univec = value.get("univec")
    if not isinstance(tiara, dict) or not isinstance(tiara.get("record_class_base_counts"), dict):
        return False
    if not isinstance(univec, dict) or not isinstance(univec.get("records"), list):
        return False
    return all(
        isinstance(record, dict) and isinstance(record.get("intervals_1based_inclusive"), list)
        for record in univec["records"]
    )


def array_argument(rows: list[dict[str, str]], concurrency: int) -> str:
    return ",".join(row["task_index"] for row in rows) + f"%{concurrency}"


def submit(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {completed.stdout!r}")
    return job_id


def wait_jobs(job_ids: list[str], poll_seconds: int) -> None:
    while job_ids:
        active = []
        for job_id in job_ids:
            completed = subprocess.run(
                ["/usr/bin/squeue", "-j", job_id, "-h"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.stdout.strip():
                active.append(job_id)
        job_ids = active
        if job_ids:
            time.sleep(poll_seconds)


def choose_busco_batch(missing: list[dict[str, str]]) -> tuple[list[dict[str, str]], int, int, str]:
    tiers: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in missing:
        bases = int(row["base_count"])
        tier = "small" if bases <= 2_000_000_000 else ("medium" if bases <= 5_000_000_000 else "large")
        tiers[tier].append(row)
    if tiers["small"]:
        return tiers["small"][:16], 64, 8, "small"
    if tiers["medium"]:
        return tiers["medium"][:6], 128, 3, "medium"
    return tiers["large"][:2], 170, 1, "large"


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair BUSCO and contamination shards in bounded SLURM batches")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    args = parser.parse_args()
    root = args.project_root.resolve()
    tasks_path = args.tasks.resolve()
    tasks = read_tasks(tasks_path)
    if args.poll_seconds < 1 or args.maximum_attempts < 1:
        raise ValueError("poll seconds and maximum attempts must be positive")

    busco_dir = root / "workspace/data_refinement_busco_shards"
    contamination_dir = root / "workspace/data_refinement_contamination_shards"
    busco_dir.mkdir(parents=True, exist_ok=True)
    contamination_dir.mkdir(parents=True, exist_ok=True)
    logs = root / "logs/slurm"
    logs.mkdir(parents=True, exist_ok=True)
    attempts: dict[tuple[str, str], int] = defaultdict(int)
    iteration = 0
    while True:
        missing_busco = [
            row
            for row in tasks
            if not busco_valid(busco_dir / f"{row['candidate_id']}.json", row["candidate_id"])
        ]
        missing_contamination = [
            row
            for row in tasks
            if not contamination_valid(contamination_dir / f"{row['candidate_id']}.json", row["candidate_id"])
        ]
        print(
            json.dumps(
                {
                    "event": "qc_progress",
                    "busco_remaining": len(missing_busco),
                    "contamination_remaining": len(missing_contamination),
                    "iteration": iteration,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not missing_busco and not missing_contamination:
            print(json.dumps({"event": "qc_complete", "task_count": len(tasks)}, sort_keys=True), flush=True)
            return 0

        iteration += 1
        submitted: list[str] = []
        if missing_contamination:
            batch = missing_contamination[:16]
            for row in batch:
                key = ("contamination", row["candidate_id"])
                attempts[key] += 1
                if attempts[key] > args.maximum_attempts:
                    raise RuntimeError(f"contamination retries exhausted for {row['candidate_id']}")
                (contamination_dir / f"{row['candidate_id']}.json").unlink(missing_ok=True)
            command = [
                "/usr/bin/sbatch",
                "--parsable",
                "--partition=q02,q03,q04,q05",
                f"--array={array_argument(batch, 8)}",
                f"--output={logs / 'contamination-repair-%A_%a.out'}",
                f"--error={logs / 'contamination-repair-%A_%a.err'}",
                "--export="
                + ",".join(
                    (
                        "ALL",
                        f"PROJECT_ROOT={root}",
                        f"TASKS={tasks_path}",
                        f"CONFIG={root / 'configs/data_refinement.yaml'}",
                        f"TIARA_IMAGE={root / 'data/reference/containers/tiara-1.0.3.sif'}",
                        f"UNIVEC_DB={root / 'data/reference/univec/UniVec_Core'}",
                        "QC_ENV=/home/user/zhangzhishuai/.local/share/mamba/envs/soygenome_qc",
                        "PYTHON_BIN=/home/user/zhangzhishuai/.local/share/mamba/envs/douke_genomemodel/bin/python",
                        f"OUTPUT_DIR={contamination_dir}",
                    )
                ),
                str(root / "scripts/slurm/run_contamination_task.sbatch"),
            ]
            job_id = submit(command)
            submitted.append(job_id)
            print(json.dumps({"event": "submitted", "kind": "contamination", "job_id": job_id, "count": len(batch)}), flush=True)

        if missing_busco:
            batch, memory_gib, concurrency, tier = choose_busco_batch(missing_busco)
            for row in batch:
                key = ("busco", row["candidate_id"])
                attempts[key] += 1
                if attempts[key] > args.maximum_attempts:
                    raise RuntimeError(f"BUSCO retries exhausted for {row['candidate_id']}")
                (busco_dir / f"{row['candidate_id']}.json").unlink(missing_ok=True)
            command = [
                "/usr/bin/sbatch",
                "--parsable",
                "--partition=q02,q03,q04,q05",
                "--cpus-per-task=8",
                f"--mem={memory_gib}G",
                f"--array={array_argument(batch, concurrency)}",
                f"--output={logs / f'busco-{tier}-repair-%A_%a.out'}",
                f"--error={logs / f'busco-{tier}-repair-%A_%a.err'}",
                "--export="
                + ",".join(
                    (
                        "ALL",
                        f"PROJECT_ROOT={root}",
                        f"TASKS={tasks_path}",
                        f"LINEAGE={root / 'data/reference/busco/lineages/eudicots_odb10'}",
                        "QC_ENV=/home/user/zhangzhishuai/.local/share/mamba/envs/soygenome_qc",
                        f"OUTPUT_DIR={busco_dir}",
                        "BUSCO_MODES=both",
                    )
                ),
                str(root / "scripts/slurm/run_busco_task.sbatch"),
            ]
            job_id = submit(command)
            submitted.append(job_id)
            print(
                json.dumps(
                    {"event": "submitted", "kind": "busco", "tier": tier, "job_id": job_id, "count": len(batch)},
                    sort_keys=True,
                ),
                flush=True,
            )
        wait_jobs(submitted, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from merge_busco_mode_shards import read_json, valid_combined, valid_mode
from submit_qc_repair_batches import array_argument, contamination_valid, submit, wait_jobs


def read_tasks(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def choose_genome_batch(missing: list[dict[str, str]]) -> tuple[list[dict[str, str]], int, int, str]:
    tiers: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in missing:
        bases = int(row["base_count"])
        if bases <= 750_000_000:
            tier = "tiny"
        elif bases <= 2_000_000_000:
            tier = "small"
        elif bases <= 5_000_000_000:
            tier = "medium"
        else:
            tier = "large"
        tiers[tier].append(row)
    if tiers["tiny"]:
        return tiers["tiny"][:16], 48, 8, "tiny"
    if tiers["small"]:
        return tiers["small"][:12], 64, 6, "small"
    if tiers["medium"]:
        return tiers["medium"][:4], 128, 2, "medium"
    return tiers["large"][:1], 170, 1, "large"


def run_merger(
    root: Path,
    tasks: Path,
    combined: Path,
    proteins: Path,
    genomes: Path,
    lineage_ready: Path,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(root / "scripts/merge_busco_mode_shards.py"),
            "--tasks",
            str(tasks),
            "--combined-dir",
            str(combined),
            "--protein-dir",
            str(proteins),
            "--genome-dir",
            str(genomes),
            "--lineage-ready",
            str(lineage_ready),
        ],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bounded split-mode BUSCO and contamination repair batches")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    args = parser.parse_args()
    root = args.project_root.resolve()
    tasks_path = args.tasks.resolve()
    tasks = read_tasks(tasks_path)
    if not tasks:
        raise ValueError("task manifest is empty")
    logs = root / "logs/slurm"
    combined_dir = root / "workspace/data_refinement_busco_shards"
    protein_dir = root / "workspace/data_refinement_busco_protein_shards"
    genome_dir = root / "workspace/data_refinement_busco_genome_shards"
    contamination_dir = root / "workspace/data_refinement_contamination_shards"
    lineage_ready = root / "data_manifests/busco_lineage_eudicots_odb10.READY"
    lineage_receipt_sha256 = lineage_ready.read_text(encoding="ascii").strip()
    if len(lineage_receipt_sha256) != 64:
        raise ValueError("invalid BUSCO lineage READY digest")
    for directory in (logs, combined_dir, protein_dir, genome_dir, contamination_dir):
        directory.mkdir(parents=True, exist_ok=True)
    attempts: dict[tuple[str, str], int] = defaultdict(int)
    iteration = 0
    while True:
        run_merger(root, tasks_path, combined_dir, protein_dir, genome_dir, lineage_ready)
        combined_missing = [
            row
            for row in tasks
            if not valid_combined(
                read_json(combined_dir / f"{row['candidate_id']}.json"),
                row["candidate_id"],
                lineage_receipt_sha256,
            )
        ]
        protein_missing = [
            row
            for row in combined_missing
            if not valid_mode(
                read_json(protein_dir / f"{row['candidate_id']}.json"),
                row["candidate_id"],
                "proteins",
                lineage_receipt_sha256,
            )
        ]
        genome_missing = [
            row
            for row in combined_missing
            if not valid_mode(
                read_json(genome_dir / f"{row['candidate_id']}.json"),
                row["candidate_id"],
                "genome",
                lineage_receipt_sha256,
            )
        ]
        contamination_missing = [
            row
            for row in tasks
            if not contamination_valid(contamination_dir / f"{row['candidate_id']}.json", row["candidate_id"])
        ]
        print(
            json.dumps(
                {
                    "event": "split_qc_progress",
                    "iteration": iteration,
                    "combined_busco_remaining": len(combined_missing),
                    "protein_remaining": len(protein_missing),
                    "genome_remaining": len(genome_missing),
                    "contamination_remaining": len(contamination_missing),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not combined_missing and not contamination_missing:
            print(json.dumps({"event": "split_qc_complete", "task_count": len(tasks)}, sort_keys=True), flush=True)
            return 0
        iteration += 1
        submitted: list[str] = []

        if contamination_missing:
            batch = contamination_missing[:16]
            for row in batch:
                key = ("contamination", row["candidate_id"])
                attempts[key] += 1
                if attempts[key] > args.maximum_attempts:
                    raise RuntimeError(f"contamination retries exhausted for {row['candidate_id']}")
                (contamination_dir / f"{row['candidate_id']}.json").unlink(missing_ok=True)
            job_id = submit(
                [
                    "/usr/bin/sbatch",
                    "--parsable",
                    "--partition=q02,q03,q04,q05",
                    "--cpus-per-task=4",
                    f"--array={array_argument(batch, 8)}",
                    f"--output={logs / 'contamination-split-repair-%A_%a.out'}",
                    f"--error={logs / 'contamination-split-repair-%A_%a.err'}",
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
            )
            submitted.append(job_id)
            print(json.dumps({"event": "submitted", "kind": "contamination", "job_id": job_id, "count": len(batch)}), flush=True)

        if protein_missing:
            batch = protein_missing[:24]
            for row in batch:
                key = ("proteins", row["candidate_id"])
                attempts[key] += 1
                if attempts[key] > args.maximum_attempts:
                    raise RuntimeError(f"protein BUSCO retries exhausted for {row['candidate_id']}")
                (protein_dir / f"{row['candidate_id']}.json").unlink(missing_ok=True)
            job_id = submit(
                [
                    "/usr/bin/sbatch",
                    "--parsable",
                    "--partition=q02,q03,q04,q05",
                    "--cpus-per-task=4",
                    "--mem=8G",
                    f"--array={array_argument(batch, 12)}",
                    f"--output={logs / 'busco-protein-split-%A_%a.out'}",
                    f"--error={logs / 'busco-protein-split-%A_%a.err'}",
                    "--export="
                    + ",".join(
                        (
                            "ALL",
                            f"PROJECT_ROOT={root}",
                            f"TASKS={tasks_path}",
                            f"LINEAGE={root / 'data/reference/busco/lineages/eudicots_odb10'}",
                            "QC_ENV=/home/user/zhangzhishuai/.local/share/mamba/envs/soygenome_qc",
                            f"OUTPUT_DIR={protein_dir}",
                            "BUSCO_MODES=proteins",
                        )
                    ),
                    str(root / "scripts/slurm/run_busco_task.sbatch"),
                ]
            )
            submitted.append(job_id)
            print(json.dumps({"event": "submitted", "kind": "busco_proteins", "job_id": job_id, "count": len(batch)}), flush=True)

        if genome_missing:
            batch, memory_gib, concurrency, tier = choose_genome_batch(genome_missing)
            for row in batch:
                key = ("genome", row["candidate_id"])
                attempts[key] += 1
                if attempts[key] > args.maximum_attempts:
                    raise RuntimeError(f"genome BUSCO retries exhausted for {row['candidate_id']}")
                (genome_dir / f"{row['candidate_id']}.json").unlink(missing_ok=True)
            job_id = submit(
                [
                    "/usr/bin/sbatch",
                    "--parsable",
                    "--partition=q02,q03,q04,q05",
                    "--cpus-per-task=4",
                    f"--mem={memory_gib}G",
                    f"--array={array_argument(batch, concurrency)}",
                    f"--output={logs / f'busco-genome-{tier}-split-%A_%a.out'}",
                    f"--error={logs / f'busco-genome-{tier}-split-%A_%a.err'}",
                    "--export="
                    + ",".join(
                        (
                            "ALL",
                            f"PROJECT_ROOT={root}",
                            f"TASKS={tasks_path}",
                            f"LINEAGE={root / 'data/reference/busco/lineages/eudicots_odb10'}",
                            "QC_ENV=/home/user/zhangzhishuai/.local/share/mamba/envs/soygenome_qc",
                            f"OUTPUT_DIR={genome_dir}",
                            "BUSCO_MODES=genome",
                        )
                    ),
                    str(root / "scripts/slurm/run_busco_task.sbatch"),
                ]
            )
            submitted.append(job_id)
            print(
                json.dumps(
                    {"event": "submitted", "kind": "busco_genome", "tier": tier, "job_id": job_id, "count": len(batch)},
                    sort_keys=True,
                ),
                flush=True,
            )
        wait_jobs(submitted, args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

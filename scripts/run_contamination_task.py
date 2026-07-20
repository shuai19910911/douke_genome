#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from legumegenomefm.data_refinement import (
    eligible_tiara_chunks,
    merge_inclusive_intervals,
    merged_interval_bases,
    univec_hit_is_high_confidence,
)


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
    return rows[0]


def stage_member(source: Path, member_name: str, destination: Path) -> None:
    if member_name not in {"", "."}:
        with zipfile.ZipFile(source) as archive:
            with archive.open(archive.getinfo(member_name)) as input_handle, destination.open("wb") as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
        return
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rb") as input_handle, destination.open("wb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)


def run_command(command: list[str], environment: dict[str, str]) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(command, env=environment, check=False, capture_output=True, text=True)
    record = {
        "command": [Path(command[0]).name, *command[1:]],
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-8000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(json.dumps(record, sort_keys=True))
    return record


def write_tiara_chunks(
    genome: Path,
    output: Path,
    chunk_length: int,
    minimum_length: int,
    minimum_acgt_fraction: float,
    expected_lengths: dict[str, int],
) -> dict[str, dict[str, object]]:
    mapping: dict[str, dict[str, object]] = {}
    seen: set[str] = set()

    def write_record(handle, sequence_id: str, sequence: str) -> None:
        if sequence_id in seen:
            raise ValueError(f"duplicate FASTA sequence ID: {sequence_id}")
        seen.add(sequence_id)
        if sequence_id not in expected_lengths or len(sequence) != expected_lengths[sequence_id]:
            raise ValueError(f"FASTA length mismatch for {sequence_id}")
        for start, end, chunk in eligible_tiara_chunks(
            sequence,
            chunk_length,
            minimum_length,
            minimum_acgt_fraction,
        ):
            chunk_id = f"chunk_{len(mapping) + 1:09d}"
            handle.write(f">{chunk_id}\n{chunk}\n")
            mapping[chunk_id] = {
                "sequence_id": sequence_id,
                "start": start,
                "end": end,
                "length": len(chunk),
            }

    with genome.open(encoding="ascii") as input_handle, output.open("w", encoding="ascii", newline="\n") as output_handle:
        sequence_id: str | None = None
        pieces: list[str] = []
        for line in input_handle:
            if line.startswith(">"):
                if sequence_id is not None:
                    write_record(output_handle, sequence_id, "".join(pieces))
                sequence_id = line[1:].strip().split()[0]
                pieces = []
            else:
                pieces.append(line.strip())
        if sequence_id is not None:
            write_record(output_handle, sequence_id, "".join(pieces))
    if seen != set(expected_lengths):
        raise ValueError("FASTA sequence IDs do not equal sequence-store manifest IDs")
    if not mapping:
        raise ValueError("no sequence chunks passed Tiara input criteria")
    return mapping


def parse_tiara(path: Path, chunks: dict[str, dict[str, object]]) -> dict[str, object]:
    counts: Counter[str] = Counter()
    bases: Counter[str] = Counter()
    record_bases: dict[str, Counter[str]] = defaultdict(Counter)
    non_eukaryotic: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            chunk_id = row["sequence_id"]
            if chunk_id not in chunks:
                raise ValueError(f"Tiara chunk ID is absent from input map: {chunk_id}")
            chunk = chunks[chunk_id]
            first = row["class_fst_stage"]
            second = row["class_snd_stage"]
            final_class = second if second not in {"", "n/a"} else first
            counts[final_class] += 1
            bases[final_class] += int(chunk["length"])
            record_bases[str(chunk["sequence_id"])][final_class] += int(chunk["length"])
            if final_class != "eukarya":
                non_eukaryotic.append(
                    {
                        "chunk_id": chunk_id,
                        "sequence_id": chunk["sequence_id"],
                        "start": chunk["start"],
                        "end": chunk["end"],
                        "length": chunk["length"],
                        "class": final_class,
                        "first_stage_class": first,
                        "second_stage_class": second,
                        "euk_probability": row.get("euk", "."),
                        "bacteria_probability": row.get("bac", "."),
                        "archaea_probability": row.get("arc", "."),
                        "organelle_probability": row.get("org", "."),
                    }
                )
    non_eukaryotic.sort(key=lambda row: (str(row["sequence_id"]), int(row["start"])))
    if sum(counts.values()) != len(chunks):
        raise ValueError("Tiara output chunk count does not equal input chunk count")
    return {
        "class_record_counts": dict(sorted(counts.items())),
        "class_base_counts": dict(sorted(bases.items())),
        "record_class_base_counts": {
            sequence_id: dict(sorted(class_counts.items()))
            for sequence_id, class_counts in sorted(record_bases.items())
        },
        "non_eukaryotic_records": non_eukaryotic,
    }


def parse_univec(path: Path, rules: list[dict[str, object]], lengths: dict[str, int]) -> dict[str, object]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    hit_counts: Counter[str] = Counter()
    max_lengths: Counter[str] = Counter()
    all_hit_count = 0
    high_confidence_hit_count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 13:
                raise ValueError(f"unexpected UniVec BLAST row at line {line_number}")
            qseqid = fields[0]
            if qseqid not in lengths:
                raise ValueError(f"UniVec query sequence ID is absent from genome manifest: {qseqid}")
            percent_identity = float(fields[3])
            alignment_length = int(fields[4])
            qstart, qend = int(fields[7]), int(fields[8])
            all_hit_count += 1
            if not univec_hit_is_high_confidence(percent_identity, alignment_length, rules):
                continue
            high_confidence_hit_count += 1
            intervals[qseqid].append((qstart, qend))
            hit_counts[qseqid] += 1
            max_lengths[qseqid] = max(max_lengths[qseqid], alignment_length)
    records = []
    for sequence_id, sequence_intervals in intervals.items():
        merged = merge_inclusive_intervals(sequence_intervals)
        records.append(
            {
                "sequence_id": sequence_id,
                "sequence_length": lengths[sequence_id],
                "high_confidence_hit_count": hit_counts[sequence_id],
                "maximum_alignment_length": max_lengths[sequence_id],
                "merged_hit_bases": merged_interval_bases(merged),
                "intervals_1based_inclusive": [list(interval) for interval in merged],
            }
        )
    records.sort(key=lambda row: (-int(row["merged_hit_bases"]), str(row["sequence_id"])))
    return {
        "all_hit_count": all_hit_count,
        "high_confidence_hit_count": high_confidence_hit_count,
        "high_confidence_record_count": len(records),
        "high_confidence_merged_bases": sum(int(row["merged_hit_bases"]) for row in records),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Tiara and UniVec contamination audits for one candidate")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tiara-image", required=True, type=Path)
    parser.add_argument("--univec-db", required=True, type=Path)
    parser.add_argument("--qc-env", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cpus", type=int, default=8)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    task = load_task(args.tasks.resolve(), args.task_index)
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))["contamination"]
    tiara_image = args.tiara_image.resolve()
    univec_db = args.univec_db.resolve()
    qc_env = args.qc_env.resolve()
    output = args.output_dir.resolve() / f"{task['candidate_id']}.json"
    if not tiara_image.is_file():
        raise FileNotFoundError(tiara_image)
    if not Path(f"{univec_db}.ndb").is_file():
        raise FileNotFoundError(f"{univec_db}.ndb")
    manifest = json.loads(
        (project_root / "data/processed/sequence_store" / task["candidate_id"] / "manifest.json").read_text()
    )
    lengths = {str(row["name"]): int(row["length"]) for row in manifest["contigs"]}
    if sum(lengths.values()) != int(task["base_count"]):
        raise ValueError("sequence-store contig lengths do not equal task base count")

    result: dict[str, object] = {
        "schema_version": "1.0",
        "candidate_id": task["candidate_id"],
        "task_index": args.task_index,
        "species": task["species"],
        "material_key": task["material_key"],
        "base_count": int(task["base_count"]),
        "status": "ERROR",
    }
    started = time.monotonic()
    scratch_parent = Path(os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp")
    try:
        with tempfile.TemporaryDirectory(prefix=f"soycontam-{task['candidate_id']}-", dir=scratch_parent) as scratch_text:
            scratch = Path(scratch_text)
            genome = scratch / "genome.fa"
            tiara_input = scratch / "tiara_chunks.fa"
            tiara_output = scratch / "tiara.tsv"
            univec_output = scratch / "univec.tsv"
            stage_member(
                project_root / "data/raw" / task["genome_relative_path"],
                task["genome_member_name"],
                genome,
            )
            chunks = write_tiara_chunks(
                genome,
                tiara_input,
                int(config["tiara_chunk_length"]),
                int(config["tiara_minimum_contig_length"]),
                float(config["tiara_minimum_acgt_fraction"]),
                lengths,
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{qc_env / 'bin'}:{environment.get('PATH', '')}"
            environment["PYTHONNOUSERSITE"] = "1"
            tiara_command = [
                "/usr/bin/singularity",
                "exec",
                "--cleanenv",
                "--bind",
                f"{scratch}:{scratch}",
                str(tiara_image),
                "tiara",
                "-i",
                str(tiara_input),
                "-o",
                str(tiara_output),
                "-m",
                str(int(config["tiara_minimum_contig_length"])),
                "-p",
                str(float(config["tiara_probability_cutoff"])),
                str(float(config["tiara_probability_cutoff"])),
                "-t",
                str(args.cpus),
                "--probabilities",
            ]
            tiara_command_record = run_command(tiara_command, environment)
            outfmt = "6 qseqid qlen sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore"
            univec_command = [
                str(qc_env / "bin/blastn"),
                "-task",
                "blastn",
                "-query",
                str(genome),
                "-db",
                str(univec_db),
                "-reward",
                "1",
                "-penalty",
                "-5",
                "-gapopen",
                "3",
                "-gapextend",
                "3",
                "-dust",
                "yes",
                "-soft_masking",
                "true",
                "-evalue",
                "700",
                "-searchsp",
                "1750000000000",
                "-num_threads",
                str(args.cpus),
                "-outfmt",
                outfmt,
                "-out",
                str(univec_output),
            ]
            univec_command_record = run_command(univec_command, environment)
            result["staging"] = {
                "genome_bytes": genome.stat().st_size,
                "contig_count": len(lengths),
                "tiara_chunk_count": len(chunks),
                "tiara_chunk_bases": sum(int(chunk["length"]) for chunk in chunks.values()),
            }
            result["tiara"] = {
                "command": tiara_command_record,
                **parse_tiara(tiara_output, chunks),
            }
            result["univec"] = {
                "command": univec_command_record,
                **parse_univec(univec_output, list(config["univec_high_confidence"]), lengths),
            }
            result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    atomic_json(output, result)
    print(json.dumps({"candidate_id": task["candidate_id"], "status": result["status"], "output": str(output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

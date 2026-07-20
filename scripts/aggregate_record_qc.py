#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from legumegenomefm.data_refinement import merge_half_open_intervals, record_is_primary_nuclear


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def render_tsv(rows: list[dict[str, object]]) -> bytes:
    if not rows:
        raise ValueError("cannot render empty TSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def parse_assembly_report(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    header: list[str] | None = None
    aliases: dict[str, dict[str, str]] = {}
    metadata: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith("# Sequence-Name\t"):
                header = line[2:].split("\t")
                continue
            if line.startswith("#"):
                text = line[1:].strip()
                if ":" in text and not text.startswith("#"):
                    key, value = text.split(":", 1)
                    metadata[key.strip()] = value.strip()
                continue
            if not line:
                continue
            if header is None:
                raise ValueError(f"assembly report data before header: {path}")
            fields = line.split("\t")
            if len(fields) != len(header):
                raise ValueError(f"assembly report column mismatch: {path}")
            row = dict(zip(header, fields))
            for field in ("Sequence-Name", "GenBank-Accn", "RefSeq-Accn", "UCSC-style-name"):
                alias = row.get(field, "")
                if alias.lower() in {"", "na"}:
                    continue
                previous = aliases.get(alias)
                if previous is not None and previous != row:
                    raise ValueError(f"assembly report alias collision for {alias}: {path}")
                aliases[alias] = row
    if header is None or not aliases:
        raise ValueError(f"assembly report has no sequence table: {path}")
    return aliases, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build primary nuclear record and contamination catalogs")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--contamination-shards", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    tasks_path = args.tasks.resolve()
    candidates_path = args.candidates.resolve()
    shards_dir = args.contamination_shards.resolve()
    config_path = args.config.resolve()
    prefix = args.output_prefix.resolve()
    tasks = read_tsv(tasks_path)
    candidate_map = {row["candidate_id"]: row for row in read_tsv(candidates_path)}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assembly_config = config["assembly"]
    contamination_config = config["contamination"]
    proxy_minimum_length = int(assembly_config["structural_proxy"]["long_sequence_threshold"])
    minimum_primary_fraction = float(assembly_config["minimum_primary_nuclear_fraction"])
    minimum_tiara_fraction = float(contamination_config["minimum_primary_tiara_evaluated_fraction"])
    maximum_prok_fraction = float(contamination_config["maximum_primary_prokaryotic_fraction"])
    maximum_vector_fraction = float(contamination_config["maximum_primary_vector_fraction"])
    mask_classes = {str(value) for value in contamination_config["tiara_mask_classes"]}
    prok_classes = {"archaea", "bacteria", "prokarya"}

    expected_ids = {task["candidate_id"] for task in tasks}
    present_ids = {path.stem for path in shards_dir.glob("*.json")}
    missing = sorted(expected_ids - present_ids)
    extras = sorted(present_ids - expected_ids)
    if missing or extras:
        raise ValueError(f"contamination shard set mismatch: missing={missing[:20]} extras={extras[:20]}")

    candidate_rows: list[dict[str, object]] = []
    record_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    for task in tasks:
        candidate_id = task["candidate_id"]
        candidate = candidate_map[candidate_id]
        manifest = json.loads(
            (root / "data/processed/sequence_store" / candidate_id / "manifest.json").read_text(encoding="utf-8")
        )
        contigs = manifest["contigs"]
        if sum(int(record["length"]) for record in contigs) != int(task["base_count"]):
            raise ValueError(f"sequence-store length mismatch for {candidate_id}")
        report_location = candidate["official_assembly_report"]
        report_aliases: dict[str, dict[str, str]] = {}
        report_metadata: dict[str, str] = {}
        if report_location != ".":
            report_aliases, report_metadata = parse_assembly_report(root / report_location)

        primary_names: set[str] = set()
        official_unmapped_long_records: list[str] = []
        candidate_record_rows: list[dict[str, object]] = []
        for index, record in enumerate(contigs):
            name = str(record["name"])
            length = int(record["length"])
            report_row = report_aliases.get(name)
            if report_aliases and report_row is None and length >= proxy_minimum_length:
                official_unmapped_long_records.append(name)
            role = report_row.get("Sequence-Role", ".") if report_row else ("unmapped" if report_aliases else ".")
            location_type = report_row.get("Assigned-Molecule-Location/Type", ".") if report_row else "."
            primary = record_is_primary_nuclear(name, length, role, location_type, proxy_minimum_length)
            if primary:
                primary_names.add(name)
            candidate_record_rows.append(
                {
                    "candidate_id": candidate_id,
                    "contig_index": index,
                    "sequence_name": name,
                    "length": length,
                    "official_sequence_role": role,
                    "official_location_type": location_type,
                    "record_policy": "PRIMARY_NUCLEAR" if primary else "EXCLUDED_NONPRIMARY",
                    "status": "PASS" if primary else "EXCLUDED",
                }
            )
        primary_bases = sum(int(row["length"]) for row in candidate_record_rows if row["record_policy"] == "PRIMARY_NUCLEAR")
        base_count = int(task["base_count"])
        primary_fraction = primary_bases / base_count if base_count else 0.0

        shard = json.loads((shards_dir / f"{candidate_id}.json").read_text(encoding="utf-8"))
        reasons: list[str] = []
        tool_error = "."
        tiara_evaluated_bases = 0
        tiara_prok_bases = 0
        tiara_unknown_bases = 0
        tiara_organelle_bases = 0
        vector_bases = 0
        masks_by_record: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        candidate_mask_rows: list[dict[str, object]] = []
        if shard.get("candidate_id") != candidate_id or shard.get("status") != "PASS":
            reasons.append("contamination_tool_error")
            tool_error = str(shard.get("error", "candidate ID mismatch"))
        else:
            tiara = shard.get("tiara", {})
            record_class_bases = tiara.get("record_class_base_counts")
            if not isinstance(record_class_bases, dict):
                reasons.append("contamination_shard_schema_incomplete")
            else:
                for name in primary_names:
                    class_counts = record_class_bases.get(name, {})
                    if not isinstance(class_counts, dict):
                        continue
                    for class_name, value in class_counts.items():
                        bases = int(value)
                        tiara_evaluated_bases += bases
                        if class_name in prok_classes:
                            tiara_prok_bases += bases
                        if class_name == "unknown":
                            tiara_unknown_bases += bases
                        if class_name in {"organelle", "mitochondria", "plastid"}:
                            tiara_organelle_bases += bases
                for item in tiara.get("non_eukaryotic_records", []):
                    name = str(item["sequence_id"])
                    class_name = str(item["class"])
                    if name in primary_names and class_name in mask_classes:
                        masks_by_record[name].append((int(item["start"]), int(item["end"]), f"tiara:{class_name}"))
            univec = shard.get("univec", {})
            for item in univec.get("records", []):
                name = str(item["sequence_id"])
                if name not in primary_names:
                    continue
                intervals = item.get("intervals_1based_inclusive")
                if not isinstance(intervals, list):
                    reasons.append("contamination_shard_schema_incomplete")
                    continue
                for start, end in intervals:
                    vector_bases += int(end) - int(start) + 1
                    masks_by_record[name].append((int(start) - 1, int(end), "univec"))

        tiara_evaluated_fraction = tiara_evaluated_bases / primary_bases if primary_bases else 0.0
        prok_fraction = tiara_prok_bases / primary_bases if primary_bases else 0.0
        vector_fraction = vector_bases / primary_bases if primary_bases else 0.0
        if official_unmapped_long_records:
            reasons.append("official_report_unmapped_long_record")
        if primary_fraction < minimum_primary_fraction:
            reasons.append("primary_nuclear_fraction_below_minimum")
        if tiara_evaluated_fraction < minimum_tiara_fraction:
            reasons.append("tiara_evaluated_fraction_below_minimum")
        if prok_fraction > maximum_prok_fraction:
            reasons.append("primary_prokaryotic_fraction_above_maximum")
        if vector_fraction > maximum_vector_fraction:
            reasons.append("primary_vector_fraction_above_maximum")
        reasons = sorted(set(reasons))

        for name, sourced_intervals in masks_by_record.items():
            merged = merge_half_open_intervals((start, end) for start, end, _ in sourced_intervals)
            for start, end in merged:
                sources = sorted(
                    {
                        source
                        for source_start, source_end, source in sourced_intervals
                        if source_start < end and start < source_end
                    }
                )
                candidate_mask_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "sequence_name": name,
                        "start_0based": start,
                        "end_0based_exclusive": end,
                        "length": end - start,
                        "sources": ";".join(sources),
                        "status": "MASK",
                    }
                )

        mask_count_by_record: Counter[str] = Counter()
        mask_bases_by_record: Counter[str] = Counter()
        for mask in candidate_mask_rows:
            sequence_name = str(mask["sequence_name"])
            mask_count_by_record[sequence_name] += 1
            mask_bases_by_record[sequence_name] += int(mask["length"])
        mask_rows.extend(candidate_mask_rows)
        for row in candidate_record_rows:
            row["mask_interval_count"] = mask_count_by_record[str(row["sequence_name"])]
            row["masked_bases"] = mask_bases_by_record[str(row["sequence_name"])]
            record_rows.append(row)

        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "species": task["species"],
                "material_key": task["material_key"],
                "base_count": base_count,
                "primary_record_count": len(primary_names),
                "primary_nuclear_bases": primary_bases,
                "primary_nuclear_fraction": round(primary_fraction, 12),
                "tiara_evaluated_primary_bases": tiara_evaluated_bases,
                "tiara_evaluated_primary_fraction": round(tiara_evaluated_fraction, 12),
                "tiara_primary_prokaryotic_bases": tiara_prok_bases,
                "tiara_primary_prokaryotic_fraction": round(prok_fraction, 12),
                "tiara_primary_unknown_bases": tiara_unknown_bases,
                "tiara_primary_organelle_bases": tiara_organelle_bases,
                "univec_primary_bases": vector_bases,
                "univec_primary_fraction": round(vector_fraction, 12),
                "mask_interval_count": len(candidate_mask_rows),
                "masked_primary_bases": sum(int(row["length"]) for row in candidate_mask_rows),
                "reported_qv": ".",
                "qv_status": "UNAVAILABLE_NO_RAW_READS_OR_SOURCE_QV",
                "reported_genome_coverage": report_metadata.get("Genome coverage", "."),
                "reported_assembly_method": report_metadata.get("Assembly method", "."),
                "reported_sequencing_technology": report_metadata.get("Sequencing technology", "."),
                "official_unmapped_long_records": ";".join(sorted(official_unmapped_long_records)) or ".",
                "gate_reasons": ";".join(reasons) or ".",
                "error": tool_error,
                "status": "PASS" if not reasons else ("ERROR" if any("tool" in reason or "schema" in reason for reason in reasons) else "FAIL"),
            }
        )

    candidate_rows.sort(key=lambda row: str(row["candidate_id"]))
    record_rows.sort(key=lambda row: (str(row["candidate_id"]), int(row["contig_index"])))
    mask_rows.sort(key=lambda row: (str(row["candidate_id"]), str(row["sequence_name"]), int(row["start_0based"])))
    candidate_tsv = render_tsv(candidate_rows)
    record_tsv = render_tsv(record_rows)
    if mask_rows:
        mask_tsv = render_tsv(mask_rows)
    else:
        mask_tsv = b"candidate_id\tsequence_name\tstart_0based\tend_0based_exclusive\tlength\tsources\tstatus\n"
    candidate_path = prefix.with_suffix(".candidates.tsv")
    record_path = prefix.with_suffix(".records.tsv")
    mask_path = prefix.with_suffix(".masks.tsv")
    atomic_write(candidate_path, candidate_tsv)
    atomic_write(record_path, record_tsv)
    atomic_write(mask_path, mask_tsv)
    status_counts: dict[str, int] = {}
    for row in candidate_rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    summary = {
        "schema_version": "1.0",
        "state": "COMPLETE" if status_counts.get("ERROR", 0) == 0 else "INCOMPLETE",
        "candidate_count": len(candidate_rows),
        "record_count": len(record_rows),
        "mask_interval_count": len(mask_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "input_sha256": {
            "tasks": sha256(tasks_path),
            "candidates": sha256(candidates_path),
            "config": sha256(config_path),
        },
        "output_sha256": {
            "candidates": hashlib.sha256(candidate_tsv).hexdigest(),
            "records": hashlib.sha256(record_tsv).hexdigest(),
            "masks": hashlib.sha256(mask_tsv).hexdigest(),
        },
    }
    atomic_write(
        prefix.with_suffix(".summary.json"),
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["state"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

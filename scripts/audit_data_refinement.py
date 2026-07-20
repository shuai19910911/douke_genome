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
from typing import Iterable

import yaml

from legumegenomefm.data_refinement import (
    annotation_is_strict,
    classify_assembly,
    nonoverlapping_capacity,
    select_unique_candidates,
)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


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


def parse_assembly_report(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("# ") or ":" not in line:
                continue
            key, value = line[2:].split(":", 1)
            fields[key.strip()] = value.strip()
    return fields


def n50(lengths: list[int]) -> int:
    total = sum(lengths)
    cumulative = 0
    for length in sorted(lengths, reverse=True):
        cumulative += length
        if cumulative * 2 >= total:
            return length
    raise ValueError("cannot calculate N50 for an empty assembly")


def strict_annotation_index(project_root: Path) -> dict[str, list[dict[str, str]]]:
    by_genome: dict[str, list[dict[str, str]]] = defaultdict(list)
    ordinary = read_tsv(project_root / "data_manifests/annotation_qc.tsv")
    archive = read_tsv(project_root / "data_manifests/archive_annotation_qc.tsv")
    for row in ordinary:
        if not annotation_is_strict(row):
            continue
        annotation = {
            "annotation_id": row["candidate_id"],
            "source": row["source"],
            "path": row["relative_path"],
            "member_name": ".",
            "gene_count": row["gene_count"],
        }
        for candidate_id in row["paired_assembly_ids"].split(";"):
            if candidate_id:
                by_genome[candidate_id].append(annotation)
    for original in archive:
        row = {**original, "is_primary_gene_model": "True"}
        if not annotation_is_strict(row):
            continue
        annotation = {
            "annotation_id": row["candidate_id"],
            "source": "soyod",
            "path": row["archive_relative_path"],
            "member_name": row["member_name"],
            "gene_count": row["gene_count"],
        }
        for candidate_id in row["paired_genome_ids"].split(";"):
            if candidate_id and candidate_id != ".":
                by_genome[candidate_id].append(annotation)
    for values in by_genome.values():
        values.sort(key=lambda item: item["annotation_id"])
    return by_genome


def records(project_root: Path, config: dict[str, object]) -> list[dict[str, object]]:
    sketch_path = project_root / str(config["candidate_universe"])
    candidates = read_tsv(sketch_path)
    assembly_rows = {
        row["candidate_id"]: row
        for row in read_tsv(project_root / "data_manifests/assembly_candidates.tsv")
    }
    archive_rows = {
        row["candidate_id"]: row
        for row in read_tsv(project_root / "data_manifests/archive_genome_qc.tsv")
    }
    orientations = {
        row["candidate_id"]: row
        for row in read_tsv(project_root / "data_manifests/orientation_identity.tsv")
    }
    near_clusters = {
        row["candidate_id"]: row
        for row in read_tsv(project_root / "data_manifests/genome_near_duplicate_clusters.tsv")
    }
    annotations = strict_annotation_index(project_root)
    candidate_ids = {row["candidate_id"] for row in candidates}
    if set(orientations) != candidate_ids or set(near_clusters) != candidate_ids:
        raise ValueError("candidate, orientation and near-duplicate ID sets differ")

    assembly_config = dict(config["assembly"])
    proxy = dict(assembly_config["structural_proxy"])
    long_threshold = int(proxy["long_sequence_threshold"])
    contexts = [int(value) for value in config["contexts"]]
    if contexts != sorted(set(contexts)) or not contexts:
        raise ValueError("contexts must be non-empty, sorted and unique")
    maximum_n_fraction = float(assembly_config["maximum_n_fraction"])
    store_root = project_root / str(config["sequence_store_root"])
    raw_root = project_root / "data/raw"
    output: list[dict[str, object]] = []
    store_digest = hashlib.sha256(b"data-refinement-store-set-v1\0")

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        store_dir = store_root / candidate_id
        ready = (store_dir / "READY").read_text(encoding="ascii").strip()
        manifest_path = store_dir / "manifest.json"
        if sha256(manifest_path) != ready:
            raise ValueError(f"store READY mismatch: {candidate_id}")
        store_digest.update(candidate_id.encode("ascii") + b"\0" + ready.encode("ascii") + b"\0")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity", {}).get("candidate_id") != candidate_id:
            raise ValueError(f"store identity mismatch: {candidate_id}")
        lengths = [int(contig["length"]) for contig in manifest["contigs"]]
        base_count = sum(lengths)
        callable_bases = sum(int(interval["length"]) for interval in manifest["callable_intervals"])
        n_fraction = (base_count - callable_bases) / base_count
        large_sequence_fraction = sum(length for length in lengths if length >= long_threshold) / base_count
        long_callable_bases = sum(
            int(interval["length"])
            for interval in manifest["callable_intervals"]
            if int(interval["length"]) >= contexts[-1]
        )
        long_callable_fraction = long_callable_bases / callable_bases if callable_bases else 0.0

        assembly_row = assembly_rows.get(candidate_id)
        if assembly_row is not None:
            source = assembly_row["source"]
            path = assembly_row["relative_path"]
            member_name = "."
        elif candidate_id in archive_rows:
            source = "soyod"
            path = archive_rows[candidate_id]["archive_relative_path"]
            member_name = archive_rows[candidate_id]["member_name"]
        else:
            raise ValueError(f"candidate has no ordinary/archive source row: {candidate_id}")

        official_level = "UNVERIFIED"
        official_report = "."
        assembly_accession = "."
        if source == "legume_family_ncbi":
            genome_path = raw_root / path
            reports = sorted((genome_path.parent.parent / "assembly_report").glob("*assembly_report.txt"))
            if len(reports) != 1:
                raise ValueError(f"expected one NCBI assembly report for {candidate_id}, found {len(reports)}")
            report = parse_assembly_report(reports[0])
            official_level = report.get("Assembly level", "UNVERIFIED")
            assembly_accession = report.get("GenBank assembly accession", report.get("RefSeq assembly accession", "."))
            official_report = reports[0].relative_to(project_root).as_posix()

        label_material = f"{path} {member_name} {candidate['material_key']}".lower()
        t2t_label = bool(assembly_config["recognize_t2t_label_as_provisional"]) and "t2t" in label_material
        assembly_evidence, assembly_tier, evidence_pending = classify_assembly(
            official_level,
            t2t_label,
            n50(lengths),
            large_sequence_fraction,
            int(proxy["minimum_n50"]),
            float(proxy["minimum_fraction_in_long_sequences"]),
        )
        strict_annotations = annotations.get(candidate_id, [])
        hard_reasons: list[str] = []
        if assembly_tier == 0:
            hard_reasons.append("insufficient_chromosome_scale_evidence")
        if n_fraction > maximum_n_fraction:
            hard_reasons.append("n_fraction_above_limit")
        if len(strict_annotations) != 1:
            hard_reasons.append(f"strict_annotation_count_{len(strict_annotations)}")
        capacities = {
            context: nonoverlapping_capacity(manifest["callable_intervals"], context)
            for context in contexts
        }
        if capacities[contexts[0]] == 0:
            hard_reasons.append("no_1k_callable_window")
        annotation = strict_annotations[0] if len(strict_annotations) == 1 else None
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "source_kind": candidate["source_kind"],
            "source": source,
            "relative_path": path,
            "member_name": member_name,
            "species": candidate["species"],
            "genus": candidate["species"].split()[0],
            "material_key": candidate["material_key"],
            "orientation_group_id": orientations[candidate_id]["orientation_group_id"],
            "near_duplicate_group_id": near_clusters[candidate_id]["near_duplicate_group_id"],
            "near_duplicate_group_size": int(near_clusters[candidate_id]["near_duplicate_group_size"]),
            "assembly_evidence": assembly_evidence,
            "assembly_tier": assembly_tier,
            "assembly_evidence_pending": evidence_pending,
            "official_assembly_level": official_level,
            "official_assembly_report": official_report,
            "assembly_accession": assembly_accession,
            "t2t_label": t2t_label,
            "base_count": base_count,
            "callable_bases": callable_bases,
            "n_fraction": n_fraction,
            "contig_count": len(lengths),
            "n50": n50(lengths),
            "large_sequence_fraction": large_sequence_fraction,
            "long_callable_fraction": long_callable_fraction,
            "strict_annotation_count": len(strict_annotations),
            "annotation_id": annotation["annotation_id"] if annotation else ".",
            "annotation_source": annotation["source"] if annotation else ".",
            "annotation_path": annotation["path"] if annotation else ".",
            "annotation_member_name": annotation["member_name"] if annotation else ".",
            "annotation_gene_count": int(annotation["gene_count"]) if annotation else 0,
            "hard_gate_pass": not hard_reasons,
            "hard_gate_reasons": ";".join(hard_reasons) if hard_reasons else ".",
            "store_manifest_sha256": ready,
        }
        for context, capacity in capacities.items():
            row[f"nonoverlap_windows_{context}"] = capacity
        output.append(row)

    status = select_unique_candidates(output)
    for row in output:
        row["selection_status"] = status[str(row["candidate_id"])]
    output.sort(key=lambda row: str(row["candidate_id"]))
    if not output:
        raise ValueError("refinement audit produced no rows")
    output[0]["store_set_sha256"] = store_digest.hexdigest()
    for row in output[1:]:
        row["store_set_sha256"] = store_digest.hexdigest()
    return output


def render_tsv(rows: list[dict[str, object]]) -> bytes:
    fieldnames = list(rows[0])
    if fieldnames[-1] != "store_set_sha256":
        raise ValueError("TSV must end in a guaranteed non-empty field")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_summary(
    rows: list[dict[str, object]],
    config: dict[str, object],
    tsv_sha256: str,
    input_hashes: dict[str, str],
) -> dict[str, object]:
    selected = [row for row in rows if row["selection_status"] == "SELECTED"]
    if not selected:
        raise ValueError("refinement policy selected no candidates")
    selected_by_material = {
        f"{row['species']}|{row['material_key']}": row["candidate_id"]
        for row in selected
    }
    for key, candidate_id in dict(config["expected_material_representatives"]).items():
        if selected_by_material.get(str(key)) != str(candidate_id):
            raise ValueError(
                f"expected representative mismatch for {key}: "
                f"{selected_by_material.get(str(key))} != {candidate_id}"
            )
    for key in config.get("required_single_materials", []):
        if str(key) not in selected_by_material:
            raise ValueError(f"required material has no unique provisional representative: {key}")
    contexts = [int(value) for value in config["contexts"]]
    context_summary = {
        str(context): {
            "eligible_source_count": sum(int(row[f"nonoverlap_windows_{context}"]) > 0 for row in selected),
            "nonoverlap_window_count": sum(int(row[f"nonoverlap_windows_{context}"]) for row in selected),
        }
        for context in contexts
    }
    pending = list(config["pending_before_formal_freeze"])
    return {
        "schema_version": "1.0",
        "state": "PROVISIONAL_NOT_TRAINABLE",
        "policy_status": config["policy_status"],
        "candidate_count": len(rows),
        "hard_gate_pass_count": sum(bool(row["hard_gate_pass"]) for row in rows),
        "selected_candidate_count": len(selected),
        "selected_species_count": len({row["species"] for row in selected}),
        "selected_genus_count": len({row["genus"] for row in selected}),
        "selected_base_count": sum(int(row["base_count"]) for row in selected),
        "selected_by_assembly_evidence": dict(sorted(Counter(str(row["assembly_evidence"]) for row in selected).items())),
        "selected_by_source": dict(sorted(Counter(str(row["source"]) for row in selected).items())),
        "selected_by_genus": dict(sorted(Counter(str(row["genus"]) for row in selected).items())),
        "selected_pending_assembly_evidence_count": sum(bool(row["assembly_evidence_pending"]) for row in selected),
        "selection_rejection_counts": dict(sorted(Counter(str(row["selection_status"]) for row in rows).items())),
        "hard_gate_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    for reason in str(row["hard_gate_reasons"]).split(";")
                    if reason != "."
                ).items()
            )
        ),
        "context_catalogs": context_summary,
        "expected_material_representatives": dict(config["expected_material_representatives"]),
        "required_single_materials": list(config.get("required_single_materials", [])),
        "pending_before_formal_freeze": pending,
        "input_sha256": input_hashes,
        "candidate_tsv_sha256": tsv_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a fail-closed chromosome/annotation/material refinement candidate audit")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("policy_status") != "candidate_only_not_trainable":
        raise ValueError("data refinement policy must remain candidate-only")
    candidate_rows = records(project_root, config)
    tsv = render_tsv(candidate_rows)
    input_paths = {
        "config": config_path,
        "candidate_universe": project_root / str(config["candidate_universe"]),
        "orientation_identity": project_root / "data_manifests/orientation_identity.tsv",
        "near_duplicate_clusters": project_root / "data_manifests/genome_near_duplicate_clusters.tsv",
        "assembly_candidates": project_root / "data_manifests/assembly_candidates.tsv",
        "archive_genome_qc": project_root / "data_manifests/archive_genome_qc.tsv",
        "annotation_qc": project_root / "data_manifests/annotation_qc.tsv",
        "archive_annotation_qc": project_root / "data_manifests/archive_annotation_qc.tsv",
    }
    input_hashes = {key: sha256(path) for key, path in input_paths.items()}
    summary = build_summary(candidate_rows, config, hashlib.sha256(tsv).hexdigest(), input_hashes)
    output_path = args.output.resolve()
    atomic_write(output_path, tsv)
    summary_path = output_path.with_suffix(".summary.json")
    atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

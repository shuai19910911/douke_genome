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

from legumegenomefm.data_refinement import (
    canonical_material_key,
    local_callable_intervals,
    select_unique_candidates,
    subtract_half_open_intervals,
)


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


def render_tsv(rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> bytes:
    if not rows and fieldnames is None:
        raise ValueError("fieldnames are required for an empty TSV")
    names = fieldnames or list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=names, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def status_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {row["candidate_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate candidate IDs in gate table")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize audited genome candidates and clean long-context intervals")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--closure", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--busco", required=True, type=Path)
    parser.add_argument("--record-qc-candidates", required=True, type=Path)
    parser.add_argument("--record-qc-records", required=True, type=Path)
    parser.add_argument("--record-qc-masks", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    candidates_path = args.candidates.resolve()
    closure_path = args.closure.resolve()
    provenance_path = args.provenance.resolve()
    busco_path = args.busco.resolve()
    record_candidates_path = args.record_qc_candidates.resolve()
    record_records_path = args.record_qc_records.resolve()
    record_masks_path = args.record_qc_masks.resolve()
    config_path = args.config.resolve()
    prefix = args.output_prefix.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contexts = sorted(int(value) for value in config["contexts"])
    minimum_context = min(contexts)
    material_aliases = config["deduplication"].get("material_aliases", {})

    candidates = read_tsv(candidates_path)
    closure = status_map(read_tsv(closure_path))
    provenance = status_map(read_tsv(provenance_path))
    busco = status_map(read_tsv(busco_path))
    record_qc = status_map(read_tsv(record_candidates_path))
    candidate_ids = {row["candidate_id"] for row in candidates}
    hard_ids = {row["candidate_id"] for row in candidates if row["hard_gate_pass"] == "True"}
    if set(closure) != hard_ids or set(provenance) != hard_ids:
        raise ValueError("closure/provenance candidate sets do not equal initial hard-gate set")
    if set(busco) != set(record_qc):
        raise ValueError("BUSCO and record-QC candidate sets differ")

    final_gate_rows: list[dict[str, object]] = []
    selection_inputs: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        reasons = [] if candidate["hard_gate_reasons"] == "." else candidate["hard_gate_reasons"].split(";")
        closure_status = closure.get(candidate_id, {}).get("status", ".")
        provenance_status = provenance.get(candidate_id, {}).get("status", ".")
        busco_status = busco.get(candidate_id, {}).get("status", ".")
        record_status = record_qc.get(candidate_id, {}).get("status", ".")
        if candidate["hard_gate_pass"] == "True":
            if closure_status != "PASS":
                reasons.append(f"annotation_closure_{closure_status.lower()}")
            if provenance_status != "PASS":
                reasons.append(f"source_provenance_{provenance_status.lower()}")
            if closure_status == "PASS" and provenance_status == "PASS":
                if busco_status != "PASS":
                    reasons.append(f"busco_{busco_status.lower()}")
                if record_status != "PASS":
                    reasons.append(f"record_qc_{record_status.lower()}")
        reasons = sorted(set(reason for reason in reasons if reason and reason != "."))
        final_pass = not reasons
        selection_row: dict[str, object] = dict(candidate)
        canonical_material = canonical_material_key(
            candidate["species"],
            candidate["material_key"],
            material_aliases,
        )
        selection_row["material_key"] = canonical_material
        selection_row["hard_gate_pass"] = final_pass
        selection_inputs.append(selection_row)
        final_gate_rows.append(
            {
                "candidate_id": candidate_id,
                "source": candidate["source"],
                "species": candidate["species"],
                "genus": candidate["genus"],
                "original_material_key": candidate["material_key"],
                "material_key": canonical_material,
                "relative_path": candidate["relative_path"],
                "member_name": candidate["member_name"],
                "annotation_id": candidate["annotation_id"],
                "annotation_path": candidate["annotation_path"],
                "annotation_member_name": candidate["annotation_member_name"],
                "assembly_evidence": candidate["assembly_evidence"],
                "official_assembly_level": candidate["official_assembly_level"],
                "base_count": candidate["base_count"],
                "n_fraction": candidate["n_fraction"],
                "n50": candidate["n50"],
                "large_sequence_fraction": candidate["large_sequence_fraction"],
                "orientation_group_id": candidate["orientation_group_id"],
                "near_duplicate_group_id": candidate["near_duplicate_group_id"],
                "candidate_near_duplicate_group_size": candidate["near_duplicate_group_size"],
                "final_near_group_selected_size": ".",
                "sampling_weight": ".",
                "initial_hard_gate_pass": candidate["hard_gate_pass"],
                "initial_hard_gate_reasons": candidate["hard_gate_reasons"],
                "annotation_closure_status": closure_status,
                "source_provenance_status": provenance_status,
                "busco_status": busco_status,
                "annotation_busco_complete_percent": busco.get(candidate_id, {}).get("annotation_complete_percent", "."),
                "genome_busco_complete_percent": busco.get(candidate_id, {}).get("genome_complete_percent", "."),
                "record_qc_status": record_status,
                "primary_nuclear_fraction": record_qc.get(candidate_id, {}).get("primary_nuclear_fraction", "."),
                "tiara_primary_prokaryotic_fraction": record_qc.get(candidate_id, {}).get(
                    "tiara_primary_prokaryotic_fraction", "."
                ),
                "univec_primary_fraction": record_qc.get(candidate_id, {}).get("univec_primary_fraction", "."),
                "qv_status": record_qc.get(candidate_id, {}).get("qv_status", "."),
                "final_gate_pass": str(final_pass),
                "final_gate_reasons": ";".join(reasons) or ".",
                "provisional_selection_status": candidate["selection_status"],
                "status": "PENDING_SELECTION",
            }
        )

    selections = select_unique_candidates(selection_inputs)
    gate_by_id = {str(row["candidate_id"]): row for row in final_gate_rows}
    for candidate_id, selection_status in selections.items():
        gate_by_id[candidate_id]["status"] = selection_status
    selected_ids = {candidate_id for candidate_id, value in selections.items() if value == "SELECTED"}
    if not selected_ids:
        raise ValueError("no candidates survived final selection")
    required_representatives: dict[str, str] = {}
    for required in config.get("required_single_materials", []):
        parts = str(required).split("|", 1)
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"invalid required material key: {required}")
        species, material = parts
        members = [
            candidate_id
            for candidate_id in selected_ids
            if gate_by_id[candidate_id]["species"] == species
            and gate_by_id[candidate_id]["material_key"] == material
        ]
        if len(members) != 1:
            raise ValueError(f"expected exactly one final {required} representative, found {members}")
        required_representatives[required] = members[0]
    selected_near_group_sizes = Counter(
        str(gate_by_id[candidate_id]["near_duplicate_group_id"]) for candidate_id in selected_ids
    )
    for candidate_id in selected_ids:
        row = gate_by_id[candidate_id]
        group_size = selected_near_group_sizes[str(row["near_duplicate_group_id"])]
        row["final_near_group_selected_size"] = group_size
        row["sampling_weight"] = round(1.0 / group_size, 12)

    record_rows = read_tsv(record_records_path)
    selected_record_rows = [
        row for row in record_rows if row["candidate_id"] in selected_ids and row["record_policy"] == "PRIMARY_NUCLEAR"
    ]
    record_names_by_candidate: dict[str, dict[int, str]] = defaultdict(dict)
    for row in selected_record_rows:
        record_names_by_candidate[row["candidate_id"]][int(row["contig_index"])] = row["sequence_name"]
    if set(record_names_by_candidate) != selected_ids:
        raise ValueError("one or more selected candidates have no primary nuclear records")

    all_masks = read_tsv(record_masks_path)
    selected_masks = [row for row in all_masks if row["candidate_id"] in selected_ids]
    masks_by_record: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in selected_masks:
        masks_by_record[(row["candidate_id"], row["sequence_name"])].append(
            (int(row["start_0based"]), int(row["end_0based_exclusive"]))
        )

    interval_path = Path(f"{prefix}.intervals.tsv")
    interval_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_interval_path = interval_path.with_name(f".{interval_path.name}.tmp.{os.getpid()}")
    context_capacity: dict[str, Counter[int]] = defaultdict(Counter)
    clean_interval_count = 0
    clean_bases = Counter()
    try:
        with temporary_interval_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "candidate_id",
                "contig_index",
                "sequence_name",
                "record_start_0based",
                "store_start",
                "length",
                "status",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for candidate_id in sorted(selected_ids):
                manifest = json.loads(
                    (root / "data/processed/sequence_store" / candidate_id / "manifest.json").read_text(encoding="utf-8")
                )
                for index, sequence_name in sorted(record_names_by_candidate[candidate_id].items()):
                    contig_offset, source_intervals = local_callable_intervals(manifest, index)
                    clean = subtract_half_open_intervals(
                        source_intervals,
                        masks_by_record.get((candidate_id, sequence_name), []),
                    )
                    for start, end in clean:
                        length = end - start
                        if length < minimum_context:
                            continue
                        writer.writerow(
                            {
                                "candidate_id": candidate_id,
                                "contig_index": index,
                                "sequence_name": sequence_name,
                                "record_start_0based": start,
                                "store_start": contig_offset + start,
                                "length": length,
                                "status": "TRAINABLE",
                            }
                        )
                        clean_interval_count += 1
                        clean_bases[candidate_id] += length
                        for context in contexts:
                            context_capacity[candidate_id][context] += length // context
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_interval_path, interval_path)
    finally:
        temporary_interval_path.unlink(missing_ok=True)

    context_rows: list[dict[str, object]] = []
    for candidate_id in sorted(selected_ids):
        for context in contexts:
            capacity = context_capacity[candidate_id][context]
            context_rows.append(
                {
                    "candidate_id": candidate_id,
                    "species": gate_by_id[candidate_id]["species"],
                    "material_key": gate_by_id[candidate_id]["material_key"],
                    "context_length": context,
                    "eligible_nonoverlap_windows": capacity,
                    "eligible": str(capacity > 0),
                    "status": "ELIGIBLE" if capacity > 0 else "INELIGIBLE_LENGTH_SPECIFIC",
                }
            )

    final_gate_rows.sort(key=lambda row: str(row["candidate_id"]))
    selected_rows = [row for row in final_gate_rows if row["candidate_id"] in selected_ids]
    selected_rows.sort(key=lambda row: (str(row["species"]), str(row["material_key"]), str(row["candidate_id"])))
    selected_record_rows.sort(key=lambda row: (row["candidate_id"], int(row["contig_index"])))
    selected_masks.sort(key=lambda row: (row["candidate_id"], row["sequence_name"], int(row["start_0based"])))

    outputs = {
        "candidates": (Path(f"{prefix}.candidates.tsv"), render_tsv(final_gate_rows)),
        "selected": (Path(f"{prefix}.selected.tsv"), render_tsv(selected_rows)),
        "records": (Path(f"{prefix}.records.tsv"), render_tsv(selected_record_rows)),
        "masks": (
            Path(f"{prefix}.masks.tsv"),
            render_tsv(
                selected_masks,
                ["candidate_id", "sequence_name", "start_0based", "end_0based_exclusive", "length", "sources", "status"],
            ),
        ),
        "contexts": (Path(f"{prefix}.contexts.tsv"), render_tsv(context_rows)),
    }
    for _, (path, payload) in outputs.items():
        atomic_write(path, payload)

    status_counts = Counter(str(row["status"]) for row in final_gate_rows)
    selected_by_source = Counter(str(row["source"]) for row in selected_rows)
    selected_by_genus = Counter(str(row["genus"]) for row in selected_rows)
    selected_by_evidence = Counter(str(row["assembly_evidence"]) for row in selected_rows)
    context_summary = {
        str(context): {
            "eligible_source_count": sum(context_capacity[candidate_id][context] > 0 for candidate_id in selected_ids),
            "nonoverlap_window_count": sum(context_capacity[candidate_id][context] for candidate_id in selected_ids),
        }
        for context in contexts
    }
    input_paths = {
        "candidates": candidates_path,
        "closure": closure_path,
        "provenance": provenance_path,
        "busco": busco_path,
        "record_qc_candidates": record_candidates_path,
        "record_qc_records": record_records_path,
        "record_qc_masks": record_masks_path,
        "config": config_path,
    }
    output_hashes = {key: hashlib.sha256(payload).hexdigest() for key, (_, payload) in outputs.items()}
    output_hashes["intervals"] = sha256(interval_path)
    summary = {
        "schema_version": "1.0",
        "state": "CANDIDATE_SET_READY_DATA_NOT_YET_PACKAGED",
        "input_candidate_count": len(candidates),
        "final_gate_pass_count": sum(row["final_gate_pass"] == "True" for row in final_gate_rows),
        "selected_candidate_count": len(selected_ids),
        "selected_near_duplicate_group_count": len(selected_near_group_sizes),
        "selected_multi_member_near_group_count": sum(size > 1 for size in selected_near_group_sizes.values()),
        "near_duplicate_weighted_effective_source_count": round(
            sum(float(gate_by_id[candidate_id]["sampling_weight"]) for candidate_id in selected_ids), 12
        ),
        "selected_primary_record_count": len(selected_record_rows),
        "selected_mask_interval_count": len(selected_masks),
        "trainable_interval_count": clean_interval_count,
        "trainable_bases_at_minimum_context": sum(clean_bases.values()),
        "required_material_representatives": required_representatives,
        "selection_status_counts": dict(sorted(status_counts.items())),
        "selected_by_source": dict(sorted(selected_by_source.items())),
        "selected_by_genus": dict(sorted(selected_by_genus.items())),
        "selected_by_assembly_evidence": dict(sorted(selected_by_evidence.items())),
        "context_catalogs": context_summary,
        "input_sha256": {key: sha256(path) for key, path in input_paths.items()},
        "output_sha256": output_hashes,
    }
    atomic_write(
        Path(f"{prefix}.summary.json"),
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

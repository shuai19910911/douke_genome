from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from legumegenomefm.sequence_store import PackedSequenceStore


@dataclass(frozen=True)
class TrainingManifestResult:
    manifest_path: Path
    summary_path: Path
    summary: dict[str, object]


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_training_manifest(
    source_rows: Iterable[dict[str, object]],
    store_root: Path,
    output_path: Path,
    *,
    store_root_reference: str,
    cold_genera: set[str],
    max_context: int,
) -> TrainingManifestResult:
    if max_context < 1:
        raise ValueError("max_context must be positive")
    store_root = Path(store_root)
    frozen: list[dict[str, object]] = []
    for source in sorted(source_rows, key=lambda row: str(row["candidate_id"])):
        candidate_id = str(source["candidate_id"])
        store = PackedSequenceStore(store_root / candidate_id)
        identity = store.manifest.get("identity", {})
        if identity.get("candidate_id") != candidate_id:
            raise ValueError(f"sequence store candidate identity mismatch: {candidate_id}")
        callable_intervals = [
            interval
            for interval in store.callable_intervals
            if int(interval["length"]) >= max_context
        ]
        callable_bases = sum(int(interval["length"]) for interval in callable_intervals)
        eligible_window_count = sum(
            int(interval["length"]) - max_context + 1 for interval in callable_intervals
        )
        if eligible_window_count <= 0:
            raise ValueError(f"no callable interval supports max context: {candidate_id}")
        genus = str(source["genus"])
        split = "cold_genus_holdout" if genus in cold_genera else "pretrain"
        frozen.append(
            {
                "candidate_id": candidate_id,
                "genus": genus,
                "species": str(source["species"]),
                "material_key": str(source["material_key"]),
                "material_version_group_id": str(source.get("material_version_group_id", f"material-{candidate_id}")),
                "material_version_group_size": int(source.get("material_version_group_size", 1)),
                "near_duplicate_group_id": str(source["near_duplicate_group_id"]),
                "near_duplicate_group_size": int(source["near_duplicate_group_size"]),
                "split": split,
                "base_count": store.base_count,
                "callable_bases": callable_bases,
                "eligible_window_count_at_max_context": eligible_window_count,
                "store_manifest_sha256": (store.store_dir / "READY").read_text(encoding="ascii").strip(),
                "sampling_weight": 0.0,
            }
        )

    pretrain = [source for source in frozen if source["split"] == "pretrain"]
    species_capacity: dict[str, int] = defaultdict(int)
    for source in pretrain:
        species_capacity[str(source["species"])] += int(source["callable_bases"])
    species_mass = {
        species: capacity**0.3 for species, capacity in species_capacity.items()
    }
    species_total = sum(species_mass.values())
    by_species: dict[str, list[dict[str, object]]] = defaultdict(list)
    for source in pretrain:
        by_species[str(source["species"])].append(source)
    for species, members in by_species.items():
        within = [
            math.sqrt(int(source["callable_bases"]))
            / math.sqrt(max(int(source["near_duplicate_group_size"]), int(source["material_version_group_size"])))
            for source in members
        ]
        within_total = sum(within)
        for source, local_mass in zip(members, within):
            source["sampling_weight"] = species_mass[species] / species_total * local_mass / within_total

    manifest = {
        "schema_version": "1.0",
        "store_root": store_root_reference,
        "max_context": max_context,
        "cold_genera": sorted(cold_genera),
        "sampling_policy": {
            "species_callable_base_temperature": 0.3,
            "within_species_callable_base_temperature": 0.5,
            "near_duplicate_group_penalty": 0.5,
            "material_version_group_penalty": 0.5,
        },
        "sources": frozen,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output_path = Path(output_path)
    _atomic_write(output_path, manifest_bytes)
    summary = {
        "schema_version": "1.0",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_count": len(frozen),
        "pretrain_source_count": len(pretrain),
        "cold_genus_source_count": len(frozen) - len(pretrain),
        "species_count": len({source["species"] for source in frozen}),
        "genus_count": len({source["genus"] for source in frozen}),
        "pretrain_callable_bases": sum(int(source["callable_bases"]) for source in pretrain),
        "cold_genus_callable_bases": sum(
            int(source["callable_bases"]) for source in frozen if source["split"] == "cold_genus_holdout"
        ),
        "max_context": max_context,
    }
    summary_path = output_path.with_suffix(".summary.json")
    _atomic_write(summary_path, (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return TrainingManifestResult(output_path, summary_path, summary)


class GenomeWindowSampler:
    def __init__(
        self,
        manifest_path: Path,
        project_root: Path,
        *,
        context_length: int,
        split: str,
        seed: int,
    ) -> None:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if context_length < 1 or context_length > int(payload["max_context"]):
            raise ValueError("context length exceeds the frozen maximum context")
        self.context_length = context_length
        self.seed = int(seed)
        self.sources: list[dict[str, object]] = []
        self.stores: list[PackedSequenceStore] = []
        self.intervals: list[list[dict[str, int]]] = []
        weights: list[float] = []
        store_root = Path(project_root) / payload["store_root"]
        for source in payload["sources"]:
            if source["split"] != split:
                continue
            store = PackedSequenceStore(store_root / source["candidate_id"])
            intervals = [
                interval
                for interval in store.callable_intervals
                if int(interval["length"]) >= context_length
            ]
            if not intervals:
                continue
            self.sources.append(source)
            self.stores.append(store)
            self.intervals.append(intervals)
            weight = float(source["sampling_weight"])
            weights.append(weight if weight > 0 else 1.0)
        if not self.sources:
            raise ValueError(f"no sources available for split: {split}")
        self.source_probabilities = np.asarray(weights, dtype=np.float64)
        self.source_probabilities /= self.source_probabilities.sum()

    def _rng(self, global_microstep: int, rank: int) -> np.random.Generator:
        material = f"{self.seed}\0{global_microstep}\0{rank}".encode("ascii")
        derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
        return np.random.default_rng(derived)

    def sample_batch(self, *, batch_size: int, global_microstep: int, rank: int) -> torch.Tensor:
        if batch_size < 1 or global_microstep < 0 or rank < 0:
            raise ValueError("invalid sampling coordinates")
        rng = self._rng(global_microstep, rank)
        batch = np.empty((batch_size, self.context_length), dtype=np.int64)
        for row in range(batch_size):
            source_index = int(rng.choice(len(self.sources), p=self.source_probabilities))
            intervals = self.intervals[source_index]
            capacities = np.asarray(
                [int(interval["length"]) - self.context_length + 1 for interval in intervals],
                dtype=np.float64,
            )
            capacities /= capacities.sum()
            interval = intervals[int(rng.choice(len(intervals), p=capacities))]
            lower = int(interval["start"])
            upper = lower + int(interval["length"]) - self.context_length
            start = int(rng.integers(lower, upper + 1))
            codes = self.stores[source_index].decode(start, self.context_length).astype(np.int64)
            if bool(rng.integers(0, 2)):
                codes = (3 - codes[::-1]).copy()
            batch[row] = codes + 2
        return torch.from_numpy(batch)

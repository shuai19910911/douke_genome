from __future__ import annotations

import json
from pathlib import Path

import torch

from legumegenomefm.sequence_store import PackedSequenceStoreWriter
from legumegenomefm.training_data import (
    GenomeWindowSampler,
    build_refined_training_manifest,
    build_training_manifest,
)


def _store(root: Path, candidate_id: str, sequence: bytes) -> None:
    writer = PackedSequenceStoreWriter(
        root / candidate_id,
        {"candidate_id": candidate_id, "implementation_sha256": "1" * 64},
    )
    writer.start_contig("chr1")
    writer.add_bases(sequence)
    writer.finalize()


def test_training_manifest_and_stateless_sampler(tmp_path: Path) -> None:
    store_root = tmp_path / "stores"
    _store(store_root, "a" * 16, b"ACGT" * 100)
    _store(store_root, "b" * 16, b"TGCA" * 100)
    sources = [
        {
            "candidate_id": "a" * 16,
            "genus": "Glycine",
            "species": "Glycine max",
            "material_key": "a",
            "near_duplicate_group_id": "near-a",
            "near_duplicate_group_size": 1,
        },
        {
            "candidate_id": "b" * 16,
            "genus": "Vigna",
            "species": "Vigna radiata",
            "material_key": "b",
            "near_duplicate_group_id": "near-b",
            "near_duplicate_group_size": 1,
        },
    ]
    result = build_training_manifest(
        sources,
        store_root,
        tmp_path / "dataset.json",
        store_root_reference="stores",
        cold_genera={"Vigna"},
        max_context=32,
    )
    assert result.summary["pretrain_source_count"] == 1
    assert result.summary["cold_genus_source_count"] == 1
    sampler = GenomeWindowSampler(result.manifest_path, tmp_path, context_length=32, split="pretrain", seed=9)
    first = sampler.sample_batch(batch_size=2, global_microstep=7, rank=0)
    repeated = sampler.sample_batch(batch_size=2, global_microstep=7, rank=0)
    another_rank = sampler.sample_batch(batch_size=2, global_microstep=7, rank=1)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, another_rank)
    assert first.shape == (2, 32)
    assert first.min().item() >= 2 and first.max().item() <= 5


def test_sampler_rejects_context_larger_than_frozen_max(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps({"schema_version": "1.0", "max_context": 16, "store_root": "stores", "sources": []}) + "\n")
    try:
        GenomeWindowSampler(path, tmp_path, context_length=32, split="pretrain", seed=1)
    except ValueError as exc:
        assert "context" in str(exc)
    else:
        raise AssertionError("oversized context was accepted")


def test_refined_manifest_sampler_uses_only_final_trainable_intervals(tmp_path: Path) -> None:
    store_root = tmp_path / "stores"
    _store(store_root, "a" * 16, b"C" * 40 + b"A" * 128 + b"G" * 232)
    _store(store_root, "b" * 16, b"G" * 40 + b"T" * 128 + b"C" * 232)
    sources = [
        {
            "candidate_id": "a" * 16,
            "genus": "Glycine",
            "species": "Glycine max",
            "material_key": "a",
            "near_duplicate_group_id": "near-a",
            "final_near_group_selected_size": 1,
        },
        {
            "candidate_id": "b" * 16,
            "genus": "Vigna",
            "species": "Vigna radiata",
            "material_key": "b",
            "near_duplicate_group_id": "near-b",
            "final_near_group_selected_size": 1,
        },
    ]
    intervals = [
        {
            "candidate_id": candidate,
            "contig_index": 0,
            "sequence_name": "chr1",
            "record_start_0based": 40,
            "store_start": 40,
            "length": 128,
            "status": "TRAINABLE",
        }
        for candidate in ("a" * 16, "b" * 16)
    ]
    capacities = [
        {
            "candidate_id": candidate,
            "context_length": context,
            "eligible_nonoverlap_windows": 128 // context,
        }
        for candidate in ("a" * 16, "b" * 16)
        for context in (16, 32, 64)
    ]
    result = build_refined_training_manifest(
        sources,
        intervals,
        capacities,
        store_root,
        tmp_path / "dataset.json",
        store_root_reference="stores",
        cold_genera={"Vigna"},
        contexts=(16, 32, 64),
        provenance={"final_summary_sha256": "1" * 64},
    )
    payload = json.loads(result.manifest_path.read_text())
    assert payload["schema_version"] == "2.0"
    assert payload["sources"][0]["trainable_intervals"][0]["store_start"] == 40
    assert result.summary["context_catalogs"]["64"]["eligible_source_count"] == 2
    sampler = GenomeWindowSampler(result.manifest_path, tmp_path, context_length=32, split="pretrain", seed=4)
    batch = sampler.sample_batch(batch_size=8, global_microstep=3, rank=0)
    assert batch.shape == (8, 32)
    assert set(batch.unique().tolist()) <= {2, 5}

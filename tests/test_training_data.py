from __future__ import annotations

import json
from pathlib import Path

import torch

from legumegenomefm.sequence_store import PackedSequenceStoreWriter
from legumegenomefm.training_data import GenomeWindowSampler, build_training_manifest


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

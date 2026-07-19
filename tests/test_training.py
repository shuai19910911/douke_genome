from __future__ import annotations

from pathlib import Path

from legumegenomefm.sequence_store import PackedSequenceStoreWriter
from legumegenomefm.training import TrainConfig, run_training
from legumegenomefm.training_data import build_training_manifest


def _manifest(tmp_path: Path) -> Path:
    store_root = tmp_path / "stores"
    candidate_id = "a" * 16
    writer = PackedSequenceStoreWriter(store_root / candidate_id, {"candidate_id": candidate_id})
    writer.start_contig("chr1")
    writer.add_bases((b"ACGTTGCA" * 100))
    writer.finalize()
    result = build_training_manifest(
        [
            {
                "candidate_id": candidate_id,
                "genus": "Glycine",
                "species": "Glycine max",
                "material_key": "a",
                "near_duplicate_group_id": "near-a",
                "near_duplicate_group_size": 1,
            }
        ],
        store_root,
        tmp_path / "dataset.json",
        store_root_reference="stores",
        cold_genera=set(),
        max_context=32,
    )
    return result.manifest_path


def test_training_checkpoint_resume_reaches_same_token_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = TrainConfig(
        dataset_manifest=manifest,
        project_root=tmp_path,
        output_dir=tmp_path / "run",
        context_length=32,
        micro_batch_size=1,
        global_batch_tokens=32,
        max_tokens=128,
        warmup_tokens=32,
        learning_rate=1e-3,
        min_lr_ratio=0.1,
        weight_decay=0.01,
        gradient_clip=1.0,
        seed=11,
        precision="fp32",
        checkpoint_every_steps=2,
        log_every_steps=1,
        model={
            "d_model": 16,
            "n_layers": 1,
            "ffn_multiple": 2,
            "kernel_size": 3,
            "dilations": [1, 2],
            "dropout": 0.0,
        },
    )
    partial = run_training(config, device_override="cpu", stop_after_steps=2)
    assert partial.step == 2 and partial.tokens_seen == 64
    resumed = run_training(config, device_override="cpu", resume=True)
    assert resumed.step == 4 and resumed.tokens_seen == 128
    assert (config.output_dir / "checkpoints" / "step_00000004" / "READY").is_file()
    assert resumed.loss > 0


def test_training_contract_rejects_nondivisible_global_batch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    config = TrainConfig(
        dataset_manifest=manifest,
        project_root=tmp_path,
        output_dir=tmp_path / "run",
        context_length=32,
        micro_batch_size=1,
        global_batch_tokens=33,
        max_tokens=100,
        warmup_tokens=0,
        learning_rate=1e-3,
        min_lr_ratio=0.1,
        weight_decay=0.0,
        gradient_clip=1.0,
        seed=1,
        precision="fp32",
        checkpoint_every_steps=1,
        log_every_steps=1,
        model={"d_model": 16, "n_layers": 1, "ffn_multiple": 2, "kernel_size": 3, "dilations": [1]},
    )
    try:
        run_training(config, device_override="cpu")
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("nondivisible global batch was accepted")

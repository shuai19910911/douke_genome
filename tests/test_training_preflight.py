from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from legumegenomefm.sequence_store import PackedSequenceStoreWriter


_SCRIPT = Path(__file__).parents[1] / "scripts" / "preflight_training.py"
_SPEC = importlib.util.spec_from_file_location("preflight_training", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _store(root: Path, candidate_id: str) -> str:
    writer = PackedSequenceStoreWriter(root / candidate_id, {"candidate_id": candidate_id})
    writer.start_contig("chr1")
    writer.add_bases(b"ACGT" * 1024)
    return writer.finalize().manifest_sha256


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    stores = project / "stores"
    first = "a" * 16
    second = "b" * 16
    first_sha = _store(stores, first)
    second_sha = _store(stores, second)
    payload = {
        "store_root": "stores",
        "sources": [
            {
                "candidate_id": first,
                "split": "pretrain",
                "sampling_weight": 1.0,
                "store_manifest_sha256": first_sha,
            },
            {
                "candidate_id": second,
                "split": "cold_genus_holdout",
                "sampling_weight": 0.0,
                "store_manifest_sha256": second_sha,
            },
        ],
    }
    manifest = project / "data_release" / "training_dataset.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    release = {
        "state": "READY",
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    release_bytes = (json.dumps(release) + "\n").encode()
    (manifest.parent / "training_dataset.release.json").write_bytes(release_bytes)
    (manifest.parent / "TRAINING_DATASET_READY").write_text(hashlib.sha256(release_bytes).hexdigest() + "\n")
    return project, manifest


def test_validate_dataset_requires_both_roles_and_store_hashes(tmp_path: Path) -> None:
    project, manifest = _dataset(tmp_path)
    assert _MODULE.validate_dataset(project, manifest) == {
        "source_count": 2,
        "pretrain": 1,
        "cold_genus_holdout": 1,
    }
    payload = json.loads(manifest.read_text())
    payload["sources"][0]["store_manifest_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not bind"):
        _MODULE.validate_dataset(project, manifest)
    release_path = manifest.parent / "training_dataset.release.json"
    release = json.loads(release_path.read_text())
    release["dataset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    release_bytes = (json.dumps(release) + "\n").encode()
    release_path.write_bytes(release_bytes)
    (manifest.parent / "TRAINING_DATASET_READY").write_text(hashlib.sha256(release_bytes).hexdigest() + "\n")
    with pytest.raises(ValueError, match="READY/hash"):
        _MODULE.validate_dataset(project, manifest)


def test_validate_mode_is_fail_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    _MODULE.validate_mode("fresh", output, None)
    output.mkdir()
    with pytest.raises(ValueError, match="refuses existing"):
        _MODULE.validate_mode("fresh", output, None)
    with pytest.raises(ValueError, match="checkpoint READY"):
        _MODULE.validate_mode("resume", output, None)
    checkpoint = output / "checkpoints" / "step_00000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "READY").write_text("x")
    _MODULE.validate_mode("resume", output, None)
    new_output = tmp_path / "new"
    _MODULE.validate_mode("initialize", new_output, checkpoint)


def test_confined_relative_rejects_escape() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _MODULE.confined_relative("../outside", "path")
    with pytest.raises(ValueError, match="unsafe"):
        _MODULE.confined_relative("/absolute", "path")


def test_gpu_contract_rejects_partial_final_global_batch_before_cuda_probe() -> None:
    payload = {
        "context_length": 1024,
        "micro_batch_size": 1,
        "global_batch_tokens": 1024,
        "max_tokens": 1025,
        "precision": "bf16",
    }
    with pytest.raises(ValueError, match="whole number"):
        _MODULE.validate_gpu_contract(payload, nproc=1, minimum_free_mib=1)


def test_frozen_stage_budgets_are_complete_global_batches() -> None:
    root = Path(__file__).parents[1]
    for path in sorted((root / "configs").glob("pretrain_stage*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["max_tokens"] % payload["global_batch_tokens"] == 0, path

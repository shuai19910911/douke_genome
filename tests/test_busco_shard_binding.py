from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "merge_busco_mode_shards.py"
_SPEC = importlib.util.spec_from_file_location("merge_busco_mode_shards_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_AGGREGATE_SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_busco_results.py"
_AGGREGATE_SPEC = importlib.util.spec_from_file_location("aggregate_busco_results_test", _AGGREGATE_SCRIPT)
assert _AGGREGATE_SPEC is not None and _AGGREGATE_SPEC.loader is not None
_AGGREGATE = importlib.util.module_from_spec(_AGGREGATE_SPEC)
_AGGREGATE_SPEC.loader.exec_module(_AGGREGATE)


def test_busco_mode_shard_must_bind_current_lineage_receipt() -> None:
    receipt_sha = "a" * 64
    payload = {
        "candidate_id": "b" * 16,
        "status": "PASS",
        "requested_modes": ["proteins"],
        "lineage_receipt_sha256": receipt_sha,
        "busco": {"proteins": {}},
    }
    assert _MODULE.valid_mode(payload, "b" * 16, "proteins", receipt_sha)
    assert not _MODULE.valid_mode(payload, "b" * 16, "proteins", "c" * 64)
    payload.pop("lineage_receipt_sha256")
    assert not _MODULE.valid_mode(payload, "b" * 16, "proteins", receipt_sha)


def test_busco_worker_validates_and_records_lineage_receipt() -> None:
    text = (Path(__file__).parents[1] / "scripts/run_busco_task.py").read_text()
    assert "validate_busco_lineage(project_root, lineage)" in text
    assert 'result["lineage_receipt_sha256"]' in text


def test_busco_controller_and_finalizer_pass_current_lineage_ready() -> None:
    root = Path(__file__).parents[1]
    controller = (root / "scripts/submit_split_qc_batches.py").read_text()
    finalizer = (root / "scripts/slurm/finalize_data_refinement.sbatch").read_text()
    assert "--lineage-ready" in controller
    assert "busco_lineage_eudicots_odb10.READY" in controller
    assert "--lineage-ready" in finalizer
    assert "verify_busco_lineage.py" in finalizer
    assert "--full" in finalizer


def test_busco_aggregator_rejects_lineage_version_mismatch() -> None:
    receipt_sha = "a" * 64
    config = {
        "lineage_receipt_sha256": receipt_sha,
        "expected_busco_version": "5.8.3",
        "expected_dataset_name": "eudicotyledons_odb10",
        "expected_creation_date": "2024-01-08",
        "expected_number_of_buscos": 2326,
        "expected_number_of_species": 31,
    }
    summary = {
        "versions": {"busco": "5.8.3"},
        "lineage_dataset": {
            "name": "eudicotyledons_odb10",
            "creation_date": "2024-01-08",
            "number_of_buscos": "2326",
            "number_of_species": "31",
        },
    }
    shard = {
        "lineage_receipt_sha256": receipt_sha,
        "busco": {
            "proteins": {"summary": summary},
            "genome": {"summary": summary},
        },
    }
    _AGGREGATE.validate_busco_provenance(shard, config)
    summary["lineage_dataset"]["creation_date"] = "wrong"
    try:
        _AGGREGATE.validate_busco_provenance(shard, config)
    except ValueError as exc:
        assert "creation_date" in str(exc)
    else:
        raise AssertionError("mismatched lineage date was accepted")

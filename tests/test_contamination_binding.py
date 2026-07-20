from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts" / "submit_qc_repair_batches.py"
_SPEC = importlib.util.spec_from_file_location("submit_qc_repair_batches_test", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _payload(reference_sha: str | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "candidate_id": "a" * 16,
        "status": "PASS",
        "tiara": {"record_class_base_counts": {}},
        "univec": {"records": [{"intervals_1based_inclusive": []}]},
    }
    if reference_sha is not None:
        payload["contamination_reference_receipt_sha256"] = reference_sha
    return payload


def test_contamination_shard_requires_direct_or_legacy_reference_binding(tmp_path: Path) -> None:
    reference_sha = "b" * 64
    path = tmp_path / f"{'a' * 16}.json"
    path.write_text(json.dumps(_payload(reference_sha)) + "\n")
    assert _MODULE.contamination_valid(path, "a" * 16, reference_sha, {})
    assert not _MODULE.contamination_valid(path, "a" * 16, "c" * 64, {})

    path.write_text(json.dumps(_payload(None)) + "\n")
    shard_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _MODULE.contamination_valid(path, "a" * 16, reference_sha, {"a" * 16: shard_sha})
    assert not _MODULE.contamination_valid(path, "a" * 16, reference_sha, {"a" * 16: "d" * 64})


def test_contamination_worker_records_reference_receipt() -> None:
    text = (Path(__file__).parents[1] / "scripts/run_contamination_task.py").read_text()
    assert "validate_contamination_references(" in text
    assert 'result["contamination_reference_receipt_sha256"]' in text


def test_contamination_finalizer_and_aggregator_verify_reference_bindings() -> None:
    root = Path(__file__).parents[1]
    finalizer = (root / "scripts/slurm/finalize_data_refinement.sbatch").read_text()
    aggregator = (root / "scripts/aggregate_record_qc.py").read_text()
    assert "verify_contamination_references.py" in finalizer
    assert "--full" in finalizer
    assert "load_contamination_legacy_bindings" in aggregator
    assert "contamination_reference_binding" in aggregator

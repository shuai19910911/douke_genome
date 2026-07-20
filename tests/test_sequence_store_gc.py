from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from legumegenomefm.sequence_store import PackedSequenceStoreWriter


_SCRIPT = Path(__file__).parents[1] / "scripts" / "prune_sequence_stores.py"
_SPEC = importlib.util.spec_from_file_location("prune_sequence_stores", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _store(root: Path, candidate_id: str) -> None:
    writer = PackedSequenceStoreWriter(root / candidate_id, {"candidate_id": candidate_id})
    writer.start_contig("chr1")
    writer.add_bases(b"ACGT" * 64)
    writer.finalize()


def test_store_gc_removes_only_unreferenced_stores_and_preserves_raw(tmp_path: Path) -> None:
    project = tmp_path / "project"
    stores = project / "data/processed/sequence_store"
    identifiers = ("a" * 16, "b" * 16, "c" * 16)
    for candidate_id in identifiers:
        _store(stores, candidate_id)
    raw_sentinel = project / "data/raw/DO_NOT_TOUCH"
    raw_sentinel.parent.mkdir(parents=True)
    raw_sentinel.write_text("raw")

    release_dir = project / "data_release"
    release_dir.mkdir()
    manifest = release_dir / "training_dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "state": "READY",
                "store_root": "data/processed/sequence_store",
                "sources": [
                    {"candidate_id": identifiers[0]},
                    {"candidate_id": identifiers[1]},
                ],
            }
        )
        + "\n"
    )
    release = {
        "state": "READY",
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    release_bytes = (json.dumps(release) + "\n").encode()
    (release_dir / "training_dataset.release.json").write_bytes(release_bytes)
    (release_dir / "TRAINING_DATASET_READY").write_text(
        hashlib.sha256(release_bytes).hexdigest() + "\n"
    )

    plan = _MODULE.build_gc_plan(project, manifest)
    assert plan["keep_candidate_ids"] == list(identifiers[:2])
    assert [row["candidate_id"] for row in plan["remove"]] == [identifiers[2]]
    receipt_path = project / "data_manifests/sequence_store_gc_receipt.json"
    receipt = _MODULE.execute_gc(project, manifest, receipt_path)
    assert receipt["state"] == "COMPLETED"
    assert (stores / identifiers[0]).is_dir()
    assert (stores / identifiers[1]).is_dir()
    assert not (stores / identifiers[2]).exists()
    assert raw_sentinel.read_text() == "raw"
    assert receipt_path.is_file()
    assert receipt_path.with_suffix(".plan.json").is_file()

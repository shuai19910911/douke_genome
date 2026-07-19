from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_matrix_meets_frozen_scope_and_leakage_requirements() -> None:
    payload = yaml.safe_load((ROOT / "configs/evaluation_matrix.yaml").read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    assert len(tasks) >= 18
    assert sum(bool(task["core"]) for task in tasks) >= 12
    assert len({task["id"] for task in tasks}) == len(tasks)
    categories = {task["category"] for task in tasks}
    assert {"base", "region", "gene", "variant", "regulatory", "breeding"} <= categories
    assert len(payload["common_leakage_groups"]) >= 4
    assert len(payload["baseline_suite"]["external"]) >= 6
    for task in tasks:
        assert task["availability"]
        assert task["split"]
        assert task["metrics"]

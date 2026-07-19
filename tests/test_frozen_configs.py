from __future__ import annotations

from pathlib import Path

import yaml

from legumegenomefm.model import LegumeGenomeConfig, LegumeGenomeModel


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "configs" / f"pretrain_stage{stage}.yaml" for stage in (1, 2, 3)]


def test_frozen_model_parameter_count() -> None:
    payload = yaml.safe_load(CONFIGS[0].read_text())
    model_values = dict(payload["model"])
    model_values["dilations"] = tuple(model_values["dilations"])
    model = LegumeGenomeModel(LegumeGenomeConfig(**model_values))
    assert model.parameter_count() == 88_946_028


def test_frozen_curriculum_has_exact_budget_and_gpu_divisibility() -> None:
    payloads = [yaml.safe_load(path.read_text()) for path in CONFIGS]
    assert sum(int(payload["max_tokens"]) for payload in payloads) == 99_999_547_392
    assert [int(payload["context_length"]) for payload in payloads] == [1024, 4096, 16384]
    for payload in payloads:
        max_tokens = int(payload["max_tokens"])
        global_tokens = int(payload["global_batch_tokens"])
        assert max_tokens % global_tokens == 0
        for world_size in (1, 2, 4, 8):
            local_micro_tokens = int(payload["context_length"]) * int(payload["micro_batch_size"]) * world_size
            assert global_tokens % local_micro_tokens == 0

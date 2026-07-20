from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from legumegenomefm.hiermamba import (
    HierMambaConfig,
    HierMambaForMaskedLM,
    reverse_complement_token_ids,
)


PROJECT_ROOT = Path(__file__).parents[1]


def candidate_config() -> HierMambaConfig:
    payload = yaml.safe_load((PROJECT_ROOT / "configs/pretrain_h20_candidate.yaml").read_text())
    return HierMambaConfig.from_mapping(payload["model"], payload["activation_checkpointing"])


def test_candidate_model_config_closes_at_128_bp_and_2048_latents() -> None:
    config = candidate_config()
    assert config.latent_stride_bp == 128
    assert config.maximum_context_bp // config.latent_stride_bp == 2048
    assert config.encoder_channels[-1] == 1024
    assert config.global_layers == 24


def test_reverse_complement_mapping_is_an_involution() -> None:
    complement = torch.tensor([0, 1, 5, 4, 3, 2, 6], dtype=torch.long)
    sequence = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)
    reverse = reverse_complement_token_ids(sequence, complement)
    assert reverse.tolist() == [[6, 2, 3, 4, 5, 1, 0]]
    assert torch.equal(reverse_complement_token_ids(reverse, complement), sequence)


def test_model_instantiation_fails_before_parameter_allocation_without_production_mamba2() -> None:
    with pytest.raises(RuntimeError, match="Mamba-2 production backend is unavailable"):
        HierMambaForMaskedLM(candidate_config())

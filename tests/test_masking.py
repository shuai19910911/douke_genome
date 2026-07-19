from __future__ import annotations

import torch

from legumegenomefm.masking import mask_span_mlm
from legumegenomefm.tokenizer import MASK_TOKEN_ID, tokenize_dna


def test_span_masking_is_deterministic_and_labels_only_selected_tokens() -> None:
    tokens = tokenize_dna("ACGT" * 64).reshape(2, 128)
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)
    first_inputs, first_labels = mask_span_mlm(tokens, first_generator, mask_probability=0.15, mean_span_length=3.0)
    second_inputs, second_labels = mask_span_mlm(tokens, second_generator, mask_probability=0.15, mean_span_length=3.0)
    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_labels, second_labels)
    selected = first_labels.ne(-100)
    assert selected.sum().item() == 40
    assert torch.equal(first_labels[selected], tokens[selected])
    assert first_inputs[selected].eq(MASK_TOKEN_ID).sum().item() > 0
    assert torch.equal(first_inputs[~selected], tokens[~selected])


def test_span_masking_rejects_invalid_probability() -> None:
    tokens = tokenize_dna("ACGT").unsqueeze(0)
    try:
        mask_span_mlm(tokens, torch.Generator(), mask_probability=0.0)
    except ValueError as exc:
        assert "probability" in str(exc)
    else:
        raise AssertionError("zero mask probability was accepted")

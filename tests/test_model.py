from __future__ import annotations

import torch

from legumegenomefm.model import LegumeGenomeConfig, LegumeGenomeModel
from legumegenomefm.tokenizer import DNA_VOCAB_SIZE, complement_logits, reverse_complement_tokens, tokenize_dna


def tiny_config() -> LegumeGenomeConfig:
    return LegumeGenomeConfig(
        vocab_size=DNA_VOCAB_SIZE,
        d_model=32,
        n_layers=2,
        ffn_multiple=2,
        kernel_size=3,
        dilations=(1, 2),
        dropout=0.0,
    )


def test_tokenizer_reverse_complement_round_trip() -> None:
    tokens = tokenize_dna("ACGTRYSWKMBDHVN")
    assert torch.equal(reverse_complement_tokens(reverse_complement_tokens(tokens)), tokens)
    assert tokens.min().item() >= 2
    assert tokens.max().item() < DNA_VOCAB_SIZE


def test_model_forward_loss_and_weight_tying() -> None:
    torch.manual_seed(3)
    model = LegumeGenomeModel(tiny_config())
    tokens = tokenize_dna("ACGT" * 16).unsqueeze(0)
    labels = tokens.clone()
    labels[:, ::2] = -100
    output = model(tokens, labels=labels)
    assert output.logits.shape == (1, 64, DNA_VOCAB_SIZE)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_rc_equivariant_logits_are_exact_in_eval_mode() -> None:
    torch.manual_seed(5)
    model = LegumeGenomeModel(tiny_config()).eval()
    tokens = tokenize_dna("ACGTTGCARYSWKMBDHVNACGTACGTACGTA").unsqueeze(0)
    rc_tokens = reverse_complement_tokens(tokens)
    with torch.no_grad():
        logits = model(tokens).logits
        rc_logits = model(rc_tokens).logits
    aligned = complement_logits(torch.flip(rc_logits, dims=(1,)))
    torch.testing.assert_close(logits, aligned, rtol=0, atol=1e-6)


def test_model_accepts_multiple_context_lengths() -> None:
    model = LegumeGenomeModel(tiny_config()).eval()
    with torch.no_grad():
        short = model(tokenize_dna("ACGT" * 8).unsqueeze(0)).logits
        long = model(tokenize_dna("ACGT" * 32).unsqueeze(0)).logits
    assert short.shape[1] == 32
    assert long.shape[1] == 128

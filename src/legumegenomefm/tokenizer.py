from __future__ import annotations

import torch


PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
_SYMBOLS = "ACGTNRYSWKMBDHV"
TOKEN_TO_ID = {symbol: index + 2 for index, symbol in enumerate(_SYMBOLS)}
ID_TO_TOKEN = {value: key for key, value in TOKEN_TO_ID.items()}
DNA_VOCAB_SIZE = len(TOKEN_TO_ID) + 2
_COMPLEMENT_SYMBOL = {
    "A": "T",
    "C": "G",
    "G": "C",
    "T": "A",
    "N": "N",
    "R": "Y",
    "Y": "R",
    "S": "S",
    "W": "W",
    "K": "M",
    "M": "K",
    "B": "V",
    "D": "H",
    "H": "D",
    "V": "B",
}
_COMPLEMENT_IDS = [PAD_TOKEN_ID, MASK_TOKEN_ID] + [
    TOKEN_TO_ID[_COMPLEMENT_SYMBOL[symbol]] for symbol in _SYMBOLS
]


def tokenize_dna(sequence: str) -> torch.Tensor:
    normalized = sequence.upper()
    values = [TOKEN_TO_ID.get(symbol, TOKEN_TO_ID["N"]) for symbol in normalized]
    return torch.tensor(values, dtype=torch.long)


def reverse_complement_tokens(tokens: torch.Tensor) -> torch.Tensor:
    mapping = torch.tensor(_COMPLEMENT_IDS, dtype=torch.long, device=tokens.device)
    return mapping[tokens.flip(-1)]


def complement_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] != DNA_VOCAB_SIZE:
        raise ValueError("logit vocabulary dimension does not match DNA vocabulary")
    mapping = torch.tensor(_COMPLEMENT_IDS, dtype=torch.long, device=logits.device)
    return logits.index_select(-1, mapping)

from __future__ import annotations

import math

import torch

from legumegenomefm.tokenizer import MASK_TOKEN_ID


def mask_span_mlm(
    input_ids: torch.Tensor,
    generator: torch.Generator,
    *,
    mask_probability: float = 0.15,
    mean_span_length: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.dtype != torch.long:
        raise ValueError("input_ids must be a rank-2 torch.long tensor")
    if input_ids.device.type != "cpu":
        raise ValueError("span masking is performed on CPU before device transfer")
    if not 0 < mask_probability < 1:
        raise ValueError("mask_probability must be between zero and one")
    if mean_span_length < 1:
        raise ValueError("mean_span_length must be at least one")
    batch, length = input_ids.shape
    target_count = max(1, math.ceil(length * mask_probability))
    selected = torch.zeros_like(input_ids, dtype=torch.bool)
    geometric_probability = 1.0 / mean_span_length
    logarithm = math.log1p(-geometric_probability) if geometric_probability < 1 else None
    for row in range(batch):
        selected_count = 0
        while selected_count < target_count:
            available = (~selected[row]).nonzero(as_tuple=False).flatten()
            if available.numel() == 0:
                break
            start = int(available[torch.randint(available.numel(), (1,), generator=generator)].item())
            if logarithm is None:
                span_length = 1
            else:
                uniform = max(float(torch.rand((), generator=generator).item()), torch.finfo(torch.float32).tiny)
                span_length = int(math.floor(math.log1p(-uniform) / logarithm)) + 1
            end = min(length, start + span_length)
            indices = torch.arange(start, end)
            indices = indices[~selected[row, indices]]
            remaining = target_count - selected_count
            indices = indices[:remaining]
            selected[row, indices] = True
            selected_count += int(indices.numel())
    labels = input_ids.clone()
    labels[~selected] = -100
    masked_inputs = input_ids.clone()
    replacement_draw = torch.rand(input_ids.shape, generator=generator)
    use_mask = selected & replacement_draw.lt(0.8)
    use_random = selected & replacement_draw.ge(0.8) & replacement_draw.lt(0.9)
    masked_inputs[use_mask] = MASK_TOKEN_ID
    random_bases = torch.randint(2, 6, input_ids.shape, generator=generator)
    masked_inputs[use_random] = random_bases[use_random]
    return masked_inputs, labels

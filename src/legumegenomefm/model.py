from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from legumegenomefm.tokenizer import DNA_VOCAB_SIZE, complement_logits, reverse_complement_tokens


@dataclass(frozen=True)
class LegumeGenomeConfig:
    vocab_size: int = DNA_VOCAB_SIZE
    d_model: int = 640
    n_layers: int = 18
    ffn_multiple: int = 3
    kernel_size: int = 7
    dilations: tuple[int, ...] = (1, 4, 16, 64)
    dropout: float = 0.1
    norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.vocab_size != DNA_VOCAB_SIZE:
            raise ValueError("formal model vocabulary must equal the frozen DNA vocabulary")
        if self.d_model < 8 or self.n_layers < 1 or self.ffn_multiple < 1:
            raise ValueError("invalid model dimensions")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3")
        if not self.dilations or any(value < 1 for value in self.dilations):
            raise ValueError("dilations must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class LegumeGenomeOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        scale = hidden.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return hidden * scale.to(hidden.dtype) * self.weight


class HierarchicalMixer(nn.Module):
    def __init__(self, config: LegumeGenomeConfig) -> None:
        super().__init__()
        width = config.d_model
        self.in_projection = nn.Linear(width, width * 2)
        self.local_convolutions = nn.ModuleList(
            nn.Conv1d(
                width,
                width,
                kernel_size=config.kernel_size,
                padding=dilation * (config.kernel_size - 1) // 2,
                dilation=dilation,
                groups=width,
                bias=False,
            )
            for dilation in config.dilations
        )
        self.branch_weights = nn.Parameter(torch.zeros(len(config.dilations) + 2))
        self.output_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_projection(hidden).chunk(2, dim=-1)
        channels = value.transpose(1, 2)
        branches = [convolution(channels) for convolution in self.local_convolutions]
        local_context = F.avg_pool1d(channels, kernel_size=3, stride=1, padding=1)
        global_context = channels.mean(dim=-1, keepdim=True).expand_as(channels)
        branches.extend((local_context, global_context))
        weights = self.branch_weights.softmax(dim=0)
        mixed = sum(weight * branch for weight, branch in zip(weights, branches))
        mixed = mixed.transpose(1, 2)
        output = F.silu(gate) * mixed
        return self.dropout(self.output_projection(output))


class SwiGLU(nn.Module):
    def __init__(self, config: LegumeGenomeConfig) -> None:
        super().__init__()
        hidden_width = config.d_model * config.ffn_multiple
        self.input_projection = nn.Linear(config.d_model, hidden_width * 2)
        self.output_projection = nn.Linear(hidden_width, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value, gate = self.input_projection(hidden).chunk(2, dim=-1)
        return self.dropout(self.output_projection(value * F.silu(gate)))


class LegumeGenomeBlock(nn.Module):
    def __init__(self, config: LegumeGenomeConfig) -> None:
        super().__init__()
        self.mixer_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mixer = HierarchicalMixer(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.mixer(self.mixer_norm(hidden))
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class LegumeGenomeModel(nn.Module):
    def __init__(self, config: LegumeGenomeConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(LegumeGenomeBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_uniform_(module.weight, a=5**0.5)

    def _forward_single_orientation(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
    ) -> LegumeGenomeOutput:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a rank-2 torch.long tensor")
        rc_input_ids = reverse_complement_tokens(input_ids)
        paired = torch.cat((input_ids, rc_input_ids), dim=0)
        paired_logits = self._forward_single_orientation(paired)
        forward_logits, reverse_logits = paired_logits.chunk(2, dim=0)
        aligned_reverse_logits = complement_logits(reverse_logits.flip(1))
        logits = (forward_logits + aligned_reverse_logits) * 0.5
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return LegumeGenomeOutput(logits=logits, loss=loss)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

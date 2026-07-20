from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class HierMambaConfig:
    vocabulary_size: int = 7
    maximum_context_bp: int = 262_144
    encoder_channels: tuple[int, ...] = (128, 192, 256, 384, 512, 768, 1024, 1024)
    encoder_strides: tuple[int, ...] = (1, 2, 2, 2, 2, 2, 2, 2)
    residual_blocks_per_scale: tuple[int, ...] = (2, 2, 2, 2, 2, 2, 2, 2)
    local_kernel_size: int = 7
    global_layers: int = 24
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 8
    chunk_size: int = 256
    activation_checkpointing: bool = True

    @property
    def latent_stride_bp(self) -> int:
        product = 1
        for stride in self.encoder_strides:
            product *= stride
        return product

    def validate(self) -> None:
        if self.vocabulary_size != 7:
            raise ValueError("the candidate tokenizer vocabulary must contain exactly seven tokens")
        if not (
            len(self.encoder_channels)
            == len(self.encoder_strides)
            == len(self.residual_blocks_per_scale)
        ):
            raise ValueError("encoder channel, stride and block schedules must have the same length")
        if self.encoder_strides != (1, *(2 for _ in self.encoder_strides[1:])):
            raise ValueError("the encoder implementation requires one base scale followed by stride-two transitions")
        if self.latent_stride_bp != 128 or self.maximum_context_bp % self.latent_stride_bp:
            raise ValueError("the model requires a 128-bp latent stride that divides the maximum context")
        if self.maximum_context_bp // self.latent_stride_bp != 2048:
            raise ValueError("the maximum global latent length must be 2048")
        if self.encoder_channels[-1] % self.headdim:
            raise ValueError("global width must be divisible by the Mamba-2 head dimension")
        if any(value < 1 for value in self.residual_blocks_per_scale):
            raise ValueError("every encoder/decoder scale must contain at least one residual block")
        if self.local_kernel_size < 3 or self.local_kernel_size % 2 == 0:
            raise ValueError("the local kernel must be odd and at least three")

    @classmethod
    def from_mapping(cls, model: Mapping[str, Any], activation_checkpointing: bool = True) -> "HierMambaConfig":
        global_core = model["global_core"]
        config = cls(
            vocabulary_size=int(model["vocabulary_size"]),
            maximum_context_bp=int(model["maximum_context_bp"]),
            encoder_channels=tuple(int(value) for value in model["encoder_channels"]),
            encoder_strides=tuple(int(value) for value in model["encoder_strides"]),
            residual_blocks_per_scale=tuple(int(value) for value in model["residual_blocks_per_scale"]),
            local_kernel_size=int(model["local_kernel_size"]),
            global_layers=int(global_core["n_layer"]),
            d_state=int(global_core["d_state"]),
            d_conv=int(global_core["d_conv"]),
            expand=int(global_core["expand"]),
            headdim=int(global_core["headdim"]),
            ngroups=int(global_core["ngroups"]),
            chunk_size=int(global_core["chunk_size"]),
            activation_checkpointing=activation_checkpointing,
        )
        config.validate()
        return config


@dataclass
class MaskedLMOutput:
    logits: Tensor
    loss_sum: Tensor | None
    masked_token_count: Tensor | None


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.RMSNorm(channels)
        self.expand = nn.Linear(channels, channels * 4)
        self.project = nn.Linear(channels * 4, channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x).transpose(1, 2)
        x = self.project(F.gelu(self.expand(self.norm(x)), approximate="tanh"))
        return residual + x.transpose(1, 2)


class LocalScale(nn.Module):
    def __init__(self, channels: int, blocks: int, kernel_size: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ConvNeXt1DBlock(channels, kernel_size) for _ in range(blocks))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def _load_mamba2_class():
    try:
        from mamba_ssm.modules.mamba2 import Mamba2
    except Exception as exc:  # pragma: no cover - exercised on the target H20 environment
        raise RuntimeError(
            "Mamba-2 production backend is unavailable; install the frozen mamba-ssm/causal-conv1d stack "
            "and pass the H20 kernel profile before model instantiation"
        ) from exc
    return Mamba2


class BidirectionalMamba2(nn.Module):
    def __init__(self, config: HierMambaConfig, layer_index: int, mamba2: type[nn.Module]) -> None:
        super().__init__()
        arguments = {
            "d_model": config.encoder_channels[-1],
            "d_state": config.d_state,
            "d_conv": config.d_conv,
            "expand": config.expand,
            "headdim": config.headdim,
            "ngroups": config.ngroups,
            "chunk_size": config.chunk_size,
            "layer_idx": layer_index,
        }
        self.forward_scan = mamba2(**arguments)
        self.reverse_scan = mamba2(**arguments)
        self.reverse_scan.in_proj.weight = self.forward_scan.in_proj.weight
        if self.forward_scan.in_proj.bias is not None:
            self.reverse_scan.in_proj.bias = self.forward_scan.in_proj.bias
        self.reverse_scan.out_proj.weight = self.forward_scan.out_proj.weight
        if self.forward_scan.out_proj.bias is not None:
            self.reverse_scan.out_proj.bias = self.forward_scan.out_proj.bias

    def forward(self, x: Tensor) -> Tensor:
        forward = self.forward_scan(x)
        reverse = self.reverse_scan(x.flip(1)).flip(1)
        return forward + reverse


class GlobalMambaBlock(nn.Module):
    def __init__(self, config: HierMambaConfig, layer_index: int, mamba2: type[nn.Module]) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(config.encoder_channels[-1])
        self.mixer = BidirectionalMamba2(config, layer_index, mamba2)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mixer(self.norm(x))


class HierMambaBackbone(nn.Module):
    def __init__(self, config: HierMambaConfig) -> None:
        super().__init__()
        config.validate()
        mamba2 = _load_mamba2_class()
        self.config = config
        channels = config.encoder_channels
        self.embedding = nn.Embedding(config.vocabulary_size, channels[0])
        self.encoder_scales = nn.ModuleList(
            LocalScale(channel, blocks, config.local_kernel_size)
            for channel, blocks in zip(channels, config.residual_blocks_per_scale)
        )
        self.downsamples = nn.ModuleList(
            nn.Conv1d(source, target, kernel_size=4, stride=2, padding=1)
            for source, target in zip(channels[:-1], channels[1:])
        )
        self.global_blocks = nn.ModuleList(
            GlobalMambaBlock(config, layer_index, mamba2) for layer_index in range(config.global_layers)
        )
        reversed_pairs = list(zip(reversed(channels[1:]), reversed(channels[:-1])))
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose1d(source, target, kernel_size=4, stride=2, padding=1)
            for source, target in reversed_pairs
        )
        decoder_blocks = list(reversed(config.residual_blocks_per_scale[:-1]))
        self.skip_fusions = nn.ModuleList(
            nn.Conv1d(target * 2, target, kernel_size=1) for _, target in reversed_pairs
        )
        self.decoder_scales = nn.ModuleList(
            LocalScale(target, blocks, config.local_kernel_size)
            for (_, target), blocks in zip(reversed_pairs, decoder_blocks)
        )
        self.final_norm = nn.RMSNorm(channels[0])

    def _apply_global(self, block: nn.Module, x: Tensor) -> Tensor:
        if self.config.activation_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, length]")
        length = int(input_ids.shape[1])
        if length < self.config.latent_stride_bp or length > self.config.maximum_context_bp:
            raise ValueError("input length is outside the formal architecture bounds")
        if length % self.config.latent_stride_bp:
            raise ValueError("input length must be divisible by the 128-bp latent stride")
        x = self.embedding(input_ids).transpose(1, 2)
        skips: list[Tensor] = []
        for index, scale in enumerate(self.encoder_scales):
            if index:
                x = self.downsamples[index - 1](x)
            x = scale(x)
            skips.append(x)
        x = x.transpose(1, 2)
        for block in self.global_blocks:
            x = self._apply_global(block, x)
        x = x.transpose(1, 2) + skips[-1]
        for upsample, fusion, scale, skip in zip(
            self.upsamples,
            self.skip_fusions,
            self.decoder_scales,
            reversed(skips[:-1]),
        ):
            x = upsample(x)
            if x.shape[-1] != skip.shape[-1]:
                raise RuntimeError("encoder and decoder sequence lengths do not close exactly")
            x = scale(fusion(torch.cat((x, skip), dim=1)))
        return self.final_norm(x.transpose(1, 2))


def reverse_complement_token_ids(input_ids: Tensor, complement_map: Tensor) -> Tensor:
    if input_ids.dtype != torch.long:
        raise ValueError("input IDs must use torch.long")
    if input_ids.numel() and bool((input_ids.min() < 0) | (input_ids.max() >= complement_map.numel())):
        raise ValueError("input IDs are outside the tokenizer vocabulary")
    return complement_map[input_ids.flip(1)]


class HierMambaForMaskedLM(nn.Module):
    def __init__(self, config: HierMambaConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = HierMambaBackbone(config)
        self.lm_head = nn.Linear(config.encoder_channels[0], config.vocabulary_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.weight
        self.register_buffer(
            "complement_map",
            torch.tensor([0, 1, 5, 4, 3, 2, 6], dtype=torch.long),
            persistent=True,
        )

    def reverse_complement(self, input_ids: Tensor) -> Tensor:
        return reverse_complement_token_ids(input_ids, self.complement_map)

    def _logits(self, input_ids: Tensor) -> Tensor:
        return self.lm_head(self.backbone(input_ids))

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> MaskedLMOutput:
        forward_logits = self._logits(input_ids)
        rc_logits = self._logits(self.reverse_complement(input_ids))
        aligned_rc_logits = rc_logits.flip(1).index_select(-1, self.complement_map)
        logits = (forward_logits + aligned_rc_logits) * 0.5
        if labels is None:
            return MaskedLMOutput(logits=logits, loss_sum=None, masked_token_count=None)
        if labels.shape != input_ids.shape:
            raise ValueError("labels and input_ids must have the same shape")
        flat_labels = labels.reshape(-1)
        valid = flat_labels.ne(-100)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, self.config.vocabulary_size),
            flat_labels,
            ignore_index=-100,
            reduction="sum",
        )
        return MaskedLMOutput(logits=logits, loss_sum=loss_sum, masked_token_count=valid.sum())

    def unique_trainable_parameter_count(self) -> int:
        seen: set[int] = set()
        total = 0
        for parameter in self.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel()
        return total

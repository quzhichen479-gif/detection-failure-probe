from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .common import ConvBNAct, LayerNorm2d, validate_nchw


class BoltzmannSparseAttention(nn.Module):
    """BoltzFormer-inspired dynamic sparse attention for small-object probes.

    A learned saliency distribution is temperature-scaled, then the highest-probability
    tokens are used as sparse key/value support. During training, Gumbel perturbation can
    approximate stochastic Boltzmann sampling while keeping a tensor-friendly top-k path.

    This is an engineering adapter, not the complete BoltzFormer sampling hierarchy.
    """

    def __init__(
        self,
        channels: int,
        sample_ratio: float = 0.125,
        temperature: float = 1.0,
        stochastic_training: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.channels = channels
        self.sample_ratio = sample_ratio
        self.temperature = temperature
        self.stochastic_training = stochastic_training
        self.norm = LayerNorm2d(channels)
        self.score = nn.Conv2d(channels, 1, 1)
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.kv = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.proj = ConvBNAct(channels, channels, 1, activation=False)

    def set_temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(value)

    def forward(self, x: Tensor) -> Tensor:
        validate_nchw(x, self.channels)
        z = self.norm(x)
        b, c, h, w = z.shape
        n = h * w
        k_count = max(1, min(n, int(math.ceil(n * self.sample_ratio))))

        logits = self.score(z).flatten(2).squeeze(1) / self.temperature
        if self.training and self.stochastic_training:
            u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
            logits = logits - torch.log(-torch.log(u))
        indices = logits.topk(k_count, dim=-1).indices

        q = self.q(z).flatten(2).transpose(1, 2)
        k_map, v_map = self.kv(z).chunk(2, dim=1)
        k_all = k_map.flatten(2).transpose(1, 2)
        v_all = v_map.flatten(2).transpose(1, 2)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, c)
        k = torch.gather(k_all, 1, gather_idx)
        v = torch.gather(v_all, 1, gather_idx)
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(c), dim=-1)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        out = out + self.local(z)
        return x + self.proj(out)

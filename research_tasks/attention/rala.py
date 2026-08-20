from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import ConvBNAct, LayerNorm2d, validate_nchw


class RALAAttention(nn.Module):
    """Rank-augmented linear attention research adapter for NCHW features.

    This implementation captures the paper's two central rank-augmentation ideas:
    token-dependent weighting of the KV buffer and token-wise output modulation. It is
    intentionally small enough for YOLO P3/P4 experiments and is not a verbatim port of RAVLT.
    """

    def __init__(self, channels: int, expansion: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        self.inner = max(16, int(channels * expansion))
        self.eps = eps
        self.norm = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, 3 * self.inner, 1, bias=False)
        self.kv_weight = nn.Sequential(
            nn.Conv2d(channels, self.inner, 1, bias=False),
            nn.Sigmoid(),
        )
        self.modulation = nn.Sequential(
            nn.Conv2d(channels, self.inner, 1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Conv2d(self.inner, self.inner, 3, padding=1, groups=self.inner, bias=False)
        self.proj = ConvBNAct(self.inner, channels, 1, activation=False)

    @staticmethod
    def _positive_kernel(x: Tensor) -> Tensor:
        return F.elu(x, alpha=1.0) + 1.0

    def forward(self, x: Tensor) -> Tensor:
        validate_nchw(x, self.channels)
        z = self.norm(x)
        q, k, v = self.qkv(z).chunk(3, dim=1)
        b, c, h, w = q.shape
        n = h * w
        q = self._positive_kernel(q).flatten(2).transpose(1, 2)
        k = self._positive_kernel(k).flatten(2).transpose(1, 2)
        v_tokens = v.flatten(2).transpose(1, 2)

        # Rank augmentation 1: token-dependent contribution to the KV buffer.
        alpha = self.kv_weight(z).mean(dim=1, keepdim=True).flatten(2).transpose(1, 2)
        weighted_v = v_tokens * alpha
        kv = torch.bmm(k.transpose(1, 2), weighted_v) / float(n)
        k_sum = k.sum(dim=1)
        denom = torch.bmm(q, k_sum.unsqueeze(-1)).clamp_min(self.eps)
        out = torch.bmm(q, kv) / denom

        # Rank augmentation 2: token-wise multiplicative modulation.
        mod = self.modulation(z).flatten(2).transpose(1, 2)
        out = out * (1.0 + torch.tanh(mod))
        out = out.transpose(1, 2).reshape(b, c, h, w)
        out = out + self.local(v)
        return x + self.proj(out)

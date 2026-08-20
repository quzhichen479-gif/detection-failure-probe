from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .common import ConvBNAct, LayerNorm2d, validate_nchw


class SHSAAttention(nn.Module):
    """Single-head self-attention on a partial channel slice, SHViT-inspired.

    Only ``attn_ratio`` of channels enters global attention. The bypass branch preserves
    local information and limits quadratic attention cost at YOLO neck resolutions.
    """

    def __init__(self, channels: int, attn_ratio: float = 0.25, qk_dim: int | None = None) -> None:
        super().__init__()
        if not 0.0 < attn_ratio <= 1.0:
            raise ValueError("attn_ratio must be in (0, 1]")
        self.channels = channels
        attn_channels = max(8, int(round(channels * attn_ratio)))
        attn_channels = min(attn_channels, channels)
        self.attn_channels = attn_channels
        self.bypass_channels = channels - attn_channels
        self.qk_dim = qk_dim or max(8, attn_channels // 2)
        self.norm = LayerNorm2d(attn_channels)
        self.qkv = nn.Conv2d(attn_channels, 2 * self.qk_dim + attn_channels, 1, bias=False)
        self.local = nn.Conv2d(
            attn_channels, attn_channels, 3, padding=1, groups=attn_channels, bias=False
        )
        self.proj = ConvBNAct(channels, channels, 1, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        validate_nchw(x, self.channels)
        bypass = x[:, : self.bypass_channels] if self.bypass_channels else None
        a = x[:, self.bypass_channels :]
        a_norm = self.norm(a)
        q, k, v = torch.split(
            self.qkv(a_norm), [self.qk_dim, self.qk_dim, self.attn_channels], dim=1
        )
        b, _, h, w = q.shape
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v_flat = v.flatten(2).transpose(1, 2)
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(self.qk_dim), dim=-1)
        global_out = torch.bmm(attn, v_flat).transpose(1, 2).reshape(b, self.attn_channels, h, w)
        a_out = global_out + self.local(v)
        merged = torch.cat([bypass, a_out], dim=1) if bypass is not None else a_out
        return x + self.proj(merged)

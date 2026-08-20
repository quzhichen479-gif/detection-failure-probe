from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .common import ConvBNAct, LayerNorm2d, validate_nchw


class BiLevelRoutingAttention(nn.Module):
    """Compact BiFormer-style bi-level routing attention for NCHW feature maps.

    Feature maps are pooled into coarse routing regions. Each query region selects top-k
    candidate regions; fine token attention is then restricted to their union. The loop over
    query regions favors clarity and research correctness over kernel-level optimization.
    """

    def __init__(self, channels: int, region_size: int = 4, topk: int = 4) -> None:
        super().__init__()
        if region_size < 1 or topk < 1:
            raise ValueError("region_size and topk must be >= 1")
        self.channels = channels
        self.region_size = region_size
        self.topk = topk
        self.norm = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.proj = ConvBNAct(channels, channels, 1, activation=False)

    def forward(self, x: Tensor) -> Tensor:
        validate_nchw(x, self.channels)
        z = self.norm(x)
        b, c, h, w = z.shape
        r = self.region_size
        hp = math.ceil(h / r) * r
        wp = math.ceil(w / r) * r
        pad = (0, wp - w, 0, hp - h)
        z_pad = F.pad(z, pad)
        q_map, k_map, v_map = self.qkv(z_pad).chunk(3, dim=1)
        gh, gw = hp // r, wp // r
        regions = gh * gw

        def to_regions(t: Tensor) -> Tensor:
            t = t.reshape(b, c, gh, r, gw, r).permute(0, 2, 4, 3, 5, 1)
            return t.reshape(b, regions, r * r, c)

        q_reg, k_reg, v_reg = map(to_regions, (q_map, k_map, v_map))
        q_desc = q_reg.mean(dim=2)
        k_desc = k_reg.mean(dim=2)
        route = torch.bmm(q_desc, k_desc.transpose(1, 2)) / math.sqrt(c)
        route_idx = route.topk(min(self.topk, regions), dim=-1).indices

        outputs: list[Tensor] = []
        for qi in range(regions):
            idx = route_idx[:, qi]
            idx_expand = idx[:, :, None, None].expand(-1, -1, r * r, c)
            k_candidates = torch.gather(k_reg, 1, idx_expand).reshape(b, -1, c)
            v_candidates = torch.gather(v_reg, 1, idx_expand).reshape(b, -1, c)
            q_tokens = q_reg[:, qi]
            attn = torch.softmax(
                torch.bmm(q_tokens, k_candidates.transpose(1, 2)) / math.sqrt(c), dim=-1
            )
            outputs.append(torch.bmm(attn, v_candidates))

        out = torch.stack(outputs, dim=1)
        out = out.reshape(b, gh, gw, r, r, c).permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(b, c, hp, wp)[..., :h, :w]
        return x + self.proj(out)

from __future__ import annotations

from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW feature maps."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if activation:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


def validate_nchw(x: Tensor, channels: int) -> None:
    if x.ndim != 4:
        raise ValueError(f"expected NCHW tensor, got shape={tuple(x.shape)}")
    if x.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, got {x.shape[1]}")

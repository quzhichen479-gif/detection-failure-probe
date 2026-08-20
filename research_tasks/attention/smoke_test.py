from __future__ import annotations

import torch

from research_tasks.attention import ATTENTION_TASKS, build_attention


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 64, 16, 16)
    for name in ATTENTION_TASKS:
        kwargs: dict[str, object] = {}
        if name == "bra":
            kwargs = {"region_size": 4, "topk": 2}
        if name == "boltzmann":
            kwargs = {"sample_ratio": 0.125, "stochastic_training": False}
        module = build_attention(name, channels=64, **kwargs).eval()
        with torch.no_grad():
            y = module(x)
        assert y.shape == x.shape, (name, x.shape, y.shape)
        assert torch.isfinite(y).all(), name
        params = sum(parameter.numel() for parameter in module.parameters())
        print(f"{name:10s} OK shape={tuple(y.shape)} params={params:,}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from torch import nn

from large_kernel_context import CEPConvLKC, StripLKC, UniRepLKControl


_VARIANTS = {
    "unirep": UniRepLKControl,
    "strip": StripLKC,
    "cepconv": CEPConvLKC,
}


def build_p3_context(variant: str, channels: int) -> nn.Module:
    """Build one frozen Phase-5 context block for the P3 classification input."""
    key = str(variant).strip().lower()
    if key not in _VARIANTS:
        raise ValueError(f"unknown Phase-5 variant {variant!r}; expected one of {tuple(_VARIANTS)}")
    if key == "unirep":
        return UniRepLKControl(channels, kernel_size=17, gamma_init=0.0)
    if key == "strip":
        return StripLKC(channels, strip_kernel=17, gamma_init=0.0)
    return CEPConvLKC(channels, kernel_size=17, center_kernel=5, gamma_init=0.0)


class ContextBeforeTower(nn.Module):
    """Apply a context adapter before an existing classification tower.

    Wrapping ``Detect.cv3[0]`` with this module changes only the P3 classification
    path: box regression still receives the untouched P3 tensor and P4/P5 are not
    modified. This is the preferred integration pattern for the frozen YOLO11
    experiment because it avoids copying or overriding Detect.forward().
    """

    def __init__(self, context: nn.Module, tower: nn.Module) -> None:
        super().__init__()
        self.context = context
        self.tower = tower

    def forward(self, x):
        return self.tower(self.context(x))


def wrap_detect_p3_classification(detect: nn.Module, variant: str, channels: int) -> nn.Module:
    """Wrap ``detect.cv3[0]`` in place and return the detect module.

    This helper deliberately depends only on the public structural contract used by
    YOLO11 Detect heads (``cv3`` is the classification tower ModuleList). The local
    Ultralytics port should still validate the exact 8.4.113 source before use.
    """
    if not hasattr(detect, "cv3"):
        raise TypeError("detect module has no cv3 classification towers")
    if len(detect.cv3) < 1:
        raise ValueError("detect.cv3 must contain the P3 classification tower")
    if isinstance(detect.cv3[0], ContextBeforeTower):
        raise RuntimeError("P3 classification tower is already context-wrapped")
    detect.cv3[0] = ContextBeforeTower(build_p3_context(variant, channels), detect.cv3[0])
    return detect


def switch_phase5_to_deploy(module: nn.Module) -> nn.Module:
    """Materialize re-parameterizable Phase-5 operators before export."""
    for child in list(module.modules()):
        if isinstance(child, (UniRepLKControl, CEPConvLKC)):
            child.switch_to_deploy()
    return module


__all__ = [
    "build_p3_context",
    "ContextBeforeTower",
    "wrap_detect_p3_classification",
    "switch_phase5_to_deploy",
]

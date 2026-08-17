from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from werg_core import WERGConfig, WERGReference


@dataclass(frozen=True)
class WERGAdapterOutput:
    logits: Tensor
    z_w: Tensor
    z_g: Tensor
    delta_logit: Tensor


class P3WERGClassificationAdapter(nn.Module):
    """Thin adapter to be called only on the accepted P3 classification branch.

    The actual Ultralytics integration must pass the accepted DBRA P3 classification feature,
    the pre-sigmoid P3 class logits, and the same normalized RGB tensor that enters the detector.
    P4/P5 and every regression path are intentionally untouched.
    """

    def __init__(self, p3_channels: int, config: WERGConfig | None = None) -> None:
        super().__init__()
        self.werg = WERGReference(p3_channels, config)

    def forward(self, p3_cls_feature: Tensor, p3_cls_logits: Tensor, rgb: Tensor) -> WERGAdapterOutput:
        logits, evidence = self.werg(p3_cls_feature, p3_cls_logits, rgb, return_aux=False)
        return WERGAdapterOutput(
            logits=logits,
            z_w=evidence["z_w"],
            z_g=evidence["z_g"],
            delta_logit=evidence["delta_logit"],
        )

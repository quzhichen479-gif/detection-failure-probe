"""Reference-only YOLO adapter for Round-4 FreqFusion + DBRA.

This file is project-authored reference code for Codex. It is NOT part of the
`detection-failure-probe` package and does not copy the upstream FreqFusion implementation.
Port it into the actual YOLO11 engineering repository after verifying upstream provenance.

Pinned upstream
---------------
repo:   https://github.com/Linwei-Chen/FreqFusion
commit: 3fb0c70637a3c194fb74294d3ce4681958b26241
file:   FreqFusion.py
blob:   b8fa94d418c3094a8d6653712b65037f70daccec

Primary Round-4 profile follows the official MMDetection Faster R-CNN/COCO FreqFusion
configuration rather than the root clean-class constructor default:
- use_high_pass=True
- use_low_pass=True
- lowpass_kernel=5
- highpass_kernel=3
- compress_ratio=8
- feature_resample=True
- semi_conv=True
- feature_resample_group=4

YOLO-specific adaptation
------------------------
Official FreqFusionCARAFEFPN refines the two branches and adds them because standard FPN
uses additive lateral fusion. YOLO11 uses concat at the final top-down P4->P3 fusion.
This wrapper therefore returns cat([hr_refined, lr_reconstructed], dim=1) so FreqFusion
replaces the stock `Upsample + Concat` pair while preserving YOLO's fusion topology.

DBRA is intentionally absent from this file. Reuse the accepted project DBRA P3-Cls-Mid
implementation/configuration unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

try:
    # Suggested path in the actual YOLO repository. Adapt only the import path if needed.
    from .third_party.freqfusion.freqfusion_upstream import FreqFusion
except Exception:  # pragma: no cover - reference repository may not vendor upstream
    FreqFusion = None


class FreqFusionConcat(nn.Module):
    """Two-input YOLO wrapper around the pinned upstream FreqFusion operator.

    Input order is fixed:
        x[0]: HR low-level feature  [B, C_hr, H, W]      (backbone P3)
        x[1]: LR high-level feature [B, C_lr, H/2, W/2]  (fused P4)

    Output:
        cat([hr_refined, lr_reconstructed], dim=1)
        -> [B, C_hr + C_lr, H, W]

    The default profile mirrors the official Faster R-CNN/COCO configuration. Set
    `feature_resample=False` only as a registered core-only attribution control.
    """

    def __init__(
        self,
        hr_channels: int,
        lr_channels: int,
        compress_ratio: int = 8,
        compressed_channels: int | None = None,
        lowpass_kernel: int = 5,
        highpass_kernel: int = 3,
        up_group: int = 1,
        encoder_kernel: int = 3,
        encoder_dilation: int = 1,
        feature_resample: bool = True,
        feature_resample_group: int = 4,
        comp_feat_upsample: bool = True,
        use_high_pass: bool = True,
        use_low_pass: bool = True,
        hr_residual: bool = True,
        semi_conv: bool = True,
        hamming_window: bool = True,
        feature_resample_norm: bool = True,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        if FreqFusion is None:
            raise ImportError(
                "Pinned FreqFusion upstream implementation is unavailable. Verify source/license "
                "and expose the upstream class before constructing FreqFusionConcat."
            )

        if hr_channels <= 0 or lr_channels <= 0:
            raise ValueError("hr_channels and lr_channels must be positive")
        if compress_ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        if lowpass_kernel <= 0 or lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel must be a positive odd integer")
        if highpass_kernel <= 0 or highpass_kernel % 2 == 0:
            raise ValueError("highpass_kernel must be a positive odd integer")
        if feature_resample_group <= 0:
            raise ValueError("feature_resample_group must be positive")

        self.hr_channels = int(hr_channels)
        self.lr_channels = int(lr_channels)
        self.out_channels = self.hr_channels + self.lr_channels
        self.compress_ratio = int(compress_ratio)
        self.use_checkpoint = bool(use_checkpoint)

        if compressed_channels is None:
            compressed_channels = self.out_channels // self.compress_ratio
        compressed_channels = int(compressed_channels)
        if compressed_channels <= 0:
            raise ValueError("derived compressed_channels must be positive")

        # Upstream LocalSimGuidedSampler requires channels divisible by its groups.
        if feature_resample and compressed_channels % feature_resample_group != 0:
            raise ValueError(
                f"compressed_channels={compressed_channels} must be divisible by "
                f"feature_resample_group={feature_resample_group}"
            )

        # With feature_resample_norm=True upstream uses GroupNorm(C//8, C).
        # For the official detection-style channel widths, requiring multiples of 8 keeps
        # that normalization valid and makes incompatibility fail early.
        if feature_resample and feature_resample_norm:
            if compressed_channels < 8 or compressed_channels % 8 != 0:
                raise ValueError(
                    "feature_resample_norm=True requires detection-profile compressed_channels "
                    "to be a multiple of 8 and >= 8 for the pinned upstream GroupNorm path; "
                    f"got {compressed_channels}"
                )

        self.compressed_channels = compressed_channels
        self.feature_resample = bool(feature_resample)

        self.freqfusion = FreqFusion(
            hr_channels=self.hr_channels,
            lr_channels=self.lr_channels,
            # Preserve pinned upstream API semantics. Spatial 2x relation is validated below.
            scale_factor=1,
            lowpass_kernel=int(lowpass_kernel),
            highpass_kernel=int(highpass_kernel),
            up_group=int(up_group),
            encoder_kernel=int(encoder_kernel),
            encoder_dilation=int(encoder_dilation),
            compressed_channels=self.compressed_channels,
            align_corners=False,
            upsample_mode="nearest",
            feature_resample=self.feature_resample,
            feature_resample_group=int(feature_resample_group),
            comp_feat_upsample=bool(comp_feat_upsample),
            use_high_pass=bool(use_high_pass),
            use_low_pass=bool(use_low_pass),
            hr_residual=bool(hr_residual),
            semi_conv=bool(semi_conv),
            hamming_window=bool(hamming_window),
            feature_resample_norm=bool(feature_resample_norm),
        )

    @staticmethod
    def _validate_inputs(x: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("FreqFusionConcat expects [hr_feature, lr_feature]")

        hr_feat, lr_feat = x
        if not isinstance(hr_feat, torch.Tensor) or not isinstance(lr_feat, torch.Tensor):
            raise TypeError("FreqFusionConcat inputs must be torch.Tensor objects")
        if hr_feat.ndim != 4 or lr_feat.ndim != 4:
            raise ValueError(
                f"expected NCHW tensors, got {tuple(hr_feat.shape)} and {tuple(lr_feat.shape)}"
            )
        if hr_feat.shape[0] != lr_feat.shape[0]:
            raise ValueError("HR/LR batch sizes differ")

        hr_h, hr_w = hr_feat.shape[-2:]
        lr_h, lr_w = lr_feat.shape[-2:]
        if hr_h != 2 * lr_h or hr_w != 2 * lr_w:
            raise ValueError(
                "FreqFusion P4->P3 integration requires an exact 2:1 spatial ratio; "
                f"got HR={hr_h}x{hr_w}, LR={lr_h}x{lr_w}"
            )
        return hr_feat, lr_feat

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        hr_feat, lr_feat = self._validate_inputs(x)

        if hr_feat.shape[1] != self.hr_channels:
            raise ValueError(f"expected HR C={self.hr_channels}, got {hr_feat.shape[1]}")
        if lr_feat.shape[1] != self.lr_channels:
            raise ValueError(f"expected LR C={self.lr_channels}, got {lr_feat.shape[1]}")

        _, hr_refined, lr_reconstructed = self.freqfusion(
            hr_feat=hr_feat,
            lr_feat=lr_feat,
            use_checkpoint=self.use_checkpoint,
        )

        target_hw = hr_feat.shape[-2:]
        if hr_refined.shape[-2:] != target_hw:
            raise RuntimeError(
                f"FreqFusion changed HR spatial size: {hr_feat.shape[-2:]} -> "
                f"{hr_refined.shape[-2:]}"
            )
        if lr_reconstructed.shape[-2:] != target_hw:
            raise RuntimeError(
                "FreqFusion LR branch was not reconstructed to HR/P3 resolution: "
                f"got {lr_reconstructed.shape[-2:]}, expected {target_hw}"
            )
        if hr_refined.shape[1] != self.hr_channels:
            raise RuntimeError("FreqFusion changed HR channel count")
        if lr_reconstructed.shape[1] != self.lr_channels:
            raise RuntimeError("FreqFusion changed LR channel count")

        return torch.cat((hr_refined, lr_reconstructed), dim=1)


class FreqFusionConcatDebug(FreqFusionConcat):
    """Offline diagnostic variant; do not put this class in the production YAML graph."""

    def forward_debug(self, x: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        hr_feat, lr_feat = self._validate_inputs(x)
        mask_lr, hr_refined, lr_reconstructed = self.freqfusion(
            hr_feat=hr_feat,
            lr_feat=lr_feat,
            use_checkpoint=self.use_checkpoint,
        )
        return {
            "mask_lr": mask_lr,
            "hr_input": hr_feat,
            "lr_input": lr_feat,
            "hr_refined": hr_refined,
            "lr_reconstructed": lr_reconstructed,
            "fused": torch.cat((hr_refined, lr_reconstructed), dim=1),
        }


def infer_freqfusion_parse_args(
    channel_table: Sequence[int],
    from_indices: Sequence[int],
    yaml_args: Sequence[object],
) -> tuple[list[object], int]:
    """Reference equivalent of the required `parse_model()` bookkeeping."""

    if not isinstance(from_indices, (list, tuple)) or len(from_indices) != 2:
        raise ValueError("FreqFusionConcat requires exactly two source indices")

    hr_idx, lr_idx = int(from_indices[0]), int(from_indices[1])
    hr_c, lr_c = int(channel_table[hr_idx]), int(channel_table[lr_idx])
    return [hr_c, lr_c, *list(yaml_args)], hr_c + lr_c


__all__ = [
    "FreqFusionConcat",
    "FreqFusionConcatDebug",
    "infer_freqfusion_parse_args",
]

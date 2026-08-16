"""Reference-only YOLO adapter for Round-4 FreqFusion + DBRA.

This file belongs to the research/specification repository. It is NOT imported by the
`detection-failure-probe` package. Codex should port the selected code into the actual
Ultralytics/YOLO11 engineering repository.

Design constraints
------------------
1. FreqFusion is a two-input feature-fusion operator, not a one-input replacement for
   `nn.Upsample`.
2. Input order is fixed: [high-resolution low-level feature, low-resolution high-level
   feature]. For the primary YOLO11 integration that is [backbone P3, fused P4].
3. The wrapper returns the concatenation of the refined high-resolution feature and the
   reconstructed low-resolution feature so its output matches the role of stock
   `Upsample -> Concat`.
4. DBRA is NOT implemented here. Round-4 must reuse the project's already-validated
   DBRA P3-Cls-Mid implementation unchanged.
5. Upstream FreqFusion source must be pinned and provenance recorded before use.

Pinned upstream specification
-----------------------------
repo:   https://github.com/Linwei-Chen/FreqFusion
commit: 3fb0c70637a3c194fb74294d3ce4681958b26241
file:   FreqFusion.py
blob:   b8fa94d418c3094a8d6653712b65037f70daccec

The upstream root did not expose a simple LICENSE file when this specification was
prepared. Verify redistribution terms before committing upstream source into another
repository. This adapter itself is project-authored code and deliberately does not copy
the upstream FreqFusion implementation.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

try:
    # Recommended destination in the real YOLO engineering repository.
    # Codex may adapt this import path to the actual local third_party layout.
    from .third_party.freqfusion.freqfusion_upstream import FreqFusion
except Exception:  # pragma: no cover - reference file may live without vendored upstream
    FreqFusion = None


class FreqFusionConcat(nn.Module):
    """YOLO-friendly two-input wrapper around the pinned upstream FreqFusion.

    The module replaces the stock `nearest x2 upsample -> concat with backbone P3`
    pair at the final top-down P4->P3 fusion.

    Inputs
    ------
    x[0]: high-resolution feature, shape [B, C_hr, H, W]
    x[1]: low-resolution feature,  shape [B, C_lr, H/2, W/2]

    Output
    ------
    cat([hr_refined, lr_reconstructed], dim=1), shape
    [B, C_hr + C_lr, H, W]

    The primary Round-4 profile intentionally follows the official clean-code defaults,
    including `feature_resample=False`. Offset-guided resampling is reserved for a
    separately registered follow-up experiment.
    """

    def __init__(
        self,
        hr_channels: int,
        lr_channels: int,
        compressed_channels: int = 64,
        lowpass_kernel: int = 5,
        highpass_kernel: int = 3,
        up_group: int = 1,
        encoder_kernel: int = 3,
        encoder_dilation: int = 1,
        feature_resample: bool = False,
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
                "Pinned FreqFusion upstream implementation is not available. "
                "Vendor or otherwise expose the verified upstream class before "
                "constructing FreqFusionConcat."
            )

        if hr_channels <= 0 or lr_channels <= 0:
            raise ValueError("hr_channels and lr_channels must be positive")
        if compressed_channels <= 0:
            raise ValueError("compressed_channels must be positive")
        if lowpass_kernel <= 0 or lowpass_kernel % 2 == 0:
            raise ValueError("lowpass_kernel must be a positive odd integer")
        if highpass_kernel <= 0 or highpass_kernel % 2 == 0:
            raise ValueError("highpass_kernel must be a positive odd integer")

        self.hr_channels = int(hr_channels)
        self.lr_channels = int(lr_channels)
        self.out_channels = self.hr_channels + self.lr_channels
        self.use_checkpoint = bool(use_checkpoint)

        self.freqfusion = FreqFusion(
            hr_channels=self.hr_channels,
            lr_channels=self.lr_channels,
            # Upstream clean usage assumes a 2:1 spatial ratio while its public
            # constructor default for scale_factor is 1. Preserve the pinned API.
            scale_factor=1,
            lowpass_kernel=int(lowpass_kernel),
            highpass_kernel=int(highpass_kernel),
            up_group=int(up_group),
            encoder_kernel=int(encoder_kernel),
            encoder_dilation=int(encoder_dilation),
            compressed_channels=int(compressed_channels),
            align_corners=False,
            upsample_mode="nearest",
            feature_resample=bool(feature_resample),
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
            raise TypeError(
                "FreqFusionConcat expects exactly two tensors: "
                "[hr_feature, lr_feature]"
            )

        hr_feat, lr_feat = x
        if not isinstance(hr_feat, torch.Tensor) or not isinstance(lr_feat, torch.Tensor):
            raise TypeError("FreqFusionConcat inputs must be torch.Tensor objects")
        if hr_feat.ndim != 4 or lr_feat.ndim != 4:
            raise ValueError(
                f"FreqFusionConcat expects NCHW 4D tensors, got "
                f"{tuple(hr_feat.shape)} and {tuple(lr_feat.shape)}"
            )
        if hr_feat.shape[0] != lr_feat.shape[0]:
            raise ValueError("hr/lr batch sizes differ")

        hr_h, hr_w = hr_feat.shape[-2:]
        lr_h, lr_w = lr_feat.shape[-2:]
        if hr_h != 2 * lr_h or hr_w != 2 * lr_w:
            raise ValueError(
                "Pinned FreqFusion integration expects exact 2:1 spatial ratio; "
                f"got hr={hr_h}x{hr_w}, lr={lr_h}x{lr_w}"
            )

        return hr_feat, lr_feat

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        hr_feat, lr_feat = self._validate_inputs(x)

        if hr_feat.shape[1] != self.hr_channels:
            raise ValueError(
                f"expected hr C={self.hr_channels}, got C={hr_feat.shape[1]}"
            )
        if lr_feat.shape[1] != self.lr_channels:
            raise ValueError(
                f"expected lr C={self.lr_channels}, got C={lr_feat.shape[1]}"
            )

        _, hr_refined, lr_reconstructed = self.freqfusion(
            hr_feat=hr_feat,
            lr_feat=lr_feat,
            use_checkpoint=self.use_checkpoint,
        )

        if hr_refined.shape[-2:] != hr_feat.shape[-2:]:
            raise RuntimeError(
                "FreqFusion changed high-resolution spatial shape unexpectedly: "
                f"{tuple(hr_feat.shape)} -> {tuple(hr_refined.shape)}"
            )
        if lr_reconstructed.shape[-2:] != hr_feat.shape[-2:]:
            raise RuntimeError(
                "FreqFusion low-resolution branch was not reconstructed to the P3 "
                f"resolution: got {tuple(lr_reconstructed.shape)} vs "
                f"target {tuple(hr_feat.shape)}"
            )
        if hr_refined.shape[1] != self.hr_channels:
            raise RuntimeError("FreqFusion changed hr channel count")
        if lr_reconstructed.shape[1] != self.lr_channels:
            raise RuntimeError("FreqFusion changed lr channel count")

        return torch.cat((hr_refined, lr_reconstructed), dim=1)


class FreqFusionConcatDebug(FreqFusionConcat):
    """Diagnostic variant returning intermediate features in addition to fused output.

    Do not use this class inside the production YAML graph. It exists for offline feature
    diagnostics on frozen validation images.
    """

    def forward_debug(
        self, x: Sequence[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        hr_feat, lr_feat = self._validate_inputs(x)
        mask_lr, hr_refined, lr_reconstructed = self.freqfusion(
            hr_feat=hr_feat,
            lr_feat=lr_feat,
            use_checkpoint=self.use_checkpoint,
        )
        fused = torch.cat((hr_refined, lr_reconstructed), dim=1)
        return {
            "mask_lr": mask_lr,
            "hr_input": hr_feat,
            "lr_input": lr_feat,
            "hr_refined": hr_refined,
            "lr_reconstructed": lr_reconstructed,
            "fused": fused,
        }


def infer_freqfusion_parse_args(
    channel_table: Sequence[int],
    from_indices: Sequence[int],
    yaml_args: Sequence[object],
) -> tuple[list[object], int]:
    """Reference helper mirroring the required `parse_model()` special-case logic.

    This helper is intentionally independent of Ultralytics internals so it can be unit
    tested here. In the real `tasks.py`, the equivalent code is only a few lines.
    """

    if not isinstance(from_indices, (list, tuple)) or len(from_indices) != 2:
        raise ValueError("FreqFusionConcat requires exactly two source indices")

    hr_idx, lr_idx = int(from_indices[0]), int(from_indices[1])
    hr_c, lr_c = int(channel_table[hr_idx]), int(channel_table[lr_idx])
    parsed_args = [hr_c, lr_c, *list(yaml_args)]
    out_channels = hr_c + lr_c
    return parsed_args, out_channels


__all__ = [
    "FreqFusionConcat",
    "FreqFusionConcatDebug",
    "infer_freqfusion_parse_args",
]

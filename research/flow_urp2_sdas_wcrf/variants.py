from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class VariantSpec:
    key: str
    name: str
    modules: Tuple[str, ...]
    detect_p3_source: str
    uses_p2_at_inference: bool
    uses_train_only_p2_aux: bool
    custom_detect: bool
    graph_contract: str


# Audited Ultralytics 8.4.113 YOLO11 anchors used by the seed79 project.
# These are *semantic anchors*, not permission to blindly patch numeric indices in a
# different runtime. Codex must re-check them against the local stock yolo11n.yaml.
ANCHORS = {
    "p2_backbone": "stock backbone P2/4 output (audited project layer 2)",
    "p3_fused": "final fused P3/8 C3k2 before PAN bottom-up (audited layer 16)",
    "p4_fused": "final fused P4/16 feature (audited layer 19)",
    "p5_fused": "final fused P5/32 feature (audited layer 22)",
    "detect_stock": "Detect(P3,P4,P5) = audited sources [16,19,22]",
}


VARIANTS = {
    # Single-module versions -------------------------------------------------
    "U": VariantSpec(
        key="U",
        name="yolo11n_urp2_seed79",
        modules=("URP2",),
        detect_p3_source="stock P3",
        uses_p2_at_inference=True,
        uses_train_only_p2_aux=False,
        custom_detect=True,
        graph_contract=(
            "Stock P2 feeds UR-P2 detail projection; stock P3/P4/P5 remain the main "
            "detector features. A custom Detect subclass computes stock P3 raw DFL/class "
            "logits, applies URP2Refiner residuals to P3 only, then follows the stock "
            "decode/loss path. P4/P5 are untouched."
        ),
    ),
    "S": VariantSpec(
        key="S",
        name="yolo11n_sdas_seed79",
        modules=("SDAS",),
        detect_p3_source="stock P3",
        uses_p2_at_inference=False,
        uses_train_only_p2_aux=True,
        custom_detect=False,
        graph_contract=(
            "Attach SDASHead to stock P2 during training only. Add lambda_sdas * center-"
            "heatmap focal loss to the stock YOLO loss. Remove the SDAS head/output from "
            "eval/export so inference is exactly stock Detect(P3,P4,P5)."
        ),
    ),
    "W": VariantSpec(
        key="W",
        name="yolo11n_wcrf_seed79",
        modules=("WCRF",),
        detect_p3_source="WCRF(P3) detect-only side branch",
        uses_p2_at_inference=False,
        uses_train_only_p2_aux=False,
        custom_detect=False,
        graph_contract=(
            "Insert WCRF after the final fused P3 C3k2 as a Detect-only side branch. "
            "The untouched stock P3 must still feed the normal bottom-up PAN P4/P5 path. "
            "Detect sources become [WCRF(P3), stock P4, stock P5]."
        ),
    ),

    # Two-module versions ----------------------------------------------------
    "US": VariantSpec(
        key="US",
        name="yolo11n_urp2_sdas_seed79",
        modules=("SDAS", "URP2"),
        detect_p3_source="stock P3",
        uses_p2_at_inference=True,
        uses_train_only_p2_aux=True,
        custom_detect=True,
        graph_contract=(
            "P2 is supervised by SDAS during training and is also the UR-P2 detail source. "
            "UR-P2 refines only P3 raw logits; P4/P5 remain stock. SDAS is removed in "
            "eval/export, UR-P2 remains."
        ),
    ),
    "UW": VariantSpec(
        key="UW",
        name="yolo11n_wcrf_urp2_seed79",
        modules=("WCRF", "URP2"),
        detect_p3_source="WCRF(P3)",
        uses_p2_at_inference=True,
        uses_train_only_p2_aux=False,
        custom_detect=True,
        graph_contract=(
            "WCRF forms a Detect-only semantic P3 branch; the untouched P3 still feeds PAN. "
            "UR-P2 then uses stock P2 detail + WCRF(P3) semantic context and refines the P3 "
            "raw logits. P4/P5 remain stock. Order is WCRF -> UR-P2, never the reverse."
        ),
    ),
    "SW": VariantSpec(
        key="SW",
        name="yolo11n_sdas_wcrf_seed79",
        modules=("SDAS", "WCRF"),
        detect_p3_source="WCRF(P3) detect-only side branch",
        uses_p2_at_inference=False,
        uses_train_only_p2_aux=True,
        custom_detect=False,
        graph_contract=(
            "SDAS supervises P2 during training only. WCRF transforms only the Detect-side "
            "P3 branch. PAN and P4/P5 stay stock. SDAS disappears for inference/export."
        ),
    ),

    # Three-module version ---------------------------------------------------
    "USW": VariantSpec(
        key="USW",
        name="yolo11n_sdas_wcrf_urp2_seed79",
        modules=("SDAS", "WCRF", "URP2"),
        detect_p3_source="WCRF(P3)",
        uses_p2_at_inference=True,
        uses_train_only_p2_aux=True,
        custom_detect=True,
        graph_contract=(
            "Training: SDAS supervises stock P2; WCRF creates the Detect-side P3 semantic "
            "branch; UR-P2 uses P2 detail + WCRF(P3) and DFL uncertainty to refine only P3 "
            "raw logits. PAN/P4/P5 are stock. Eval/export: SDAS is removed; WCRF+UR-P2 stay."
        ),
    ),
}


TRAIN_ORDER = ("U", "S", "W", "US", "UW", "SW", "USW")


def get_variant(key: str) -> VariantSpec:
    key = key.upper()
    if key not in VARIANTS:
        raise KeyError(f"unknown variant {key!r}; choose from {tuple(VARIANTS)}")
    return VARIANTS[key]


__all__ = ["ANCHORS", "VARIANTS", "TRAIN_ORDER", "VariantSpec", "get_variant"]

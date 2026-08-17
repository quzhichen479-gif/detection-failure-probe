"""Pre-training loss-surface probe for SRB-IoU.

This script never loads a model or dataset.  It scans synthetic box geometry to
compare IoU, EIoU, MPDIoU, RB-v0 and SRB before any detector training.

Example:

    python research_tracks/srb_iou_loss/loss_surface_probe.py --out outputs/srb_surface
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from srb_iou import rb_iou_v0_loss, srb_iou_loss

LossFn = Callable[[Tensor, Tensor], Tensor]


def box_from_state(state: Tensor) -> Tensor:
    """Convert [cx, cy, log_w, log_h] to a single xyxy box."""
    cx, cy, log_w, log_h = state.unbind(dim=-1)
    w = log_w.exp()
    h = log_h.exp()
    return torch.stack(
        (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
        dim=-1,
    )


def box(cx: float, cy: float, w: float, h: float) -> Tensor:
    return torch.tensor(
        [[cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]],
        dtype=torch.float64,
    )


def _intersection_union(pred: Tensor, target: Tensor, eps: float = 1e-12) -> tuple[Tensor, ...]:
    px1, py1, px2, py2 = pred.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = target.unbind(dim=-1)
    pw = (px2 - px1).clamp_min(eps)
    ph = (py2 - py1).clamp_min(eps)
    gw = (gx2 - gx1).clamp_min(eps)
    gh = (gy2 - gy1).clamp_min(eps)
    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)
    iw = (ix2 - ix1).clamp_min(0.0)
    ih = (iy2 - iy1).clamp_min(0.0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter
    return inter, union, pw, ph, gw, gh


def iou_loss(pred: Tensor, target: Tensor) -> Tensor:
    inter, union, *_ = _intersection_union(pred, target)
    return (1.0 - inter / union).mean()


def eiou_loss(pred: Tensor, target: Tensor) -> Tensor:
    inter, union, pw, ph, gw, gh = _intersection_union(pred, target)
    iou = inter / union
    px1, py1, px2, py2 = pred.unbind(dim=-1)
    gx1, gy1, gx2, gy2 = target.unbind(dim=-1)
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    gcx = (gx1 + gx2) / 2.0
    gcy = (gy1 + gy2) / 2.0
    cw = torch.maximum(px2, gx2) - torch.minimum(px1, gx1)
    ch = torch.maximum(py2, gy2) - torch.minimum(py1, gy1)
    eps = torch.finfo(pred.dtype).eps
    center = ((pcx - gcx).square() + (pcy - gcy).square()) / (
        cw.square() + ch.square() + eps
    )
    width = (pw - gw).square() / (cw.square() + eps)
    height = (ph - gh).square() / (ch.square() + eps)
    return (1.0 - iou + center + width + height).mean()


def mpdiou_loss(pred: Tensor, target: Tensor, canvas_diag_sq: float = 640.0**2 * 2.0) -> Tensor:
    """Reference MPDIoU-like point-distance control for surface comparison."""
    inter, union, *_ = _intersection_union(pred, target)
    iou = inter / union
    p1 = pred[..., :2]
    p2 = pred[..., 2:]
    g1 = target[..., :2]
    g2 = target[..., 2:]
    d1 = (p1 - g1).square().sum(dim=-1)
    d2 = (p2 - g2).square().sum(dim=-1)
    return (1.0 - iou + (d1 + d2) / float(canvas_diag_sq)).mean()


def make_losses(delta: float) -> dict[str, LossFn]:
    return {
        "iou": iou_loss,
        "eiou": eiou_loss,
        "mpdiou": mpdiou_loss,
        "rb_v0": lambda p, t: rb_iou_v0_loss(
            p,
            t,
            delta=delta,
            reduction="mean",
        ),
        "srb": lambda p, t: srb_iou_loss(
            p,
            t,
            delta=delta,
            reduction="mean",
        ),
    }


def state_for_box(cx: float, cy: float, w: float, h: float) -> Tensor:
    return torch.tensor(
        [cx, cy, math.log(w), math.log(h)],
        dtype=torch.float64,
        requires_grad=True,
    )


def eval_state(loss_fn: LossFn, state: Tensor, target: Tensor) -> dict[str, float]:
    pred = box_from_state(state).unsqueeze(0)
    loss = loss_fn(pred, target)
    grad = torch.autograd.grad(loss, state, create_graph=False)[0]
    return {
        "loss": float(loss.detach()),
        "grad_norm": float(grad.norm().detach()),
        "grad_cx": float(grad[0].detach()),
        "grad_cy": float(grad[1].detach()),
        "grad_logw": float(grad[2].detach()),
        "grad_logh": float(grad[3].detach()),
    }


def hessian_eigs(loss_fn: LossFn, state: Tensor, target: Tensor) -> tuple[float, float]:
    def objective(x: Tensor) -> Tensor:
        return loss_fn(box_from_state(x).unsqueeze(0), target)

    hessian = torch.autograd.functional.hessian(objective, state.detach())
    eigs = torch.linalg.eigvalsh(hessian)
    return float(eigs.min()), float(eigs.max())


def translation_rows(
    losses: dict[str, LossFn],
    sizes: list[tuple[float, float]],
) -> list[dict[str, float | str]]:
    fractions = [1e-4, 0.01, 0.05, 0.125, 0.25, 0.5, 0.75, 0.99, 1.01, 1.5, 2.0]
    rows: list[dict[str, float | str]] = []
    for w, h in sizes:
        target = box(0.0, 0.0, w, h)
        for frac in fractions:
            dx = frac * w
            for name, fn in losses.items():
                state = state_for_box(dx, 0.0, w, h)
                metrics = eval_state(fn, state, target)
                rows.append(
                    {
                        "scan": "translation_x",
                        "loss_name": name,
                        "gt_w": w,
                        "gt_h": h,
                        "fraction": frac,
                        "dx": dx,
                        **metrics,
                    }
                )
    return rows


def scale_rows(
    losses: dict[str, LossFn],
    sizes: list[tuple[float, float]],
) -> list[dict[str, float | str]]:
    ratios = [0.25, 0.5, 0.75, 0.9, 0.99, 1.01, 1.1, 1.25, 1.5, 2.0, 4.0]
    rows: list[dict[str, float | str]] = []
    for w, h in sizes:
        target = box(0.0, 0.0, w, h)
        for ratio in ratios:
            for name, fn in losses.items():
                state = state_for_box(0.0, 0.0, w * ratio, h * ratio)
                metrics = eval_state(fn, state, target)
                rows.append(
                    {
                        "scan": "isotropic_scale",
                        "loss_name": name,
                        "gt_w": w,
                        "gt_h": h,
                        "ratio": ratio,
                        **metrics,
                    }
                )
    return rows


def curvature_rows(
    losses: dict[str, LossFn],
    sizes: list[tuple[float, float]],
) -> list[dict[str, float | str]]:
    # Stay inside smooth overlap regions; exact contact and exact equality are
    # topology boundaries and must not be used to claim a classical Hessian.
    probes = [(0.05, 1.0), (0.25, 1.0), (0.05, 0.9), (0.05, 1.1)]
    rows: list[dict[str, float | str]] = []
    for w, h in sizes:
        target = box(0.0, 0.0, w, h)
        for dx_frac, scale in probes:
            state = state_for_box(dx_frac * w, 0.0, w * scale, h * scale)
            for name, fn in losses.items():
                lo, hi = hessian_eigs(fn, state, target)
                rows.append(
                    {
                        "scan": "hessian",
                        "loss_name": name,
                        "gt_w": w,
                        "gt_h": h,
                        "dx_fraction": dx_frac,
                        "scale": scale,
                        "lambda_min": lo,
                        "lambda_max": hi,
                    }
                )
    return rows


def stationary_scan(
    loss_fn: LossFn,
    *,
    seed: int,
    n_starts: int,
    steps: int,
) -> dict[str, float | int]:
    """Descriptive random search for non-GT low-gradient basins, not a proof."""
    generator = torch.Generator().manual_seed(seed)
    target = box(0.0, 0.0, 8.0, 8.0)
    suspicious = 0
    converged = 0
    best_bad_loss = float("inf")

    for _ in range(n_starts):
        cx = float(torch.empty(1).uniform_(-24.0, 24.0, generator=generator))
        cy = float(torch.empty(1).uniform_(-24.0, 24.0, generator=generator))
        w = float(torch.empty(1).uniform_(0.75, 24.0, generator=generator))
        h = float(torch.empty(1).uniform_(0.75, 24.0, generator=generator))
        state = state_for_box(cx, cy, w, h)
        opt = torch.optim.Adam([state], lr=0.08)

        for _step in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(box_from_state(state).unsqueeze(0), target)
            loss.backward()
            opt.step()

        with torch.enable_grad():
            pred = box_from_state(state).unsqueeze(0)
            loss = loss_fn(pred, target)
            grad = torch.autograd.grad(loss, state)[0]
            loss_v = float(loss.detach())
            grad_v = float(grad.norm().detach())

        if loss_v < 1e-5:
            converged += 1
        elif grad_v < 1e-5:
            suspicious += 1
            best_bad_loss = min(best_bad_loss, loss_v)

    return {
        "n_starts": n_starts,
        "converged_to_gt_like": converged,
        "suspicious_non_gt_low_grad": suspicious,
        "best_suspicious_loss": None if suspicious == 0 else best_bad_loss,
    }


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(out_dir: Path, rows: list[dict[str, float | str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    selected = [
        row
        for row in rows
        if row["scan"] == "translation_x"
        and row["gt_w"] == 8.0
        and row["gt_h"] == 8.0
    ]
    names = sorted({str(row["loss_name"]) for row in selected})
    for metric in ("loss", "grad_norm"):
        fig, ax = plt.subplots()
        for name in names:
            part = sorted(
                (row for row in selected if row["loss_name"] == name),
                key=lambda row: float(row["fraction"]),
            )
            ax.plot(
                [float(row["fraction"]) for row in part],
                [float(row[metric]) for row in part],
                marker="o",
                label=name,
            )
        ax.set_xlabel("horizontal offset / GT width")
        ax.set_ylabel(metric)
        ax.set_title(f"8x8 translation scan: {metric}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"translation_8x8_{metric}.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/srb_loss_surface"))
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--stationary-starts", type=int, default=100)
    parser.add_argument("--stationary-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    losses = make_losses(args.delta)
    sizes = [
        (4.0, 4.0),
        (8.0, 8.0),
        (16.0, 16.0),
        (32.0, 32.0),
        (64.0, 64.0),
        (32.0, 4.0),
        (32.0, 8.0),
        (64.0, 4.0),
    ]

    translation = translation_rows(losses, sizes)
    scale = scale_rows(losses, sizes)
    curvature = curvature_rows(losses, sizes)
    write_csv(args.out / "translation_scan.csv", translation)
    write_csv(args.out / "scale_scan.csv", scale)
    write_csv(args.out / "curvature_scan.csv", curvature)
    maybe_plot(args.out, translation)

    stationary = stationary_scan(
        losses["srb"],
        seed=args.seed,
        n_starts=args.stationary_starts,
        steps=args.stationary_steps,
    )
    summary = {
        "delta": args.delta,
        "sizes": sizes,
        "stationary_scan": stationary,
        "notes": [
            "Hessians are evaluated only inside smooth overlap regions.",
            "The stationary scan is descriptive and is not a proof of global convexity.",
            "MPDIoU uses a 640x640 canvas-diagonal normalization for this probe.",
        ],
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

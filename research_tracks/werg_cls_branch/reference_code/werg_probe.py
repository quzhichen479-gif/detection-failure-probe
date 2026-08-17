from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from werg_core import WERGConfig, WERGReference


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def roc_auc(labels: Iterable[int], scores: Iterable[float]) -> float | None:
    y = [int(v) for v in labels]
    s = [float(v) for v in scores]
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(s)
    rank_sum_pos = sum(r for r, label in zip(ranks, y, strict=True) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_at_recall(labels: Tensor, scores: Tensor, recall: float) -> float:
    positives = scores[labels == 1]
    if positives.numel() == 0:
        raise ValueError("cannot define a recall threshold without positives")
    q = max(0.0, min(1.0, 1.0 - float(recall)))
    return float(torch.quantile(positives, q).item())


def binary_metrics(labels: Tensor, scores: Tensor, recall: float = 0.90) -> dict[str, float | int | None]:
    labels = labels.to(torch.int64).cpu()
    scores = scores.to(torch.float32).cpu()
    auc = roc_auc(labels.tolist(), scores.tolist())
    threshold = threshold_at_recall(labels, scores, recall)
    pred = scores >= threshold
    tp = int(((pred == 1) & (labels == 1)).sum().item())
    fp = int(((pred == 1) & (labels == 0)).sum().item())
    fn = int(((pred == 0) & (labels == 1)).sum().item())
    tn = int(((pred == 0) & (labels == 0)).sum().item())
    achieved_recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return {
        "auc": auc,
        "target_recall": recall,
        "threshold": threshold,
        "achieved_recall": achieved_recall,
        "false_positives": fp,
        "false_positive_rate": fpr,
        "tp": tp,
        "tn": tn,
        "fn": fn,
    }


def _design_matrix(rows: list[dict[str, object]], extended: bool) -> Tensor:
    result = []
    for row in rows:
        semantic = float(row["semantic_logit"])
        if not extended:
            result.append([semantic])
            continue
        zw = float(row["z_w"])
        zg = float(row["z_g"])
        result.append([semantic, zw, zg, zw * zw, zw * zg, zg * zg])
    return torch.tensor(result, dtype=torch.float32)


def fit_logistic_probe(
    fit_rows: list[dict[str, object]],
    eval_rows: list[dict[str, object]],
    *,
    extended: bool,
    steps: int = 400,
    lr: float = 0.05,
) -> tuple[Tensor, dict[str, float | int | None]]:
    x_fit = _design_matrix(fit_rows, extended)
    y_fit = torch.tensor([int(r["label"]) for r in fit_rows], dtype=torch.float32)
    x_eval = _design_matrix(eval_rows, extended)
    y_eval = torch.tensor([int(r["label"]) for r in eval_rows], dtype=torch.int64)

    mean = x_fit.mean(dim=0, keepdim=True)
    std = x_fit.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_fit = (x_fit - mean) / std
    x_eval = (x_eval - mean) / std

    weight = torch.zeros(x_fit.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = x_fit @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_fit)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eval_score = x_eval @ weight + bias
    return torch.cat([weight.detach(), bias.detach().view(1)]), binary_metrics(y_eval, eval_score)


def extract_rows(bundle: dict[str, object], device: str) -> list[dict[str, object]]:
    cfg = WERGConfig(detach_statistics=True)
    rows: list[dict[str, object]] = []
    model: WERGReference | None = None

    for sample in bundle.get("samples", []):
        if not isinstance(sample, dict):
            raise TypeError("each sample must be a dict")
        feature = torch.as_tensor(sample["p3_feature"]).unsqueeze(0).to(device)
        rgb = torch.as_tensor(sample["rgb"]).unsqueeze(0).to(device)
        semantic = torch.as_tensor(sample["semantic_logit"]).to(device)
        if semantic.ndim == 2:
            semantic = semantic.unsqueeze(0)
        if semantic.ndim != 3:
            raise ValueError("semantic_logit must be HxW or CxHxW")
        semantic = semantic.unsqueeze(0)

        if model is None:
            model = WERGReference(int(feature.shape[1]), cfg).to(device).eval()
        elif model.water.channels != feature.shape[1]:
            raise ValueError("all samples must use the same P3 channel count")

        with torch.no_grad():
            water = model.water(feature)["z_w"][0, 0]
            geom = model.geometry(rgb, feature.shape[-2:])["z_g"][0, 0]

        for candidate in sample.get("candidates", []):
            if not isinstance(candidate, dict):
                raise TypeError("candidate must be a dict")
            iy = int(candidate["y"])
            ix = int(candidate["x"])
            if not (0 <= iy < water.shape[0] and 0 <= ix < water.shape[1]):
                raise IndexError(f"candidate {(iy, ix)} is outside P3 map {tuple(water.shape)}")
            if semantic.shape[1] == 1:
                sem = semantic[0, 0, iy, ix]
            else:
                sem = semantic[0, :, iy, ix].max()
            rows.append(
                {
                    "sample_id": str(sample.get("id", "")),
                    "y": iy,
                    "x": ix,
                    "label": int(candidate["label"]),
                    "group": str(candidate.get("group", "unknown")),
                    "probe_split": str(candidate.get("probe_split", "unassigned")),
                    "semantic_logit": float(sem.item()),
                    "z_w": float(water[iy, ix].item()),
                    "z_g": float(geom[iy, ix].item()),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.int64)
    summary: dict[str, object] = {
        "num_candidates": len(rows),
        "num_positive": int(labels.sum().item()),
        "num_negative": int((labels == 0).sum().item()),
        "individual_auc": {},
        "groups": {},
    }
    for key in ("semantic_logit", "z_w", "z_g"):
        scores = [float(r[key]) for r in rows]
        summary["individual_auc"][key] = roc_auc(labels.tolist(), scores)

    groups = sorted({str(r["group"]) for r in rows})
    for group in groups:
        subset = [r for r in rows if str(r["group"]) == group]
        y = [int(r["label"]) for r in subset]
        summary["groups"][group] = {
            "n": len(subset),
            "positive": sum(y),
            "auc_z_w": roc_auc(y, [float(r["z_w"]) for r in subset]),
            "auc_z_g": roc_auc(y, [float(r["z_g"]) for r in subset]),
        }

    fit_rows = [r for r in rows if r["probe_split"] == "fit"]
    eval_rows = [r for r in rows if r["probe_split"] == "eval"]
    if fit_rows and eval_rows and {int(r["label"]) for r in fit_rows} == {0, 1} and {int(r["label"]) for r in eval_rows} == {0, 1}:
        _, sem_metrics = fit_logistic_probe(fit_rows, eval_rows, extended=False)
        weights, ext_metrics = fit_logistic_probe(fit_rows, eval_rows, extended=True)
        summary["conditional_probe"] = {
            "semantic_only": sem_metrics,
            "semantic_plus_werg": ext_metrics,
            "extended_weights_plus_bias": weights.tolist(),
            "fp_reduction_at_target_recall": (
                None
                if sem_metrics["false_positives"] in (None, 0)
                else (
                    float(sem_metrics["false_positives"]) - float(ext_metrics["false_positives"])
                )
                / float(sem_metrics["false_positives"])
            ),
        }
    else:
        summary["conditional_probe"] = {
            "status": "not_run",
            "reason": "provide both fit/eval candidates with positive and negative labels",
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="WERG frozen-detector zero-training probe")
    parser.add_argument("--bundle", type=Path, required=True, help="torch .pt bundle; see README schema")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError("bundle root must be a dict")
    rows = extract_rows(bundle, args.device)
    if not rows:
        raise RuntimeError("bundle contains no candidates")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "werg_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    summary["bundle"] = str(args.bundle)
    summary["device"] = args.device
    json_path = args.out_dir / "werg_probe_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class GroundTruth:
    gt_id: int
    class_id: int
    box: np.ndarray  # xyxy, original-image pixels


@dataclass(frozen=True)
class ImageSample:
    image_id: str
    image_path: Path
    label_path: Path
    gts: tuple[GroundTruth, ...]


@dataclass
class InferenceResult:
    boxes: np.ndarray       # [N,4] xyxy in original-image pixels
    scores: np.ndarray      # [N]
    classes: np.ndarray     # [N]
    raw_boxes: np.ndarray | None = None      # [M,4], original-image pixels
    raw_scores: np.ndarray | None = None     # [M,nc]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join([str(base_seed), *map(str, parts)]).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % (2**32)


def _resolve_root(yaml_path: Path, raw_root: str | None) -> Path:
    if not raw_root:
        return yaml_path.parent.resolve()
    root = Path(raw_root)
    return root.resolve() if root.is_absolute() else (yaml_path.parent / root).resolve()


def _iter_split_paths(root: Path, split_value: Any) -> list[Path]:
    values = split_value if isinstance(split_value, list) else [split_value]
    out: list[Path] = []
    for value in values:
        p = Path(value)
        p = p if p.is_absolute() else root / p
        if p.suffix.lower() == ".txt":
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s:
                    continue
                q = Path(s)
                out.append(q if q.is_absolute() else root / q)
        elif p.is_dir():
            out.extend(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_SUFFIXES)
        elif p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
        else:
            raise FileNotFoundError(f"Unsupported/missing dataset split entry: {p}")
    return sorted({x.resolve() for x in out})


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [p.lower() for p in parts]
    if "images" in lowered:
        idx = len(lowered) - 1 - lowered[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def load_yolo_label(label_path: Path, width: int, height: int) -> tuple[GroundTruth, ...]:
    if not label_path.exists():
        return tuple()
    gts: list[GroundTruth] = []
    for gt_id, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        fields = line.strip().split()
        if len(fields) < 5:
            continue
        cls, xc, yc, w, h = map(float, fields[:5])
        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height
        box = np.array(
            [max(0.0, x1), max(0.0, y1), min(float(width), x2), min(float(height), y2)],
            dtype=np.float32,
        )
        gts.append(GroundTruth(gt_id=gt_id, class_id=int(cls), box=box))
    return tuple(gts)


def load_dataset(yaml_file: str | Path, split: str, max_images: int | None = None) -> list[ImageSample]:
    yaml_path = Path(yaml_file).resolve()
    spec = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if split not in spec:
        raise KeyError(f"Split {split!r} not present in {yaml_path}")
    root = _resolve_root(yaml_path, spec.get("path"))
    image_paths = _iter_split_paths(root, spec[split])
    if max_images and max_images > 0:
        image_paths = image_paths[:max_images]

    samples: list[ImageSample] = []
    for image_path in image_paths:
        with Image.open(image_path) as im:
            width, height = im.size
        label_path = label_path_for_image(image_path)
        gts = load_yolo_label(label_path, width, height)
        try:
            image_id = str(image_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            image_id = image_path.name
        samples.append(ImageSample(image_id, image_path, label_path, gts))
    return samples


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.astype(np.float32, copy=True)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def scale_letterbox_boxes(boxes: np.ndarray, input_hw: tuple[int, int], orig_hw: tuple[int, int]) -> np.ndarray:
    if boxes.size == 0:
        return boxes.astype(np.float32, copy=True)
    in_h, in_w = input_hw
    orig_h, orig_w = orig_hw
    gain = min(in_h / orig_h, in_w / orig_w)
    pad_x = (in_w - orig_w * gain) / 2
    pad_y = (in_h - orig_h * gain) / 2
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / gain
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / gain
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def classify_original_predictions(
    pred: InferenceResult,
    gts: tuple[GroundTruth, ...],
    tp_iou: float = 0.5,
    strict_fp_iou: float = 0.1,
) -> list[dict[str, Any]]:
    gt_boxes = np.array([g.box for g in gts], dtype=np.float32).reshape(-1, 4)
    gt_cls = np.array([g.class_id for g in gts], dtype=np.int64)
    ious = iou_matrix(pred.boxes, gt_boxes)
    order = np.argsort(-pred.scores)
    matched_gt: set[int] = set()
    records: list[dict[str, Any]] = []

    for pred_idx in order:
        cls_id = int(pred.classes[pred_idx])
        max_any_iou = float(ious[pred_idx].max()) if len(gts) else 0.0
        best_gt = -1
        best_iou = 0.0
        if len(gts):
            eligible = np.where(gt_cls == cls_id)[0]
            if len(eligible):
                local = eligible[np.argmax(ious[pred_idx, eligible])]
                best_gt = int(local)
                best_iou = float(ious[pred_idx, local])

        candidate_type = "other"
        gt_id: int | None = None
        if best_gt >= 0 and best_iou >= tp_iou and best_gt not in matched_gt:
            candidate_type = "tp"
            matched_gt.add(best_gt)
            gt_id = best_gt
        elif max_any_iou < strict_fp_iou:
            candidate_type = "strict_background_fp"
        elif max_any_iou < tp_iou:
            candidate_type = "near_gt_localization_error"

        raw_idx = None
        raw_score = None
        if pred.raw_boxes is not None and pred.raw_scores is not None and len(pred.raw_boxes):
            same_cls_scores = pred.raw_scores[:, cls_id]
            raw_ious = iou_matrix(pred.boxes[pred_idx : pred_idx + 1], pred.raw_boxes)[0]
            # Prefer geometric identity; score breaks near ties.
            objective = raw_ious + 1e-3 * same_cls_scores
            raw_idx = int(np.argmax(objective))
            raw_score = float(same_cls_scores[raw_idx])

        records.append(
            {
                "pred_idx": int(pred_idx),
                "candidate_type": candidate_type,
                "gt_id": gt_id,
                "class_id": cls_id,
                "score_x": float(pred.scores[pred_idx]),
                "box_x": pred.boxes[pred_idx].astype(float).tolist(),
                "iou_gt_x": best_iou if gt_id is not None else max_any_iou,
                "max_any_iou_x": max_any_iou,
                "raw_idx": raw_idx,
                "raw_score_x": raw_score,
            }
        )
    return records


def correspond_prediction(
    record: dict[str, Any],
    pert: InferenceResult,
    gts: tuple[GroundTruth, ...],
    fp_match_iou: float = 0.3,
    tp_min_gt_iou: float = 0.1,
) -> dict[str, Any]:
    cls_id = int(record["class_id"])
    same = np.where(pert.classes.astype(int) == cls_id)[0]
    matched_idx: int | None = None
    iou_gt = 0.0

    if record["candidate_type"] == "tp" and record["gt_id"] is not None:
        gt = gts[int(record["gt_id"])]
        if len(same):
            vals = iou_matrix(pert.boxes[same], gt.box[None, :])[:, 0]
            k = int(np.argmax(vals))
            if float(vals[k]) >= tp_min_gt_iou:
                matched_idx = int(same[k])
                iou_gt = float(vals[k])
    else:
        if len(same):
            ref = np.asarray(record["box_x"], dtype=np.float32)[None, :]
            vals = iou_matrix(pert.boxes[same], ref)[:, 0]
            k = int(np.argmax(vals))
            if float(vals[k]) >= fp_match_iou:
                matched_idx = int(same[k])

    if matched_idx is None:
        score_xw = 0.0
        box_xw = None
        matched = False
    else:
        score_xw = float(pert.scores[matched_idx])
        box_xw = pert.boxes[matched_idx].astype(float).tolist()
        matched = True

    raw_score_xw = None
    raw_idx = record.get("raw_idx")
    if (
        raw_idx is not None
        and pert.raw_scores is not None
        and 0 <= int(raw_idx) < len(pert.raw_scores)
        and cls_id < pert.raw_scores.shape[1]
    ):
        raw_score_xw = float(pert.raw_scores[int(raw_idx), cls_id])

    return {
        "score_xw": score_xw,
        "box_xw": box_xw,
        "matched_xw": matched,
        "disappeared": not matched,
        "iou_gt_xw": iou_gt if record["candidate_type"] == "tp" else None,
        "raw_score_xw": raw_score_xw,
    }


def _draw_rect(mask: np.ndarray, box: np.ndarray, margin: int = 0) -> None:
    h, w = mask.shape
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0, int(math.floor(x1)) - margin)
    y1 = max(0, int(math.floor(y1)) - margin)
    x2 = min(w, int(math.ceil(x2)) + margin)
    y2 = min(h, int(math.ceil(y2)) + margin)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True


def build_intervention_masks(
    image_hw: tuple[int, int],
    gts: tuple[GroundTruth, ...],
    min_protect_px: int = 8,
    protect_fraction: float = 0.25,
    near_radius_px: int = 64,
) -> dict[str, np.ndarray]:
    h, w = image_hw
    protected = np.zeros((h, w), dtype=bool)
    near_outer = np.zeros((h, w), dtype=bool)
    for gt in gts:
        bw = max(1.0, float(gt.box[2] - gt.box[0]))
        bh = max(1.0, float(gt.box[3] - gt.box[1]))
        margin = max(min_protect_px, int(round(protect_fraction * min(bw, bh))))
        _draw_rect(protected, gt.box, margin)
        _draw_rect(near_outer, gt.box, max(near_radius_px, margin * 3))
    allowed = ~protected
    near = near_outer & allowed
    far = (~near_outer) & allowed
    if not gts:
        near[:] = False
        far[:] = True
    return {"protected": protected, "near": near, "far": far, "all": allowed}


def _choose_centers(mask: np.ndarray, rng: np.random.Generator, n: int) -> list[tuple[int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=len(xs) < n)
    return [(int(xs[i]), int(ys[i])) for i in np.atleast_1d(idx)]


def apply_perturbation(
    image: np.ndarray,
    allowed_mask: np.ndarray,
    kind: str,
    strength: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return perturbed RGB uint8 image and the exact changed-pixel mask."""
    src = image.astype(np.float32)
    h, w, _ = src.shape
    mask = allowed_mask.astype(bool)
    if kind == "null":
        return image.copy(), np.zeros((h, w), dtype=bool)

    out = src.copy()
    alpha = np.zeros((h, w), dtype=np.float32)

    if kind == "photometric":
        # Controlled nuisance, intentionally generic rather than water-specific.
        delta = 18.0 * strength
        alpha[mask] = 1.0
        transformed = np.clip((src - 127.5) * (1 + 0.12 * strength) + 127.5 + delta, 0, 255)
        out[mask] = transformed[mask]

    elif kind == "glare":
        yy, xx = np.mgrid[0:h, 0:w]
        field = np.zeros((h, w), dtype=np.float32)
        for cx, cy in _choose_centers(mask, rng, 3):
            sx = rng.uniform(12, max(14, 0.10 * w)) * (0.6 + 0.5 * strength)
            sy = rng.uniform(4, max(6, 0.03 * h)) * (0.6 + 0.5 * strength)
            theta = rng.uniform(0, math.pi)
            ct, st = math.cos(theta), math.sin(theta)
            dx = xx - cx
            dy = yy - cy
            xr = ct * dx + st * dy
            yr = -st * dx + ct * dy
            blob = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
            field = np.maximum(field, blob.astype(np.float32))
        alpha = np.clip(field * (0.18 + 0.28 * strength), 0, 0.75)
        alpha *= mask
        white = np.full_like(src, 255.0)
        out = src * (1 - alpha[..., None]) + white * alpha[..., None]

    elif kind == "ripple":
        theta = rng.uniform(0, math.pi)
        frequency = rng.uniform(0.015, 0.055)
        phase = rng.uniform(0, 2 * math.pi)
        yy, xx = np.mgrid[0:h, 0:w]
        carrier = np.sin(
            2 * math.pi * frequency * (math.cos(theta) * xx + math.sin(theta) * yy) + phase
        ).astype(np.float32)
        amp = (6.0 + 14.0 * strength) * carrier
        alpha[mask] = 1.0
        out[mask] = np.clip(src[mask] + amp[mask, None], 0, 255)

    elif kind == "highfreq":
        noise = rng.standard_normal((h, w)).astype(np.float32)
        smooth = (
            noise
            + np.roll(noise, 1, 0)
            + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1)
            + np.roll(noise, -1, 1)
        ) / 5.0
        hp = noise - smooth
        hp /= max(float(hp.std()), 1e-6)
        amp = hp * (5.0 + 12.0 * strength)
        alpha[mask] = 1.0
        out[mask] = np.clip(src[mask] + amp[mask, None], 0, 255)

    else:
        raise ValueError(f"Unknown perturbation kind: {kind}")

    changed = mask & np.any(np.abs(out - src) > 1e-6, axis=2)
    return np.clip(out, 0, 255).astype(np.uint8), changed


def sensitivity_auc(tp_values: Iterable[float], fp_values: Iterable[float]) -> float | None:
    """P(|delta_FP| > |delta_TP|) with half credit for ties."""
    tp = np.sort(np.asarray(list(tp_values), dtype=np.float64))
    fp = np.asarray(list(fp_values), dtype=np.float64)
    if len(tp) == 0 or len(fp) == 0:
        return None
    less = np.searchsorted(tp, fp, side="left")
    leq = np.searchsorted(tp, fp, side="right")
    equal = leq - less
    return float(np.sum(less + 0.5 * equal) / (len(tp) * len(fp)))


def safe_median(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=np.float64)
    return None if len(arr) == 0 else float(np.median(arr))


def bootstrap_image_level(
    rows: list[dict[str, Any]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_image.setdefault(str(row["image_id"]), []).append(row)
    image_ids = sorted(by_image)
    if not image_ids:
        return {"n_boot": 0}

    rng = np.random.default_rng(seed)
    ratios: list[float] = []
    aucs: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample_rows = [r for iid in sampled for r in by_image[str(iid)]]
        tp = [float(r["abs_delta_score"]) for r in sample_rows if r["candidate_type"] == "tp"]
        fp = [
            float(r["abs_delta_score"])
            for r in sample_rows
            if r["candidate_type"] == "strict_background_fp"
        ]
        med_tp = safe_median(tp)
        med_fp = safe_median(fp)
        if med_tp is not None and med_fp is not None:
            ratios.append(med_fp / max(med_tp, 1e-9))
        auc = sensitivity_auc(tp, fp)
        if auc is not None:
            aucs.append(auc)

    def ci(vals: list[float]) -> list[float] | None:
        if not vals:
            return None
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

    return {
        "n_boot": n_boot,
        "sensitivity_ratio_ci95": ci(ratios),
        "auc_ci95": ci(aucs),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = [float(r["abs_delta_score"]) for r in rows if r["candidate_type"] == "tp"]
    fp = [
        float(r["abs_delta_score"])
        for r in rows
        if r["candidate_type"] == "strict_background_fp"
    ]
    high_fp = [
        float(r["abs_delta_score"])
        for r in rows
        if r["candidate_type"] == "strict_background_fp"
        and bool(r.get("high_conf_fp", float(r["score_x"]) >= 0.75))
    ]
    raw_tp = [
        float(r["pre_nms_abs_delta"])
        for r in rows
        if r["candidate_type"] == "tp" and r.get("pre_nms_abs_delta") is not None
    ]
    raw_fp = [
        float(r["pre_nms_abs_delta"])
        for r in rows
        if r["candidate_type"] == "strict_background_fp"
        and r.get("pre_nms_abs_delta") is not None
    ]
    med_tp = safe_median(tp)
    med_fp = safe_median(fp)
    raw_med_tp = safe_median(raw_tp)
    raw_med_fp = safe_median(raw_fp)
    return {
        "n_rows": len(rows),
        "n_tp": len(tp),
        "n_strict_fp": len(fp),
        "n_high_conf_fp": len(high_fp),
        "median_abs_delta_tp": med_tp,
        "median_abs_delta_strict_fp": med_fp,
        "median_abs_delta_high_conf_fp": safe_median(high_fp),
        "sensitivity_ratio": (
            None if med_tp is None or med_fp is None else med_fp / max(med_tp, 1e-9)
        ),
        "auc_tp_vs_strict_fp": sensitivity_auc(tp, fp),
        "auc_tp_vs_high_conf_fp": sensitivity_auc(tp, high_fp),
        "n_pre_nms_tp": len(raw_tp),
        "n_pre_nms_strict_fp": len(raw_fp),
        "pre_nms_median_abs_delta_tp": raw_med_tp,
        "pre_nms_median_abs_delta_strict_fp": raw_med_fp,
        "pre_nms_sensitivity_ratio": (
            None
            if raw_med_tp is None or raw_med_fp is None
            else raw_med_fp / max(raw_med_tp, 1e-9)
        ),
        "pre_nms_auc_tp_vs_strict_fp": sensitivity_auc(raw_tp, raw_fp),
        "tp_disappearance_rate": (
            None
            if not tp
            else float(
                np.mean([bool(r["disappeared"]) for r in rows if r["candidate_type"] == "tp"])
            )
        ),
        "fp_disappearance_rate": (
            None
            if not fp
            else float(
                np.mean(
                    [
                        bool(r["disappeared"])
                        for r in rows
                        if r["candidate_type"] == "strict_background_fp"
                    ]
                )
            )
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            cooked = {}
            for k in keys:
                v = row.get(k)
                if isinstance(v, (dict, list, tuple)):
                    cooked[k] = json.dumps(v, ensure_ascii=False)
                else:
                    cooked[k] = v
            writer.writerow(cooked)

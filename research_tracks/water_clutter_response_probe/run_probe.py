from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from probe_core import (  # noqa: E402
    apply_perturbation,
    bootstrap_image_level,
    build_intervention_masks,
    classify_original_predictions,
    correspond_prediction,
    load_dataset,
    sha256_file,
    stable_seed,
    summarize_rows,
    write_csv,
)
from ultralytics_adapter import UltralyticsYOLOAdapter  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero-training Baseline vs DBRA water-clutter response probe."
    )
    p.add_argument("--config", required=True, help="YAML config. See probe_config.example.yaml.")
    return p.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = [
        "baseline_weights",
        "dbra_weights",
        "dataset_yaml",
        "split",
        "imgsz",
        "conf",
        "iou_nms",
        "device",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config fields: {missing}")
    split = str(cfg["split"]).lower()
    if "test" in split:
        raise ValueError(
            "This project probe must not use the test split. Use the frozen validation/audit split."
        )
    for key in ("baseline_weights", "dbra_weights", "dataset_yaml"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"{key} does not exist: {cfg[key]}")
    return cfg


def git_info(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_status": run("status", "--short"),
        "git_diff_stat": run("diff", "--stat"),
    }


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def make_run_dir(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("output_dir", "outputs/water_response_probe")).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = cfg.get("run_id") or f"baseline-vs-dbra-{stamp}"
    out = root / str(run_id)
    out.mkdir(parents=True, exist_ok=False)
    for sub in ("raw", "analysis", "sanity/intervention_examples", "fp_review_contact_sheet"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def gt_boxes_payload(sample) -> list[dict[str, Any]]:
    return [
        {
            "gt_id": gt.gt_id,
            "class_id": gt.class_id,
            "box": gt.box.astype(float).tolist(),
        }
        for gt in sample.gts
    ]


def save_sanity(
    out_dir: Path,
    image_id: str,
    variant_name: str,
    image: np.ndarray,
    changed_mask: np.ndarray,
    protected: np.ndarray,
) -> None:
    safe = image_id.replace("/", "__").replace("\\", "__")
    stem = f"{safe}__{variant_name}"
    Image.fromarray(image).save(out_dir / f"{stem}.jpg", quality=92)
    overlay = np.zeros((*changed_mask.shape, 3), dtype=np.uint8)
    overlay[changed_mask] = np.array([255, 0, 0], dtype=np.uint8)
    overlay[protected] = np.array([0, 255, 0], dtype=np.uint8)
    Image.fromarray(overlay).save(out_dir / f"{stem}__mask.png")


def variants_for_image(
    image: np.ndarray,
    sample,
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    h, w = image.shape[:2]
    masks = build_intervention_masks(
        (h, w),
        sample.gts,
        min_protect_px=int(cfg.get("min_protect_px", 8)),
        protect_fraction=float(cfg.get("protect_fraction", 0.25)),
        near_radius_px=int(cfg.get("near_radius_px", 64)),
    )

    perturbations = cfg.get("perturbations", ["null", "photometric", "glare", "ripple", "highfreq"])
    strengths = [float(x) for x in cfg.get("strengths", [0.35, 0.7, 1.0])]
    regions = cfg.get("regions", ["near", "far"])
    seed = int(cfg.get("probe_seed", 3407))

    variants: list[dict[str, Any]] = [
        {
            "name": "null",
            "kind": "null",
            "strength": 0.0,
            "region": "all",
            "seed": stable_seed(seed, sample.image_id, "null"),
            "image": image.copy(),
            "changed_mask": np.zeros((h, w), dtype=bool),
        }
    ]

    for kind in perturbations:
        if kind == "null":
            continue
        use_regions = ["all"] if kind == "photometric" else list(regions)
        for region in use_regions:
            if region not in masks:
                raise ValueError(f"Unknown region {region!r}; available={list(masks)}")
            for strength in strengths:
                variant_seed = stable_seed(seed, sample.image_id, kind, strength, region)
                rng = np.random.default_rng(variant_seed)
                pert, changed = apply_perturbation(
                    image=image,
                    allowed_mask=masks[region],
                    kind=kind,
                    strength=strength,
                    rng=rng,
                )
                variants.append(
                    {
                        "name": f"{kind}_{region}_s{strength:g}",
                        "kind": kind,
                        "strength": strength,
                        "region": region,
                        "seed": variant_seed,
                        "image": pert,
                        "changed_mask": changed,
                    }
                )
    return variants, masks


def add_response_rows(
    rows: list[dict[str, Any]],
    model_name: str,
    sample,
    original_records,
    pert,
    variant,
    cfg,
) -> None:
    for rec in original_records:
        if rec["candidate_type"] not in {
            "tp",
            "strict_background_fp",
            "near_gt_localization_error",
        }:
            continue
        corr = correspond_prediction(
            rec,
            pert,
            sample.gts,
            fp_match_iou=float(cfg.get("fp_correspondence_iou", 0.3)),
            tp_min_gt_iou=float(cfg.get("tp_correspondence_gt_iou", 0.1)),
        )
        score_x = float(rec["score_x"])
        score_xw = float(corr["score_xw"])
        raw_x = rec.get("raw_score_x")
        raw_xw = corr.get("raw_score_xw")
        gt_id = rec.get("gt_id")
        if gt_id is not None:
            gt = sample.gts[int(gt_id)]
            bw = float(gt.box[2] - gt.box[0])
            bh = float(gt.box[3] - gt.box[1])
        else:
            box = np.asarray(rec["box_x"], dtype=float)
            bw = float(box[2] - box[0])
            bh = float(box[3] - box[1])

        rows.append(
            {
                "image_id": sample.image_id,
                "model": model_name,
                "perturbation": variant["kind"],
                "strength": variant["strength"],
                "intervention_near_far": variant["region"],
                "candidate_type": rec["candidate_type"],
                "gt_id": gt_id,
                "class_id": rec["class_id"],
                "score_x": score_x,
                "score_xw": score_xw,
                "delta_score": score_xw - score_x,
                "abs_delta_score": abs(score_xw - score_x),
                "box_x": rec["box_x"],
                "box_xw": corr["box_xw"],
                "iou_gt_x": rec["iou_gt_x"],
                "iou_gt_xw": corr["iou_gt_xw"],
                "delta_iou": (
                    None
                    if corr["iou_gt_xw"] is None
                    else float(corr["iou_gt_xw"]) - float(rec["iou_gt_x"])
                ),
                "matched_xw": corr["matched_xw"],
                "disappeared": corr["disappeared"],
                "raw_idx": rec.get("raw_idx"),
                "raw_score_x": raw_x,
                "raw_score_xw": raw_xw,
                "pre_nms_delta": (
                    None if raw_x is None or raw_xw is None else float(raw_xw) - float(raw_x)
                ),
                "pre_nms_abs_delta": (
                    None if raw_x is None or raw_xw is None else abs(float(raw_xw) - float(raw_x))
                ),
                "object_width": bw,
                "object_height": bh,
                "object_area": bw * bh,
                "object_short_side": min(bw, bh),
                "strict_fp": rec["candidate_type"] == "strict_background_fp",
                "near_gt_localization_error": rec["candidate_type"]
                == "near_gt_localization_error",
                "high_conf_fp": (
                    rec["candidate_type"] == "strict_background_fp"
                    and score_x >= float(cfg.get("high_conf_threshold", 0.75))
                ),
                "water_fp_label": "",
                "water_fp_label_source": "",
            }
        )


def group_summaries(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["model"],
            row["perturbation"],
            float(row["strength"]),
            row["intervention_near_far"],
        )
        groups[key].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda x: str(x[0])):
        summary = summarize_rows(group)
        summary.update(
            {
                "model": key[0],
                "perturbation": key[1],
                "strength": key[2],
                "region": key[3],
            }
        )
        if key[1] != "null":
            summary["bootstrap"] = bootstrap_image_level(
                group,
                n_boot=int(cfg.get("bootstrap_reps", 500)),
                seed=stable_seed(int(cfg.get("probe_seed", 3407)), *key),
            )
        out.append(summary)
    return out


def null_valid(summary_rows: list[dict[str, Any]], tol: float) -> tuple[bool, list[str]]:
    problems: list[str] = []
    for s in summary_rows:
        if s["perturbation"] != "null":
            continue
        for field in ("median_abs_delta_tp", "median_abs_delta_strict_fp"):
            value = s.get(field)
            if value is not None and float(value) > tol:
                problems.append(f"{s['model']} {field}={value} > {tol}")
    return not problems, problems


def aggregate_non_null(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    subset = [r for r in rows if r["model"] == model and r["perturbation"] != "null"]
    return summarize_rows(subset)


def make_fp_review(rows: list[dict[str, Any]], out_dir: Path, top_k: int = 100) -> None:
    candidates = [
        r
        for r in rows
        if r["candidate_type"] == "strict_background_fp" and r["perturbation"] != "null"
    ]
    candidates.sort(key=lambda r: float(r["abs_delta_score"]), reverse=True)
    review: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for r in candidates:
        key = (str(r["image_id"]), str(r["model"]), str(r["box_x"]))
        if key in seen:
            continue
        seen.add(key)
        review.append(
            {
                "image_id": r["image_id"],
                "model": r["model"],
                "class_id": r["class_id"],
                "box_x": r["box_x"],
                "score_x": r["score_x"],
                "max_abs_delta_score": r["abs_delta_score"],
                "water": "",
                "reflection": "",
                "wave": "",
                "foam": "",
                "shore": "",
                "vegetation_reflection": "",
                "other_background": "",
                "uncertain": "",
                "reviewer_note": "",
            }
        )
        if len(review) >= top_k:
            break
    write_csv(out_dir / "fp_review_manifest.csv", review)


def decision(rows: list[dict[str, Any]], summaries: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    valid, problems = null_valid(summaries, float(cfg.get("null_tolerance", 1e-5)))
    if not valid:
        return {
            "case": "E",
            "decision": "PROBE_INVALID",
            "reason": "Null control failed.",
            "details": problems,
        }

    dbra = aggregate_non_null(rows, "dbra")
    base = aggregate_non_null(rows, "baseline")
    auc = dbra.get("auc_tp_vs_strict_fp")
    ratio = dbra.get("sensitivity_ratio")
    high_auc = dbra.get("auc_tp_vs_high_conf_fp")

    if auc is not None and ratio is not None and auc >= 0.60 and ratio >= 1.5:
        return {
            "case": "B",
            "decision": "NEED_MANUAL_WATER_FP_AUDIT",
            "reason": (
                "Strict-background FP sensitivity supports the nuisance-response hypothesis, "
                "but these FPs are not yet verified as water/reflection/wave errors."
            ),
            "baseline": base,
            "dbra": dbra,
            "dbra_high_conf_auc": high_auc,
        }

    base_auc = base.get("auc_tp_vs_strict_fp")
    base_ratio = base.get("sensitivity_ratio")
    if (
        base_auc is not None
        and base_ratio is not None
        and base_auc >= 0.60
        and base_ratio >= 1.5
        and (auc is None or auc < base_auc)
    ):
        return {
            "case": "C",
            "decision": "DO_NOT_PROCEED_WCR_YET",
            "reason": "Baseline shows stronger nuisance sensitivity than DBRA; DBRA may already suppress it.",
            "baseline": base,
            "dbra": dbra,
        }

    return {
        "case": "D",
        "decision": "DO_NOT_PROCEED_WCR",
        "reason": "Current strict-background sensitivity signal does not pass the preregistered gate.",
        "baseline": base,
        "dbra": dbra,
    }


def write_report(
    run_dir: Path,
    manifest: dict[str, Any],
    decision_obj: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# Baseline vs DBRA Water-Clutter Response Probe",
        "",
        f"**Decision:** `{decision_obj['decision']}`",
        f"**Case:** {decision_obj['case']}",
        "",
        "## 1. Executive conclusion",
        "",
        decision_obj["reason"],
        "",
        "This probe is zero-training and does not establish a new method claim.",
        "Strict background FPs are not automatically treated as water/reflection FPs.",
        "",
        "## 2. Protocol",
        "",
        "```json",
        json.dumps(manifest, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 3. Decision object",
        "",
        "```json",
        json.dumps(decision_obj, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 4. Group summaries",
        "",
        "| model | perturbation | strength | region | n TP | n strict FP | median |Δs| TP | median |Δs| FP | ratio | AUC |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        def fmt(v):
            return "" if v is None else f"{float(v):.6f}"

        lines.append(
            "| {model} | {perturbation} | {strength:g} | {region} | {n_tp} | "
            "{n_strict_fp} | {tp} | {fp} | {ratio} | {auc} |".format(
                model=s["model"],
                perturbation=s["perturbation"],
                strength=float(s["strength"]),
                region=s["region"],
                n_tp=s["n_tp"],
                n_strict_fp=s["n_strict_fp"],
                tp=fmt(s["median_abs_delta_tp"]),
                fp=fmt(s["median_abs_delta_strict_fp"]),
                ratio=fmt(s["sensitivity_ratio"]),
                auc=fmt(s["auc_tp_vs_strict_fp"]),
            )
        )
    lines += [
        "",
        "## 5. Interpretation boundary",
        "",
        "- Do not call every strict FP a water FP.",
        "- Do not use this run to tune DBRA.",
        "- Do not use the project test split for follow-up selection.",
        "- If the strict-FP gate passes, manually audit the top sensitive FPs before implementing WCR.",
        "- If null control fails, invalidate the probe before interpreting any perturbation.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    np.random.seed(int(cfg.get("probe_seed", 3407)))

    run_dir = make_run_dir(cfg)
    source_git = git_info(Path.cwd())
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_checkpoint": str(Path(cfg["baseline_weights"]).resolve()),
        "dbra_checkpoint": str(Path(cfg["dbra_weights"]).resolve()),
        "baseline_model_hash": sha256_file(cfg["baseline_weights"]),
        "dbra_model_hash": sha256_file(cfg["dbra_weights"]),
        "dataset_yaml": str(Path(cfg["dataset_yaml"]).resolve()),
        "split": cfg["split"],
        "input_size": int(cfg["imgsz"]),
        "conf": float(cfg["conf"]),
        "iou_nms": float(cfg["iou_nms"]),
        "device": cfg["device"],
        "probe_seed": int(cfg.get("probe_seed", 3407)),
        "python": sys.version,
        "platform": platform.platform(),
        "ultralytics_version": package_version("ultralytics"),
        "torch_version": package_version("torch"),
        **source_git,
    }
    (run_dir / "probe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if source_git.get("git_status"):
        (run_dir / "git_status.txt").write_text(str(source_git["git_status"]), encoding="utf-8")
    if source_git.get("git_diff_stat"):
        (run_dir / "git_diff_stat.txt").write_text(
            str(source_git["git_diff_stat"]), encoding="utf-8"
        )

    samples = load_dataset(
        cfg["dataset_yaml"],
        str(cfg["split"]),
        max_images=int(cfg.get("max_images", 0)) or None,
    )
    if not samples:
        raise RuntimeError("No images resolved from the configured validation/audit split.")

    adapters = {
        "baseline": UltralyticsYOLOAdapter(
            cfg["baseline_weights"],
            imgsz=int(cfg["imgsz"]),
            conf=float(cfg["conf"]),
            iou=float(cfg["iou_nms"]),
            device=cfg["device"],
            raw_candidate_mode=str(cfg.get("raw_candidate_mode", "auto")),
        ),
        "dbra": UltralyticsYOLOAdapter(
            cfg["dbra_weights"],
            imgsz=int(cfg["imgsz"]),
            conf=float(cfg["conf"]),
            iou=float(cfg["iou_nms"]),
            device=cfg["device"],
            raw_candidate_mode=str(cfg.get("raw_candidate_mode", "auto")),
        ),
    }

    rows: list[dict[str, Any]] = []
    perturb_manifest_path = run_dir / "perturbation_manifest.jsonl"
    sanity_n = int(cfg.get("sanity_images", 20))

    try:
        with perturb_manifest_path.open("w", encoding="utf-8") as perturb_log:
            for sample_idx, sample in enumerate(samples):
                image = np.asarray(Image.open(sample.image_path).convert("RGB"))
                variants, masks = variants_for_image(image, sample, cfg)

                for variant in variants:
                    entry = {
                        "image_id": sample.image_id,
                        "variant": variant["name"],
                        "kind": variant["kind"],
                        "strength": variant["strength"],
                        "region": variant["region"],
                        "seed": variant["seed"],
                        "changed_pixels": int(variant["changed_mask"].sum()),
                        "protected_pixels": int(masks["protected"].sum()),
                        "gt": gt_boxes_payload(sample),
                    }
                    perturb_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    if sample_idx < sanity_n:
                        save_sanity(
                            run_dir / "sanity/intervention_examples",
                            sample.image_id,
                            variant["name"],
                            variant["image"],
                            variant["changed_mask"],
                            masks["protected"],
                        )

                # The exact same in-memory variants are consumed by both models.
                for model_name, adapter in adapters.items():
                    original = adapter.predict(image)
                    original_records = classify_original_predictions(
                        original,
                        sample.gts,
                        tp_iou=float(cfg.get("tp_iou", 0.5)),
                        strict_fp_iou=float(cfg.get("strict_fp_iou", 0.1)),
                    )
                    for variant in variants:
                        pert = adapter.predict(variant["image"])
                        add_response_rows(
                            rows,
                            model_name,
                            sample,
                            original_records,
                            pert,
                            variant,
                            cfg,
                        )
    finally:
        for adapter in adapters.values():
            adapter.close()

    write_csv(run_dir / "analysis/per_detection_response.csv", rows)
    summaries = group_summaries(rows, cfg)
    write_csv(run_dir / "analysis/summary_by_condition.csv", summaries)
    (run_dir / "analysis/summary_by_condition.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_fp_review(
        rows,
        run_dir,
        top_k=int(cfg.get("fp_review_top_k", 100)),
    )
    decision_obj = decision(rows, summaries, cfg)
    (run_dir / "analysis/decision.json").write_text(
        json.dumps(decision_obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(run_dir, manifest, decision_obj, summaries)

    print(f"Probe complete: {run_dir}")
    print(f"Decision: {decision_obj['decision']}")


if __name__ == "__main__":
    main()

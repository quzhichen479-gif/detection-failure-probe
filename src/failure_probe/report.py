"""Standalone HTML report generation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from failure_probe.errors import RunFormatError
from failure_probe.paths import atomic_write_text, validate_run_dir


def generate_report(run_dir: str | Path) -> Path:
    """Generate a self-contained summary report inside a validated run directory."""
    root = validate_run_dir(run_dir)
    audit = _read_generated_json(root / "audit.json")
    analysis_path = root / "analysis.json"
    analysis = _read_generated_json(analysis_path) if analysis_path.is_file() else None
    resolution = _read_generated_json(root / "resolution.json")
    markup = _render_report(audit, analysis, resolution)
    output = root / "report.html"
    atomic_write_text(output, markup)
    return output


def _read_generated_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RunFormatError(f"Missing or unsafe run artifact: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunFormatError(f"Invalid run artifact {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RunFormatError(f"Run artifact must contain an object: {path.name}")
    return payload


def _render_report(
    audit: dict[str, Any],
    analysis: dict[str, Any] | None,
    resolution: dict[str, Any],
) -> str:
    audit_summary = audit["summary"]
    cards = [
        ("Images", audit_summary["images"]),
        ("Valid boxes", audit_summary["valid_annotations"]),
        ("Invalid boxes", audit_summary["invalid_annotations"]),
        ("Missing labels", audit_summary["missing_labels"]),
        ("Empty labels", audit_summary["empty_label_images"]),
        ("Small-object ratio", f"{audit_summary['small_object_ratio']:.1%}"),
    ]
    if analysis:
        summary = analysis["summary"]
        cards.extend([("TP", summary["tp"]), ("FP", summary["fp"]), ("FN", summary["fn"])])
    class_rows = "".join(
        _row(
            item["class_id"],
            item["name"],
            item["annotations"],
            item["images"],
        )
        for item in audit["class_distribution"]
    )
    issue_rows = "".join(
        _row(issue["type"], issue["severity"], issue.get("image") or "—", issue["message"])
        for issue in audit["issues"][:100]
    ) or _row("No issues", "—", "—", "—")
    resolution_rows = "".join(
        _row(
            item["resolution"],
            item["valid_objects"],
            _format_number(item["mean_pixel_width"]),
            _format_number(item["mean_pixel_height"]),
            f"{item['survival']['min_side_ge_4px']['ratio']:.1%}",
        )
        for item in resolution["resolutions"]
    )
    failure_section = ""
    if analysis:
        failures = analysis["summary"]["failure_types"]
        failure_rows = "".join(_row(name, count) for name, count in failures.items()) or _row(
            "No prediction failures", 0
        )
        failure_section = f"""
        <section><h2>Prediction failures</h2>
        <table><thead><tr><th>Type</th><th>Count</th></tr></thead><tbody>{failure_rows}</tbody></table>
        <p class="muted">Matching: {html.escape(analysis['method']['matching'])}; IoU threshold
        {analysis['method']['match_iou']}. Classification/localization errors can contribute one FP
        and one FN.</p></section>"""
    cards_html = "".join(
        f'<div class="card"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in cards
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Detection Failure Probe report</title>
<style>
:root{{--bg:#0b1020;--panel:#151c31;--ink:#edf2ff;--muted:#9ba8c7;--accent:#64d2ff;--line:#2a3555}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:40px 20px}}h1{{margin-bottom:4px}}h2{{margin-top:34px}}
.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px}}
.card,section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px}}
.card span{{display:block;color:var(--muted)}}.card strong{{font-size:1.7rem;color:var(--accent)}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;text-align:left;border-bottom:1px solid var(--line)}}
th{{color:var(--muted)}}code{{color:var(--accent)}}
</style></head><body><main>
<h1>Detection Failure Probe</h1><p class="muted">Local dataset and prediction diagnostic report</p>
<div class="cards">{cards_html}</div>
<section><h2>Class distribution</h2><table><thead><tr><th>ID</th><th>Class</th><th>Boxes</th><th>Images</th></tr></thead><tbody>{class_rows}</tbody></table></section>
{failure_section}
<section><h2>Resolution survival</h2><table><thead><tr><th>Input</th><th>Objects</th><th>Mean width</th><th>Mean height</th><th>Min side ≥ 4 px</th></tr></thead><tbody>{resolution_rows}</tbody></table>
<p class="muted">{html.escape(resolution['disclaimer'])}</p></section>
<section><h2>Audit findings</h2><table><thead><tr><th>Type</th><th>Severity</th><th>Image</th><th>Detail</th></tr></thead><tbody>{issue_rows}</tbody></table>
<p class="muted">Showing at most 100 findings. Use <code>audit.json</code> for complete data and <code>failure-probe review</code> for image review.</p></section>
</main></body></html>"""


def _row(*values: Any) -> str:
    return "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>"


def _format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f} px"

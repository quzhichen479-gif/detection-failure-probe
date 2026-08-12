"""Command-line interface for Detection Failure Probe."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from failure_probe.errors import FailureProbeError
from failure_probe.report import generate_report
from failure_probe.review import serve_review
from failure_probe.workflow import run_analysis, run_audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="failure-probe",
        description="Local-first audits and failure analysis for YOLO object detection.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0a0")
    subcommands = parser.add_subparsers(dest="command", required=True)

    audit = subcommands.add_parser("audit", help="Audit a YOLO dataset")
    audit.add_argument("dataset", type=Path, help="Path to dataset.yaml")
    _add_run_options(audit)

    analyze = subcommands.add_parser("analyze", help="Analyze prediction failures")
    analyze.add_argument("--dataset", required=True, type=Path, help="Path to dataset.yaml")
    analyze.add_argument("--predictions", required=True, type=Path, help="Prediction JSON")
    analyze.add_argument("--match-iou", type=float, default=0.5)
    analyze.add_argument("--localization-iou", type=float, default=0.1)
    _add_run_options(analyze)

    review = subcommands.add_parser("review", help="Open the local visual review UI")
    review.add_argument("run_dir", type=Path)
    review.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost", "::1"])
    review.add_argument("--port", type=int, default=8765)
    review.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser automatically",
    )

    report = subcommands.add_parser("report", help="Regenerate the static HTML report")
    report.add_argument("run_dir", type=Path)
    return parser


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", help="Safe optional run directory name")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[320, 640, 1280])


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            run_dir = run_audit(
                args.dataset,
                runs_dir=args.runs_dir,
                run_name=args.run_name,
                resolutions=args.resolutions,
            )
            _print_run(run_dir)
        elif args.command == "analyze":
            run_dir = run_analysis(
                args.dataset,
                args.predictions,
                runs_dir=args.runs_dir,
                run_name=args.run_name,
                resolutions=args.resolutions,
                match_iou=args.match_iou,
                localization_iou=args.localization_iou,
            )
            _print_run(run_dir)
        elif args.command == "review":
            if not 0 <= args.port <= 65535:
                raise ValueError("port must be between 0 and 65535")
            serve_review(
                args.run_dir,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
            )
        elif args.command == "report":
            output = generate_report(args.run_dir)
            print(f"Report: {output}")
        return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except (FailureProbeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _print_run(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((run_dir / "audit.json").read_text(encoding="utf-8"))
    print(f"Created {manifest['run_type']} run: {run_dir}")
    print(
        f"Images: {audit['summary']['images']} | "
        f"valid boxes: {audit['summary']['valid_annotations']} | "
        f"issues: {len(audit['issues'])}"
    )
    analysis_path = run_dir / "analysis.json"
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        summary = analysis["summary"]
        print(f"TP: {summary['tp']} | FP: {summary['fp']} | FN: {summary['fn']}")
    print(f"Report: {run_dir / 'report.html'}")
    print(f"Review: failure-probe review \"{run_dir}\"")


if __name__ == "__main__":
    raise SystemExit(main())

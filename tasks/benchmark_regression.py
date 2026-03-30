"""Benchmark regression checker for tracker performance reports.

Usage:
  uv run python tasks/benchmark_regression.py \
    --current docs/media/benchmark.csv \
    --baseline docs/media/benchmark.csv \
    --max-fps-regression-pct 15 \
    --max-p95-increase-pct 20
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Thresholds:
    max_fps_regression_pct: float
    max_p95_increase_pct: float
    max_median_increase_pct: float


@dataclass(frozen=True)
class RowMetrics:
    tracker: str
    throughput_fps: float
    p95_latency_ms: float
    median_latency_ms: float


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(value: str | None, *, default: float = 0.0) -> float:
    if value is None:
        return default
    value = value.strip()
    if value == "":
        return default
    return float(value)


def _normalize_tracker_name(row: dict[str, Any]) -> str:
    for key in ("tracker", "tracker_name", "tracker_label"):
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    return "unknown"


def _extract_metrics(row: dict[str, Any]) -> RowMetrics:
    return RowMetrics(
        tracker=_normalize_tracker_name(row),
        throughput_fps=_to_float(
            row.get("throughput_fps") or row.get("avg_fps") or row.get("effective_fps")
        ),
        p95_latency_ms=_to_float(
            row.get("p95_latency_ms") or row.get("latency_p95_ms") or row.get("p95_ms")
        ),
        median_latency_ms=_to_float(
            row.get("median_latency_ms")
            or row.get("latency_median_ms")
            or row.get("median_ms")
        ),
    )


def _index_by_tracker(rows: list[dict[str, Any]]) -> dict[str, RowMetrics]:
    indexed: dict[str, RowMetrics] = {}
    for row in rows:
        metrics = _extract_metrics(row)
        indexed[metrics.tracker] = metrics
    return indexed


def _pct_change(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else float("inf")
    return ((new - old) / old) * 100.0


def _evaluate_tracker(
    *,
    tracker: str,
    current: RowMetrics,
    baseline: RowMetrics,
    thresholds: Thresholds,
) -> list[str]:
    failures: list[str] = []

    fps_change = _pct_change(current.throughput_fps, baseline.throughput_fps)
    if fps_change < 0 and abs(fps_change) > thresholds.max_fps_regression_pct:
        failures.append(
            f"{tracker}: throughput_fps regressed by {abs(fps_change):.2f}% "
            f"(threshold {thresholds.max_fps_regression_pct:.2f}%)"
        )

    p95_change = _pct_change(current.p95_latency_ms, baseline.p95_latency_ms)
    if p95_change > thresholds.max_p95_increase_pct:
        failures.append(
            f"{tracker}: p95_latency_ms increased by {p95_change:.2f}% "
            f"(threshold {thresholds.max_p95_increase_pct:.2f}%)"
        )

    median_change = _pct_change(current.median_latency_ms, baseline.median_latency_ms)
    if median_change > thresholds.max_median_increase_pct:
        failures.append(
            f"{tracker}: median_latency_ms increased by {median_change:.2f}% "
            f"(threshold {thresholds.max_median_increase_pct:.2f}%)"
        )

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check benchmark regressions against a baseline CSV"
    )
    parser.add_argument("--current", required=True, help="Path to current benchmark CSV")
    parser.add_argument("--baseline", required=True, help="Path to baseline benchmark CSV")
    parser.add_argument(
        "--max-fps-regression-pct",
        type=float,
        default=15.0,
        help="Allowed throughput FPS drop percentage before failure",
    )
    parser.add_argument(
        "--max-p95-increase-pct",
        type=float,
        default=20.0,
        help="Allowed p95 latency increase percentage before failure",
    )
    parser.add_argument(
        "--max-median-increase-pct",
        type=float,
        default=20.0,
        help="Allowed median latency increase percentage before failure",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_path = Path(args.current)
    baseline_path = Path(args.baseline)

    if not current_path.exists():
        raise SystemExit(f"Current benchmark file does not exist: {current_path}")
    if not baseline_path.exists():
        raise SystemExit(f"Baseline benchmark file does not exist: {baseline_path}")

    thresholds = Thresholds(
        max_fps_regression_pct=args.max_fps_regression_pct,
        max_p95_increase_pct=args.max_p95_increase_pct,
        max_median_increase_pct=args.max_median_increase_pct,
    )

    current_rows = _index_by_tracker(_load_rows(current_path))
    baseline_rows = _index_by_tracker(_load_rows(baseline_path))

    failures: list[str] = []
    checked: list[str] = []

    for tracker, current in current_rows.items():
        baseline = baseline_rows.get(tracker)
        if baseline is None:
            continue
        checked.append(tracker)
        failures.extend(
            _evaluate_tracker(
                tracker=tracker,
                current=current,
                baseline=baseline,
                thresholds=thresholds,
            )
        )

    summary = {
        "current": str(current_path),
        "baseline": str(baseline_path),
        "checked_trackers": checked,
        "failure_count": len(failures),
        "failures": failures,
        "thresholds": {
            "max_fps_regression_pct": thresholds.max_fps_regression_pct,
            "max_p95_increase_pct": thresholds.max_p95_increase_pct,
            "max_median_increase_pct": thresholds.max_median_increase_pct,
        },
    }

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"Checked trackers: {', '.join(checked) if checked else '(none)'}")
        if failures:
            print("Regression failures:")
            for failure in failures:
                print(f"  - {failure}")
        else:
            print("No regression failures detected.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

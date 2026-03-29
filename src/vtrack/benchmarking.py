"""Tracking benchmark workflow and report export helpers."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np

from vtrack.analytics import VehicleAnalytics
from vtrack.config import DEFAULT_BENCHMARK_TRACKERS
from vtrack.settings import InferenceConfig
from vtrack.track import VehicleTracker
from vtrack.visualize import filter_detections_by_confidence, ultralytics_to_detections


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def _benchmark_run(
    *,
    source: str | Path | int,
    inference: InferenceConfig,
    tracker_name: str,
    analytics: VehicleAnalytics,
    max_frames: int | None,
    warmup_frames: int,
) -> dict[str, Any]:
    tracker = VehicleTracker(
        model_path=inference.model_path,
        track_conf=inference.track_conf,
        tracker=tracker_name,
        device=inference.device,
        imgsz=inference.imgsz,
        iou=inference.iou,
        max_det=inference.max_det,
        half=inference.half,
        vid_stride=inference.vid_stride,
        stream_buffer=inference.stream_buffer,
        agnostic_nms=inference.agnostic_nms,
    )
    analytics.class_names = tracker.class_names

    print(f"Benchmarking tracker: {tracker.describe_tracker()}")

    frame_latencies_ms: list[float] = []
    frames_processed = 0
    iterator = iter(tracker.track(source))
    wall_start = time.perf_counter()

    while max_frames is None or frames_processed < max_frames:
        frame_start = time.perf_counter()
        try:
            result = next(iterator)
        except StopIteration:
            break

        detections = filter_detections_by_confidence(
            ultralytics_to_detections(result),
            inference.min_confidence,
        )
        analytics.update(detections)
        frames_processed += 1

        frame_ms = (time.perf_counter() - frame_start) * 1000.0
        if frames_processed > warmup_frames:
            frame_latencies_ms.append(frame_ms)

    wall_time_sec = time.perf_counter() - wall_start
    durations = [track.duration_frames for track in analytics.tracks.values()]
    analytics_summary = analytics.get_summary()
    instantaneous_fps = [1000.0 / frame_ms for frame_ms in frame_latencies_ms if frame_ms > 0]
    tracker_label = (
        tracker.resolved_tracker.name if tracker.resolved_tracker.is_builtin else tracker_name
    )

    run_report: dict[str, Any] = {
        "tracker": tracker_label,
        "tracker_path": tracker.resolved_tracker.path,
        "requested_tracker": tracker.resolved_tracker.requested,
        "device": inference.device or "auto",
        "imgsz": inference.imgsz,
        "vid_stride": inference.vid_stride,
        "frames_processed": frames_processed,
        "timed_frames": len(frame_latencies_ms),
        "wall_time_sec": round(wall_time_sec, 6),
        "avg_fps": round(len(frame_latencies_ms) / (sum(frame_latencies_ms) / 1000.0), 3)
        if frame_latencies_ms
        else 0.0,
        "median_fps": round(float(median(instantaneous_fps)), 3) if instantaneous_fps else 0.0,
        "p95_frame_ms": round(_p95(frame_latencies_ms), 3),
        "unique_tracks": analytics_summary["unique_vehicles"],
        "avg_track_duration_frames": round(analytics_summary["avg_track_duration_frames"], 3),
        "median_track_duration_frames": round(float(median(durations)), 3) if durations else 0.0,
        "short_tracks_lt_5_frames": sum(duration < 5 for duration in durations),
        "short_tracks_lt_15_frames": sum(duration < 15 for duration in durations),
        "per_class_counts": analytics_summary["per_class_counts"],
    }

    if analytics.line_zone is not None or analytics.polygon_zone is not None:
        run_report.update(
            {
                "line_crossings_in": analytics_summary["line_crossings_in"],
                "line_crossings_out": analytics_summary["line_crossings_out"],
                "zone_current_count": analytics_summary["zone_current_count"],
            }
        )

    return run_report


def export_benchmark_csv(report: dict[str, Any], path: str | Path) -> None:
    """Write one summary CSV row per tracker benchmark run."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for run in report["runs"]:
        row = dict(run)
        row["per_class_counts"] = json.dumps(run["per_class_counts"], sort_keys=True)
        rows.append(row)

    if not rows:
        return

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark CSV exported to {output_path}")


def benchmark_trackers(
    *,
    source: str | Path | int,
    inference: InferenceConfig,
    trackers: list[str] | None = None,
    analytics_factory: Callable[[], VehicleAnalytics] | None = None,
    max_frames: int | None = None,
    warmup_frames: int = 30,
    export_csv: str | None = None,
) -> dict[str, Any]:
    """Benchmark one or more tracker presets on a shared source."""
    if warmup_frames < 0:
        raise ValueError("--warmup-frames must be zero or positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive when provided.")

    tracker_list = list(trackers) if trackers else list(DEFAULT_BENCHMARK_TRACKERS)
    analytics_factory = analytics_factory or VehicleAnalytics

    report = {
        "source": str(source),
        "model_path": inference.model_path,
        "min_confidence": inference.min_confidence,
        "track_conf": inference.track_conf,
        "warmup_frames": warmup_frames,
        "max_frames": max_frames,
        "runs": [],
    }

    for tracker_name in tracker_list:
        analytics = analytics_factory()
        report["runs"].append(
            _benchmark_run(
                source=source,
                inference=inference,
                tracker_name=tracker_name,
                analytics=analytics,
                max_frames=max_frames,
                warmup_frames=warmup_frames,
            )
        )

    if export_csv:
        export_benchmark_csv(report, export_csv)

    return report

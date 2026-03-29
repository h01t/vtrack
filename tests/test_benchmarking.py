import csv
from types import SimpleNamespace

import numpy as np
import supervision as sv

from vtrack.analytics import VehicleAnalytics
from vtrack.benchmarking import benchmark_trackers
from vtrack.settings import InferenceConfig


class FakeVehicleTracker:
    def __init__(self, *args, tracker: str, **kwargs):
        self.class_names = {0: "car"}
        self.resolved_tracker = SimpleNamespace(
            name=tracker,
            path=f"/tmp/{tracker}.yaml",
            requested=tracker,
            is_builtin=True,
        )

    def describe_tracker(self) -> str:
        return f"{self.resolved_tracker.name} ({self.resolved_tracker.path})"

    def track(self, source):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        yield SimpleNamespace(orig_img=frame)
        yield SimpleNamespace(orig_img=frame)


def test_benchmark_trackers_reports_expected_metrics(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("vtrack.benchmarking.VehicleTracker", FakeVehicleTracker)
    monkeypatch.setattr(
        "vtrack.benchmarking.ultralytics_to_detections",
        lambda result: sv.Detections(
            xyxy=np.array([[0, 0, 4, 4], [5, 5, 9, 9]], dtype=np.float32),
            confidence=np.array([0.2, 0.9], dtype=np.float32),
            class_id=np.array([0, 0], dtype=np.int32),
            tracker_id=np.array([10, 11], dtype=np.int32),
        ),
    )

    def analytics_factory() -> VehicleAnalytics:
        return VehicleAnalytics(
            line_zone=sv.LineZone(start=sv.Point(0, 2), end=sv.Point(9, 2)),
            class_names={0: "car"},
        )

    csv_path = tmp_path / "benchmark.csv"
    report = benchmark_trackers(
        source="data/test-video.mp4",
        inference=InferenceConfig(
            model_path="models/best.pt",
            min_confidence=0.25,
            track_conf=0.1,
            device="cpu",
        ),
        trackers=["bytetrack", "botsort"],
        analytics_factory=analytics_factory,
        max_frames=2,
        warmup_frames=0,
        export_csv=str(csv_path),
    )

    assert len(report["runs"]) == 2
    first_run = report["runs"][0]
    assert first_run["tracker"] == "bytetrack"
    assert first_run["frames_processed"] == 2
    assert first_run["timed_frames"] == 2
    assert first_run["unique_tracks"] == 1
    assert first_run["short_tracks_lt_5_frames"] == 1
    assert "line_crossings_in" in first_run
    assert "line_crossings_out" in first_run
    assert "per_class_counts" in first_run

    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["tracker"] == "bytetrack"
    assert rows[1]["tracker"] == "botsort"

from types import SimpleNamespace

import numpy as np
import supervision as sv

from vtrack.analytics import VehicleAnalytics
from vtrack.pipeline import VehiclePipeline


class FakeTracker:
    def __init__(self, *args, **kwargs):
        self.class_names = {0: "car"}


class FakeVisualizer:
    def __init__(self, *args, **kwargs):
        pass


class StreamingFakeTracker(FakeTracker):
    def describe_tracker(self) -> str:
        return "fake-tracker"

    def track(self, source):
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        yield SimpleNamespace(orig_img=frame)


class CapturingVisualizer(FakeVisualizer):
    captured_lengths: list[int] = []

    def annotate(self, frame, detections):
        type(self).captured_lengths.append(len(detections))
        return frame


def test_pipeline_builds_polygon_zone_annotator_from_zone(monkeypatch) -> None:
    monkeypatch.setattr("vtrack.pipeline.VehicleTracker", FakeTracker)
    monkeypatch.setattr("vtrack.pipeline.Visualizer", FakeVisualizer)

    analytics = VehicleAnalytics(
        line_zone=sv.LineZone(start=sv.Point(0, 0), end=sv.Point(10, 0)),
        polygon_zone=sv.PolygonZone(
            polygon=np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int64)
        ),
    )

    pipeline = VehiclePipeline(model_path="models/best.pt", analytics=analytics)

    assert pipeline.line_zone_annotator is not None
    assert pipeline.polygon_zone_annotator is not None

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    annotated = pipeline.polygon_zone_annotator.annotate(frame)
    assert annotated.shape == frame.shape


def test_pipeline_filters_low_confidence_detections_before_analytics_and_overlay(
    monkeypatch,
) -> None:
    monkeypatch.setattr("vtrack.pipeline.VehicleTracker", StreamingFakeTracker)
    monkeypatch.setattr("vtrack.pipeline.Visualizer", CapturingVisualizer)
    monkeypatch.setattr(
        "vtrack.pipeline.ultralytics_to_detections",
        lambda result: sv.Detections(
            xyxy=np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32),
            confidence=np.array([0.2, 0.9], dtype=np.float32),
            class_id=np.array([0, 0], dtype=np.int32),
            tracker_id=np.array([1, 2], dtype=np.int32),
        ),
    )

    CapturingVisualizer.captured_lengths = []
    analytics = VehicleAnalytics(class_names={0: "car"})
    pipeline = VehiclePipeline(
        model_path="models/best.pt",
        confidence=0.25,
        analytics=analytics,
    )

    pipeline.run("data/test-video.mp4", display=False)

    assert CapturingVisualizer.captured_lengths == [1]
    assert analytics.frame_log[0]["detections"] == 1
    assert len(analytics.tracks) == 1

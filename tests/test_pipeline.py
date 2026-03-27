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

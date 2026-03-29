from pathlib import Path

import pytest

from vtrack.track import VehicleTracker


class FakeYOLO:
    last_kwargs = None

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.names = {0: "car"}

    def track(self, **kwargs):
        type(self).last_kwargs = kwargs
        return iter(())


def test_vehicle_tracker_passes_track_conf_and_runtime_kwargs(monkeypatch) -> None:
    monkeypatch.setattr("vtrack.track.YOLO", FakeYOLO)

    tracker = VehicleTracker(
        model_path="models/best.pt",
        track_conf=0.1,
        tracker="bytetrack",
        device="cpu",
        imgsz=960,
        iou=0.6,
        max_det=99,
        half=True,
        vid_stride=2,
        stream_buffer=True,
        agnostic_nms=True,
    )
    list(tracker.track("data/test-video.mp4"))

    assert FakeYOLO.last_kwargs is not None
    assert FakeYOLO.last_kwargs["conf"] == pytest.approx(0.1)
    assert FakeYOLO.last_kwargs["device"] == "cpu"
    assert FakeYOLO.last_kwargs["imgsz"] == 960
    assert FakeYOLO.last_kwargs["iou"] == pytest.approx(0.6)
    assert FakeYOLO.last_kwargs["max_det"] == 99
    assert FakeYOLO.last_kwargs["half"] is True
    assert FakeYOLO.last_kwargs["vid_stride"] == 2
    assert FakeYOLO.last_kwargs["stream_buffer"] is True
    assert FakeYOLO.last_kwargs["agnostic_nms"] is True
    assert Path(FakeYOLO.last_kwargs["tracker"]).name == "bytetrack.yaml"
    assert tracker.describe_tracker().startswith("bytetrack")

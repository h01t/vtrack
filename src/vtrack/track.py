"""Multi-object vehicle tracking using Ultralytics built-in trackers."""

from pathlib import Path

from ultralytics import YOLO

from vtrack.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL, get_class_config


class VehicleTracker:
    """Track vehicles across video frames with persistent IDs."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        tracker: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes, self.class_names = get_class_config(model_path)
        self.tracker_config = tracker

    def track(self, source: str | Path | int, stream: bool = True):
        """Track vehicles in a video source. Yields results per frame.

        Args:
            source: Video file path, camera index (0), RTSP URL, or YouTube URL.
            stream: If True, yields results frame-by-frame (memory efficient).
        """
        results = self.model.track(
            source=source,
            conf=self.confidence,
            classes=self.vehicle_classes,
            tracker=self.tracker_config,
            stream=stream,
            persist=True,
            verbose=False,
        )
        return results

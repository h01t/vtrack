"""Vehicle detection module wrapping Ultralytics YOLOv11."""

from pathlib import Path

from ultralytics import YOLO

from vtrack.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL, VEHICLE_CLASSES


class VehicleDetector:
    """Detect vehicles in images and video using YOLOv11."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.vehicle_classes = VEHICLE_CLASSES

    def detect_image(self, source: str | Path, save: bool = True):
        """Run detection on a single image."""
        results = self.model(
            source,
            conf=self.confidence,
            classes=self.vehicle_classes,
            save=save,
        )
        return results

    def detect_video(self, source: str | Path, save: bool = True, stream: bool = True):
        """Run detection on a video file or stream."""
        results = self.model(
            source,
            conf=self.confidence,
            classes=self.vehicle_classes,
            save=save,
            stream=stream,
        )
        return results

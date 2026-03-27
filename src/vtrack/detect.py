"""Vehicle detection module wrapping Ultralytics YOLOv11."""

from pathlib import Path

from ultralytics import YOLO

from vtrack.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL
from vtrack.model_profiles import resolve_model_profile


class VehicleDetector:
    """Detect vehicles in images and video using YOLOv11."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.profile = resolve_model_profile(self.model, source=model_path)
        self.vehicle_classes = self.profile.class_filter
        self.class_names = self.profile.class_names

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

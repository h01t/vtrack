"""Multi-object vehicle tracking using Ultralytics built-in trackers."""

from pathlib import Path

from ultralytics import YOLO

from vtrack.config import DEFAULT_MODEL, DEFAULT_TRACK_CONFIDENCE, DEFAULT_TRACKER
from vtrack.model_profiles import resolve_model_profile
from vtrack.settings import InferenceConfig, validate_inference_device
from vtrack.tracker_presets import ResolvedTrackerConfig, resolve_tracker_config


class VehicleTracker:
    """Track vehicles across video frames with persistent IDs."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float | None = None,
        track_conf: float | None = None,
        tracker: str = DEFAULT_TRACKER,
        device: str | None = None,
        imgsz: int = InferenceConfig().imgsz,
        iou: float = InferenceConfig().iou,
        max_det: int = InferenceConfig().max_det,
        half: bool = InferenceConfig().half,
        vid_stride: int = InferenceConfig().vid_stride,
        stream_buffer: bool = InferenceConfig().stream_buffer,
        agnostic_nms: bool = InferenceConfig().agnostic_nms,
    ):
        validate_inference_device(device)
        self.model = YOLO(model_path)
        self.track_conf = (
            track_conf
            if track_conf is not None
            else confidence
            if confidence is not None
            else DEFAULT_TRACK_CONFIDENCE
        )
        self.profile = resolve_model_profile(self.model, source=model_path)
        self.vehicle_classes = self.profile.class_filter
        self.class_names = self.profile.class_names
        self.resolved_tracker: ResolvedTrackerConfig = resolve_tracker_config(tracker)
        self.tracker_config = self.resolved_tracker.path
        self.device = device
        self.imgsz = imgsz
        self.iou = iou
        self.max_det = max_det
        self.half = half
        self.vid_stride = vid_stride
        self.stream_buffer = stream_buffer
        self.agnostic_nms = agnostic_nms

    def describe_tracker(self) -> str:
        return self.resolved_tracker.description

    def track(self, source: str | Path | int, stream: bool = True):
        """Track vehicles in a video source. Yields results per frame.

        Args:
            source: Video file path, camera index (0), RTSP URL, or YouTube URL.
            stream: If True, yields results frame-by-frame (memory efficient).
        """
        kwargs: dict[str, object] = {
            "source": source,
            "classes": self.vehicle_classes,
            "tracker": self.tracker_config,
            "stream": stream,
            "persist": True,
            "verbose": False,
            "conf": self.track_conf,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "max_det": self.max_det,
            "half": self.half,
            "vid_stride": self.vid_stride,
            "stream_buffer": self.stream_buffer,
            "agnostic_nms": self.agnostic_nms,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.track(**kwargs)
        return results

"""Vehicle detection module wrapping Ultralytics YOLOv11."""

from pathlib import Path

from ultralytics import YOLO

from vtrack.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL
from vtrack.errors import ModelLoadError, PipelineRuntimeError
from vtrack.logging_utils import build_log_context, get_logger, log_event
from vtrack.model_profiles import resolve_model_profile
from vtrack.settings import InferenceConfig, validate_inference_device

LOGGER = get_logger(__name__)


class VehicleDetector:
    """Detect vehicles in images and video using YOLOv11."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str | None = None,
        imgsz: int = InferenceConfig().imgsz,
        iou: float = InferenceConfig().iou,
        max_det: int = InferenceConfig().max_det,
        half: bool = InferenceConfig().half,
        agnostic_nms: bool = InferenceConfig().agnostic_nms,
    ):
        validate_inference_device(device)
        try:
            self.model = YOLO(model_path)
            self.profile = resolve_model_profile(self.model, source=model_path)
        except Exception as exc:  # pragma: no cover - depends on ultralytics runtime errors
            log_event(
                LOGGER,
                40,
                "failed to initialize detector model",
                build_log_context(
                    event="detector_init_failed",
                    model=model_path,
                    device=device,
                    error_type=type(exc).__name__,
                ),
            )
            raise ModelLoadError(f"Failed to initialize detector model: {model_path}") from exc

        self.confidence = confidence
        self.vehicle_classes = self.profile.class_filter
        self.class_names = self.profile.class_names
        self.device = device
        self.imgsz = imgsz
        self.iou = iou
        self.max_det = max_det
        self.half = half
        self.agnostic_nms = agnostic_nms

    def _predict_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "conf": self.confidence,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "max_det": self.max_det,
            "half": self.half,
            "agnostic_nms": self.agnostic_nms,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def detect_image(self, source: str | Path, save: bool = True):
        """Run detection on a single image."""
        try:
            results = self.model(
                source,
                classes=self.vehicle_classes,
                save=save,
                **self._predict_kwargs(),
            )
        except Exception as exc:  # pragma: no cover - depends on ultralytics runtime errors
            log_event(
                LOGGER,
                40,
                "image detection failed",
                build_log_context(
                    event="detect_image_failed",
                    source=str(source),
                    model=getattr(self.model, "ckpt_path", None) or DEFAULT_MODEL,
                    device=self.device,
                    error_type=type(exc).__name__,
                ),
            )
            raise PipelineRuntimeError(f"Image detection failed for source: {source}") from exc
        return results

    def detect_video(self, source: str | Path, save: bool = True, stream: bool = True):
        """Run detection on a video file or stream."""
        try:
            results = self.model(
                source,
                classes=self.vehicle_classes,
                save=save,
                stream=stream,
                **self._predict_kwargs(),
            )
        except Exception as exc:  # pragma: no cover - depends on ultralytics runtime errors
            log_event(
                LOGGER,
                40,
                "video detection failed",
                build_log_context(
                    event="detect_video_failed",
                    source=str(source),
                    model=getattr(self.model, "ckpt_path", None) or DEFAULT_MODEL,
                    device=self.device,
                    error_type=type(exc).__name__,
                ),
            )
            raise PipelineRuntimeError(f"Video detection failed for source: {source}") from exc
        return results

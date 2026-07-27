"""Localhost FastAPI inference service (detect + session track)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vtrack.settings import InferenceConfig, validate_inference_device

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
except ImportError:  # pragma: no cover - exercised when api extra missing
    FastAPI = None  # type: ignore[assignment,misc]
    File = Form = HTTPException = UploadFile = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class DetectionBox:
    xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None


class InferenceBackend(Protocol):
    """Injectable backend so CI can exercise the API without loading weights."""

    model_path: str
    device: str | None

    def detect(self, frame: np.ndarray) -> list[DetectionBox]:
        ...

    def track(self, frame: np.ndarray, *, session_id: str) -> list[DetectionBox]:
        ...


class UltralyticsBackend:
    """YOLO detect + per-session track(persist=True) state."""

    def __init__(self, inference: InferenceConfig, *, max_sessions: int = 8):
        from ultralytics import YOLO

        from vtrack.model_profiles import resolve_model_profile
        from vtrack.tracker_presets import resolve_tracker_config

        validate_inference_device(inference.device)
        self.inference = inference
        self.model_path = inference.model_path
        self.device = inference.device
        self.max_sessions = max_sessions
        self._detect_model = YOLO(inference.model_path)
        self._profile = resolve_model_profile(self._detect_model, source=inference.model_path)
        self._tracker_path = resolve_tracker_config(inference.tracker).path
        self._sessions: OrderedDict[str, Any] = OrderedDict()

    def _predict_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "conf": self.inference.min_confidence,
            "imgsz": self.inference.imgsz,
            "iou": self.inference.iou,
            "max_det": self.inference.max_det,
            "half": self.inference.half,
            "agnostic_nms": self.inference.agnostic_nms,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        if self._profile.class_filter is not None:
            kwargs["classes"] = self._profile.class_filter
        return kwargs

    def _boxes_from_result(self, result: Any, *, with_tracks: bool) -> list[DetectionBox]:
        names = result.names or self._profile.class_names
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        out: list[DetectionBox] = []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros(len(boxes))
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(boxes))
        track_ids = None
        if with_tracks and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy()

        for index in range(len(boxes)):
            class_id = int(clss[index])
            track_id = int(track_ids[index]) if track_ids is not None else None
            out.append(
                DetectionBox(
                    xyxy=[float(v) for v in xyxy[index].tolist()],
                    confidence=float(confs[index]),
                    class_id=class_id,
                    class_name=str(names.get(class_id, f"cls_{class_id}")),
                    track_id=track_id,
                )
            )
        return out

    def detect(self, frame: np.ndarray) -> list[DetectionBox]:
        results = self._detect_model.predict(source=frame, **self._predict_kwargs())
        return self._boxes_from_result(results[0], with_tracks=False)

    def _session_model(self, session_id: str) -> Any:
        from ultralytics import YOLO

        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]

        while len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)

        model = YOLO(self.model_path)
        self._sessions[session_id] = model
        return model

    def track(self, frame: np.ndarray, *, session_id: str) -> list[DetectionBox]:
        model = self._session_model(session_id)
        kwargs = self._predict_kwargs()
        kwargs.update(
            {
                "tracker": self._tracker_path,
                "persist": True,
            }
        )
        results = model.track(source=frame, **kwargs)
        return self._boxes_from_result(results[0], with_tracks=True)


def create_app(
    *,
    backend: InferenceBackend | None = None,
    inference: InferenceConfig | None = None,
):
    """Build the FastAPI app. Pass ``backend`` in tests to avoid loading weights."""
    if FastAPI is None:  # pragma: no cover
        raise ImportError(
            "API dependencies missing. Install with: uv sync --extra api"
        )

    import cv2

    if backend is None:
        if inference is None:
            inference = InferenceConfig(device="cuda")
        backend = UltralyticsBackend(inference)

    app = FastAPI(
        title="vtrack inference API",
        description="Localhost-only vehicle detection and tracking PoC API.",
        version="0.1.0",
    )
    app.state.backend = backend

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model": backend.model_path,
            "device": backend.device,
        }

    async def _read_image(upload: UploadFile) -> np.ndarray:
        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Empty image upload")
        array = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Unable to decode image")
        return frame

    def _serialize(boxes: list[DetectionBox]) -> dict[str, object]:
        return {
            "count": len(boxes),
            "detections": [
                {
                    "xyxy": box.xyxy,
                    "confidence": box.confidence,
                    "class_id": box.class_id,
                    "class_name": box.class_name,
                    **({"track_id": box.track_id} if box.track_id is not None else {}),
                }
                for box in boxes
            ],
        }

    @app.post("/v1/detect")
    async def detect(file: UploadFile = File(...)) -> dict[str, object]:
        frame = await _read_image(file)
        return _serialize(backend.detect(frame))

    @app.post("/v1/track")
    async def track(
        file: UploadFile = File(...),
        session_id: str = Form(...),
    ) -> dict[str, object]:
        if not session_id.strip():
            raise HTTPException(status_code=400, detail="session_id is required")
        frame = await _read_image(file)
        payload = _serialize(backend.track(frame, session_id=session_id.strip()))
        payload["session_id"] = session_id.strip()
        return payload

    return app


def run_server(
    *,
    inference: InferenceConfig,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start uvicorn. Host must stay on loopback for this PoC."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "API dependencies missing. Install with: uv sync --extra api"
        ) from exc

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            f"Refusing to bind host={host!r}; use 127.0.0.1 for the localhost PoC API."
        )

    app = create_app(inference=inference)
    uvicorn.run(app, host=host, port=port, log_level="info")

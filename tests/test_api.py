"""API unit tests with an injected fake backend (no GPU weights required)."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from vtrack.api.app import DetectionBox, create_app


class FakeBackend:
    model_path = "fake.pt"
    device = "cpu"

    def detect(self, frame: np.ndarray) -> list[DetectionBox]:
        h, w = frame.shape[:2]
        return [
            DetectionBox(
                xyxy=[1.0, 2.0, min(10.0, w - 1), min(20.0, h - 1)],
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ]

    def track(self, frame: np.ndarray, *, session_id: str) -> list[DetectionBox]:
        boxes = self.detect(frame)
        return [
            DetectionBox(
                xyxy=box.xyxy,
                confidence=box.confidence,
                class_id=box.class_id,
                class_name=box.class_name,
                track_id=7 if session_id else None,
            )
            for box in boxes
        ]


def _png_bytes(width: int = 32, height: int = 24) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (40, 80, 120)
    ok, encoded = cv2.imencode(".png", frame)
    assert ok
    return encoded.tobytes()


@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(backend=FakeBackend())
    return TestClient(app)


def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "fake.pt"
    assert payload["device"] == "cpu"


def test_detect(client) -> None:
    response = client.post(
        "/v1/detect",
        files={"file": ("frame.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["detections"][0]["class_name"] == "car"
    assert "track_id" not in payload["detections"][0]


def test_track(client) -> None:
    response = client.post(
        "/v1/track",
        data={"session_id": "demo"},
        files={"file": ("frame.png", io.BytesIO(_png_bytes()), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "demo"
    assert payload["detections"][0]["track_id"] == 7


def test_run_server_rejects_non_loopback() -> None:
    from vtrack.api.app import run_server
    from vtrack.settings import InferenceConfig

    with pytest.raises(ValueError, match="127.0.0.1"):
        run_server(inference=InferenceConfig(device="cpu"), host="0.0.0.0", port=8000)

"""vtrack — Vehicle detection and tracking pipeline using YOLOv11."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EvaluationConfig": "vtrack.settings",
    "InferenceConfig": "vtrack.settings",
    "ProjectPaths": "vtrack.settings",
    "RemoteConfig": "vtrack.settings",
    "TrainingConfig": "vtrack.settings",
    "VehicleAnalytics": "vtrack.analytics",
    "VehicleDetector": "vtrack.detect",
    "VehiclePipeline": "vtrack.pipeline",
    "VehicleTracker": "vtrack.track",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

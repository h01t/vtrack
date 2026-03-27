"""Model profile detection based on checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ultralytics import YOLO

from vtrack.config import (
    COCO_VEHICLE_CLASSES,
    COCO_VEHICLE_NAMES,
    DEFAULT_MODEL,
    KITTI_NAMES,
)


def _normalize_names(
    names: Mapping[int, str] | list[str] | tuple[str, ...] | None,
) -> dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


@dataclass(frozen=True)
class ModelProfile:
    """Resolved class metadata for a model checkpoint."""

    source: str
    class_names: dict[int, str]
    class_filter: list[int] | None


def profile_from_names(
    names: Mapping[int, str] | list[str] | tuple[str, ...] | None,
    *,
    source: str,
) -> ModelProfile:
    """Resolve a runtime profile from checkpoint names."""
    normalized = _normalize_names(names)
    if normalized == KITTI_NAMES:
        return ModelProfile(source=source, class_names=normalized, class_filter=None)

    is_coco = all(normalized.get(class_id) == name for class_id, name in COCO_VEHICLE_NAMES.items())
    if is_coco:
        return ModelProfile(
            source=source,
            class_names=normalized,
            class_filter=list(COCO_VEHICLE_CLASSES),
        )

    fallback_names = COCO_VEHICLE_NAMES if Path(source).name == DEFAULT_MODEL else normalized
    fallback_filter = list(COCO_VEHICLE_CLASSES) if fallback_names == COCO_VEHICLE_NAMES else None
    return ModelProfile(source=source, class_names=fallback_names, class_filter=fallback_filter)


def resolve_model_profile(
    model_or_path: YOLO | str,
    *,
    source: str | None = None,
) -> ModelProfile:
    """Build a model profile from a loaded YOLO model or checkpoint path."""
    if isinstance(model_or_path, YOLO):
        model = model_or_path
        resolved_source = source or str(getattr(model, "ckpt_path", DEFAULT_MODEL))
    else:
        resolved_source = source or str(model_or_path)
        model = YOLO(resolved_source)
    return profile_from_names(getattr(model, "names", None), source=resolved_source)

"""Shared runtime dataclasses and typed payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class SourceSpec:
    """Normalized source descriptor used by runtime validation."""

    raw: str | int | Path
    kind: Literal["webcam", "file", "url", "stream"]
    resolved_path: Path | None = None


@dataclass(frozen=True)
class RunLimits:
    """Optional execution limits used by long-running commands."""

    max_frames: int | None = None
    max_seconds: float | None = None


@dataclass(frozen=True)
class RuntimeContext:
    """Stable runtime context fields for diagnostics/logging."""

    command: str
    model_path: str
    tracker: str | None
    device: str | None
    source_kind: str


@dataclass(frozen=True)
class RunStats:
    """Lightweight run summary stats."""

    frames_processed: int
    wall_time_sec: float
    avg_fps: float

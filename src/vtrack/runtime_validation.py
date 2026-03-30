"""Runtime source and execution-limit validation helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from vtrack.errors import SourceValidationError
from vtrack.runtime_types import RunLimits, SourceSpec


def parse_source_spec(value: str | int | Path) -> SourceSpec:
    """Normalize a source argument into a SourceSpec."""
    if isinstance(value, int):
        return SourceSpec(raw=value, kind="webcam", resolved_path=None)

    if isinstance(value, Path):
        return SourceSpec(raw=value, kind="file", resolved_path=value)

    if isinstance(value, str) and value.isdigit():
        return SourceSpec(raw=int(value), kind="webcam", resolved_path=None)

    if isinstance(value, (str, Path)):
        text = str(value)
        parsed = urlparse(text)
        if parsed.scheme in {"http", "https", "rtsp", "rtmp"}:
            return SourceSpec(raw=text, kind="url", resolved_path=None)

        path = Path(text)
        if path.exists():
            return SourceSpec(raw=text, kind="file", resolved_path=path)

        if "://" not in text and not path.exists():
            raise SourceValidationError(f"Source path does not exist: {text}")

        return SourceSpec(raw=text, kind="stream", resolved_path=None)

    raise SourceValidationError(f"Unsupported source type: {type(value)!r}")


def validate_source_for_command(
    source: SourceSpec,
    *,
    allow_stream: bool = True,
) -> SourceSpec:
    """Validate source compatibility for command execution."""
    if source.kind == "webcam" and isinstance(source.raw, int) and source.raw < 0:
        raise SourceValidationError("Webcam index must be zero or positive.")

    if (
        source.kind == "file"
        and source.resolved_path is not None
        and not source.resolved_path.exists()
    ):
        raise SourceValidationError(f"Source path does not exist: {source.resolved_path}")

    if not allow_stream and source.kind in {"url", "stream"}:
        raise SourceValidationError("This command does not support URL/stream sources.")

    return source


def validate_run_limits(max_frames: int | None, max_seconds: float | None) -> RunLimits:
    """Validate optional frame/time limits used by long-running commands."""
    if max_frames is not None and max_frames <= 0:
        raise SourceValidationError("--max-frames must be positive when provided.")
    if max_seconds is not None and max_seconds <= 0:
        raise SourceValidationError("--max-seconds must be positive when provided.")
    return RunLimits(max_frames=max_frames, max_seconds=max_seconds)

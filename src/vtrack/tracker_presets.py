"""Versioned tracker preset resolution for reproducible runtime behavior."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

_TRACKER_PRESET_FILES = {
    "bytetrack": "bytetrack.yaml",
    "bytetrack-occlusion": "bytetrack-occlusion.yaml",
    "botsort": "botsort.yaml",
}

_TRACKER_COMPAT_ALIASES = {
    "bytetrack.yaml": "bytetrack",
    "botsort.yaml": "botsort",
}


@dataclass(frozen=True)
class ResolvedTrackerConfig:
    """Resolved tracker preset or explicit YAML path."""

    requested: str
    name: str
    path: str
    is_builtin: bool

    @property
    def description(self) -> str:
        if self.is_builtin:
            return f"{self.name} ({self.path})"
        return self.path


def available_tracker_presets() -> tuple[str, ...]:
    """Return built-in tracker preset aliases."""
    return tuple(_TRACKER_PRESET_FILES)


def resolve_tracker_config(tracker: str) -> ResolvedTrackerConfig:
    """Resolve a tracker preset alias or an explicit YAML path."""
    requested = tracker.strip()
    if not requested:
        raise ValueError("Tracker must be a preset alias or a YAML path.")

    canonical = _TRACKER_COMPAT_ALIASES.get(requested, requested)
    if canonical in _TRACKER_PRESET_FILES:
        path = Path(
            str(files("vtrack").joinpath("trackers", _TRACKER_PRESET_FILES[canonical]))
        ).resolve()
        return ResolvedTrackerConfig(
            requested=requested,
            name=canonical,
            path=str(path),
            is_builtin=True,
        )

    explicit_path = Path(requested).expanduser()
    if not explicit_path.exists():
        raise FileNotFoundError(
            f"Tracker preset {tracker!r} was not found. Use a built-in preset alias or "
            "an existing YAML path."
        )

    return ResolvedTrackerConfig(
        requested=requested,
        name=explicit_path.name,
        path=str(explicit_path.resolve()),
        is_builtin=False,
    )

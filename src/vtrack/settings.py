"""Typed settings and project path helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from posixpath import dirname
from typing import Any, Mapping

from vtrack.config import (
    DEFAULT_AGNOSTIC_NMS,
    DEFAULT_CONFIDENCE,
    DEFAULT_HALF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_MODEL,
    DEFAULT_STREAM_BUFFER,
    DEFAULT_TRACK_CONFIDENCE,
    DEFAULT_TRACKER,
    DEFAULT_VID_STRIDE,
)


def _discover_project_root() -> Path:
    """Return the repository root for an editable src-layout checkout."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").exists():
        return root
    return Path.cwd()


def _default_remote_dir(*, project_name: str, project_root: Path | None = None) -> str:
    local_home = Path.home()
    if project_root is not None:
        try:
            relative = project_root.resolve().relative_to(local_home)
        except ValueError:
            return f"~/{project_name}"
        return f"~/{relative.as_posix()}"
    return f"~/{project_name}"


def default_remote_datasets_dir(remote_dir: str | None, *, project_name: str) -> str:
    """Choose a remote datasets root adjacent to the remote checkout by default."""
    normalized_remote_dir = normalize_remote_dir(raw_dir=remote_dir, project_name=project_name)
    if normalized_remote_dir == "~":
        return "~/datasets"
    if normalized_remote_dir.startswith("~/"):
        parent = dirname(normalized_remote_dir[2:])
        return f"~/{parent}/datasets" if parent not in ("", ".") else "~/datasets"
    if normalized_remote_dir.startswith("/"):
        parent = dirname(normalized_remote_dir.rstrip("/"))
        return f"{parent}/datasets"
    return "~/datasets"


def normalize_remote_dir(
    raw_dir: str | None,
    *,
    project_name: str,
    project_root: Path | None = None,
) -> str:
    """Normalize remote directories while preserving remote-home semantics.

    Shell exports like `VTRACK_REMOTE_DIR=~/object-det` expand locally before Python
    sees them. If the resulting path lives under the local home directory, convert it
    back to a `~/...` path so the remote shell resolves it relative to the remote home.
    """
    if not raw_dir:
        return _default_remote_dir(project_name=project_name, project_root=project_root)

    if raw_dir == "~" or raw_dir.startswith("~/"):
        return raw_dir

    local_home = Path.home()
    expanded = Path(raw_dir).expanduser()
    if expanded.is_absolute():
        try:
            relative = expanded.relative_to(local_home)
        except ValueError:
            return raw_dir
        return f"~/{relative.as_posix()}"

    return raw_dir


class InferenceDeviceError(RuntimeError):
    """Raised when a requested inference device cannot be used."""


def validate_inference_device(device: str | None) -> None:
    """Fail fast for explicitly requested MPS inference on unsupported machines."""
    if device is None or device.lower() != "mps":
        return

    import torch

    if not torch.backends.mps.is_available():
        raise InferenceDeviceError(
            "MPS inference requested via --device mps, but torch.backends.mps.is_available() "
            "is False on this machine."
        )


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical local paths used across runtime, training, and evaluation."""

    root: Path = field(default_factory=_discover_project_root)

    @property
    def src_dir(self) -> Path:
        return self.root / "src"

    @property
    def scripts_dir(self) -> Path:
        return self.root / "scripts"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def train_artifacts_dir(self) -> Path:
        return self.artifacts_dir / "train"

    @property
    def eval_artifacts_dir(self) -> Path:
        return self.artifacts_dir / "eval"

    @property
    def raw_runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def raw_training_dir(self) -> Path:
        return self.raw_runs_dir / "training"

    @property
    def raw_evaluation_dir(self) -> Path:
        return self.raw_runs_dir / "evaluation"

    def ensure_runtime_dirs(self) -> None:
        for path in (
            self.models_dir,
            self.artifacts_dir,
            self.train_artifacts_dir,
            self.eval_artifacts_dir,
            self.raw_runs_dir,
            self.raw_training_dir,
            self.raw_evaluation_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class InferenceConfig:
    """Defaults for detection and tracking workflows."""

    model_path: str = DEFAULT_MODEL
    min_confidence: float = DEFAULT_CONFIDENCE
    track_conf: float = DEFAULT_TRACK_CONFIDENCE
    tracker: str = DEFAULT_TRACKER
    trace_length: int = 30
    device: str | None = None
    imgsz: int = DEFAULT_IMGSZ
    iou: float = DEFAULT_IOU
    max_det: int = DEFAULT_MAX_DET
    half: bool = DEFAULT_HALF
    vid_stride: int = DEFAULT_VID_STRIDE
    stream_buffer: bool = DEFAULT_STREAM_BUFFER
    agnostic_nms: bool = DEFAULT_AGNOSTIC_NMS

    @property
    def confidence(self) -> float:
        """Compatibility alias for code paths that still read `.confidence`."""
        return self.min_confidence

    def predict_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conf": self.min_confidence,
            "imgsz": self.imgsz,
            "iou": self.iou,
            "max_det": self.max_det,
            "half": self.half,
            "agnostic_nms": self.agnostic_nms,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def track_kwargs(self) -> dict[str, Any]:
        kwargs = self.predict_kwargs()
        kwargs.update(
            {
                "conf": self.track_conf,
                "vid_stride": self.vid_stride,
                "stream_buffer": self.stream_buffer,
            }
        )
        return kwargs


@dataclass(frozen=True)
class TrainingConfig:
    """Training settings shared by local and remote training commands."""

    model_path: str = DEFAULT_MODEL
    data: str = "kitti.yaml"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    device: str = "mps"
    name: str = "vehicle_v1"
    amp: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation settings for a fine-tuned checkpoint and optional baseline."""

    model_path: str
    data: str = "kitti.yaml"
    name: str = "eval"
    baseline_model_path: str = DEFAULT_MODEL
    compare: bool = False


@dataclass(frozen=True)
class RemoteConfig:
    """Remote training connection details."""

    host: str | None = None
    remote_dir: str | None = None
    datasets_dir: str | None = None
    remote_python: str = "python3"

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        project_name: str = "object-det",
        project_root: Path | None = None,
    ) -> "RemoteConfig":
        data = os.environ if env is None else env
        return cls(
            host=data.get("VTRACK_REMOTE_HOST"),
            remote_dir=normalize_remote_dir(
                data.get("VTRACK_REMOTE_DIR"),
                project_name=project_name,
                project_root=project_root,
            ),
            datasets_dir=normalize_remote_dir(
                data.get("VTRACK_REMOTE_DATASETS_DIR"),
                project_name="datasets",
                project_root=project_root.parent if project_root is not None else None,
            )
            if data.get("VTRACK_REMOTE_DATASETS_DIR")
            else default_remote_datasets_dir(
                normalize_remote_dir(
                    data.get("VTRACK_REMOTE_DIR"),
                    project_name=project_name,
                    project_root=project_root,
                ),
                project_name=project_name,
            ),
            remote_python=data.get("VTRACK_REMOTE_PYTHON", "python3"),
        )

    def require_host(self) -> str:
        if not self.host:
            raise ValueError(
                "Remote host is not configured. Set VTRACK_REMOTE_HOST or pass --host."
            )
        return self.host

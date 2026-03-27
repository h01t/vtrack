"""Artifact bundle publishing for training and evaluation workflows."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vtrack.settings import ProjectPaths

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".svg"}
TEXT_SUFFIXES = {".csv", ".json", ".txt", ".yaml", ".yml"}
WEIGHT_SUFFIXES = {".mlmodel", ".onnx", ".pt"}


@dataclass(frozen=True)
class ArtifactBundle:
    """Paths for a normalized artifact bundle."""

    bundle_dir: Path
    manifest_path: Path
    summary_path: Path


def extract_metrics_summary(metrics: Any) -> dict[str, Any]:
    """Convert an Ultralytics metrics object into a JSON-friendly summary."""
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    names = getattr(metrics, "names", {}) or {}
    if isinstance(names, Mapping):
        normalized_names = {int(key): str(value) for key, value in names.items()}
    else:
        normalized_names = {index: str(value) for index, value in enumerate(names)}

    overall = {
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "precision": float(getattr(box, "mp", 0.0) or 0.0),
        "recall": float(getattr(box, "mr", 0.0) or 0.0),
        "speed_ms_per_image": {
            key: float(value) for key, value in speed.items()
        },
    }

    per_class: list[dict[str, Any]] = []
    if box is not None and hasattr(box, "class_result"):
        for class_id, class_name in normalized_names.items():
            try:
                precision, recall, map50, map50_95 = box.class_result(class_id)
            except Exception:
                continue
            per_class.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "precision": float(precision),
                    "recall": float(recall),
                    "map50": float(map50),
                    "map50_95": float(map50_95),
                }
            )

    return {
        "overall": overall,
        "per_class": per_class,
    }


def compute_baseline_delta(
    finetuned_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
) -> dict[str, float]:
    """Compute overall metric deltas between two summaries."""
    finetuned = finetuned_summary["overall"]
    baseline = baseline_summary["overall"]
    return {
        "map50": float(finetuned["map50"]) - float(baseline["map50"]),
        "map50_95": float(finetuned["map50_95"]) - float(baseline["map50_95"]),
        "precision": float(finetuned["precision"]) - float(baseline["precision"]),
        "recall": float(finetuned["recall"]) - float(baseline["recall"]),
    }


def _git_sha(paths: ProjectPaths) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=paths.root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _copy_selected_raw_outputs(raw_output_path: Path, bundle_dir: Path) -> None:
    if not raw_output_path.exists():
        return

    plots_dir = bundle_dir / "plots"
    weights_dir = bundle_dir / "weights"
    files_dir = bundle_dir / "files"

    for source in raw_output_path.rglob("*"):
        if not source.is_file():
            continue

        suffix = source.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            target = plots_dir / source.name
        elif suffix in WEIGHT_SUFFIXES:
            target = weights_dir / source.name
        elif suffix in TEXT_SUFFIXES:
            target = files_dir / source.name
        else:
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _copy_checkpoint(checkpoint_path: str | Path | None, bundle_dir: Path) -> None:
    if checkpoint_path is None:
        return
    source = Path(checkpoint_path)
    if not source.exists():
        return
    target = bundle_dir / "weights" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def publish_artifact_bundle(
    *,
    paths: ProjectPaths,
    run_type: str,
    run_id: str,
    summary: Mapping[str, Any],
    command: list[str],
    raw_output_path: str | Path | None = None,
    extra_raw_outputs: Mapping[str, str | Path] | None = None,
    dataset_path: str | None = None,
    checkpoint_path: str | None = None,
    baseline_path: str | None = None,
) -> ArtifactBundle:
    """Write a normalized artifact bundle for a run."""
    base_dir = paths.train_artifacts_dir if run_type == "train" else paths.eval_artifacts_dir
    bundle_dir = base_dir / run_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    raw_path = Path(raw_output_path).resolve() if raw_output_path else None
    _copy_selected_raw_outputs(raw_path, bundle_dir) if raw_path else None
    for label, extra_path in (extra_raw_outputs or {}).items():
        _copy_selected_raw_outputs(Path(extra_path).resolve(), bundle_dir / label)
    _copy_checkpoint(checkpoint_path, bundle_dir)

    created_at = datetime.now(UTC).isoformat()
    manifest = {
        "run_type": run_type,
        "run_id": run_id,
        "created_at": created_at,
        "git_sha": _git_sha(paths),
        "command": command,
        "dataset_path": dataset_path,
        "checkpoint_path": checkpoint_path,
        "baseline_path": baseline_path,
        "raw_output_path": str(raw_path) if raw_path else None,
    }

    manifest_path = bundle_dir / "manifest.json"
    summary_path = bundle_dir / "summary.json"

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return ArtifactBundle(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def sync_checkpoints_to_models(
    *,
    paths: ProjectPaths,
    checkpoint_dir: str | Path,
    run_name: str,
) -> dict[str, Path]:
    """Copy best/last checkpoints into the local models directory."""
    source_dir = Path(checkpoint_dir)
    paths.models_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}
    for filename in ("best.pt", "last.pt"):
        source = source_dir / filename
        if not source.exists():
            continue

        canonical_target = paths.models_dir / filename
        named_target = paths.models_dir / f"{run_name}_{filename}"
        shutil.copy2(source, canonical_target)
        shutil.copy2(source, named_target)
        copied[filename] = canonical_target

    return copied

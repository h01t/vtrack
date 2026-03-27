"""High-level workflows shared by the CLI and compatibility wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from vtrack.analytics import VehicleAnalytics
from vtrack.artifacts import (
    ArtifactBundle,
    compute_baseline_delta,
    extract_metrics_summary,
    publish_artifact_bundle,
    sync_checkpoints_to_models,
)
from vtrack.detect import VehicleDetector
from vtrack.pipeline import VehiclePipeline
from vtrack.remote import build_remote_command_set, run_remote_training_commands
from vtrack.settings import (
    EvaluationConfig,
    InferenceConfig,
    ProjectPaths,
    RemoteConfig,
    TrainingConfig,
)


def run_demo(
    *,
    source: str | Path | int,
    inference: InferenceConfig,
    analytics: VehicleAnalytics | None = None,
    display: bool = True,
    save_path: str | None = None,
    export_csv: str | None = None,
    export_json: str | None = None,
) -> None:
    pipeline = VehiclePipeline(
        model_path=inference.model_path,
        confidence=inference.confidence,
        tracker=inference.tracker,
        trace_length=inference.trace_length,
        analytics=analytics,
    )
    pipeline.run(
        source=source,
        display=display,
        save_path=save_path,
        export_csv=export_csv,
        export_json=export_json,
    )


def run_detect_image(
    *,
    source: str | Path,
    inference: InferenceConfig,
    save: bool = True,
) -> list[Any]:
    detector = VehicleDetector(
        model_path=inference.model_path,
        confidence=inference.confidence,
    )
    return list(detector.detect_image(source=source, save=save))


def run_detect_video(
    *,
    source: str | Path,
    inference: InferenceConfig,
    save: bool = True,
    stream: bool = True,
) -> Any:
    detector = VehicleDetector(
        model_path=inference.model_path,
        confidence=inference.confidence,
    )
    return detector.detect_video(source=source, save=save, stream=stream)


def run_training(
    *,
    training: TrainingConfig,
    paths: ProjectPaths,
    command: list[str],
) -> tuple[Any, ArtifactBundle]:
    paths.ensure_runtime_dirs()

    model = YOLO(training.model_path)
    metrics = model.train(
        amp=training.amp,
        data=training.data,
        epochs=training.epochs,
        imgsz=training.imgsz,
        batch=training.batch,
        device=training.device,
        project=str(paths.raw_training_dir.resolve()),
        name=training.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        patience=10,
        save=True,
        plots=True,
    )

    raw_output_path = Path(getattr(metrics, "save_dir", paths.raw_training_dir / training.name))
    checkpoint_dir = raw_output_path / "weights"
    copied = sync_checkpoints_to_models(
        paths=paths,
        checkpoint_dir=checkpoint_dir,
        run_name=training.name,
    )

    summary = extract_metrics_summary(metrics)
    summary["training"] = {
        "run_name": training.name,
        "device": training.device,
        "data": training.data,
        "epochs": training.epochs,
        "imgsz": training.imgsz,
        "batch": training.batch,
        "amp": training.amp,
        "published_checkpoints": {name: str(path) for name, path in copied.items()},
    }

    bundle = publish_artifact_bundle(
        paths=paths,
        run_type="train",
        run_id=training.name,
        summary=summary,
        command=command,
        raw_output_path=raw_output_path,
        dataset_path=training.data,
        checkpoint_path=str(copied.get("best.pt") or (checkpoint_dir / "best.pt")),
    )
    return metrics, bundle


def run_evaluation(
    *,
    evaluation: EvaluationConfig,
    paths: ProjectPaths,
    command: list[str],
) -> dict[str, Any]:
    paths.ensure_runtime_dirs()

    finetuned_model = YOLO(evaluation.model_path)
    finetuned_metrics = finetuned_model.val(
        data=evaluation.data,
        project=str(paths.raw_evaluation_dir.resolve()),
        name=f"{evaluation.name}_finetuned",
        exist_ok=True,
        plots=True,
    )
    finetuned_summary = extract_metrics_summary(finetuned_metrics)

    baseline_summary = None
    baseline_raw_output_path = None
    if evaluation.compare:
        baseline_model = YOLO(evaluation.baseline_model_path)
        baseline_metrics = baseline_model.val(
            data=evaluation.data,
            project=str(paths.raw_evaluation_dir.resolve()),
            name=f"{evaluation.name}_baseline",
            exist_ok=True,
            plots=True,
        )
        baseline_summary = extract_metrics_summary(baseline_metrics)
        baseline_raw_output_path = Path(
            getattr(
                baseline_metrics,
                "save_dir",
                paths.raw_evaluation_dir / f"{evaluation.name}_baseline",
            )
        )

    summary: dict[str, Any] = {
        "finetuned": finetuned_summary,
    }
    if baseline_summary is not None:
        summary["baseline"] = baseline_summary
        summary["delta"] = compute_baseline_delta(
            finetuned_summary=finetuned_summary,
            baseline_summary=baseline_summary,
        )

    bundle = publish_artifact_bundle(
        paths=paths,
        run_type="eval",
        run_id=evaluation.name,
        summary=summary,
        command=command,
        raw_output_path=Path(
            getattr(
                finetuned_metrics,
                "save_dir",
                paths.raw_evaluation_dir / f"{evaluation.name}_finetuned",
            )
        ),
        extra_raw_outputs=(
            {"baseline": baseline_raw_output_path}
            if baseline_raw_output_path is not None
            else None
        ),
        dataset_path=evaluation.data,
        checkpoint_path=evaluation.model_path,
        baseline_path=evaluation.baseline_model_path if evaluation.compare else None,
    )
    return {
        "finetuned_metrics": finetuned_metrics,
        "baseline_summary": baseline_summary,
        "summary": summary,
        "bundle": bundle,
    }


def run_remote_training(
    *,
    training: TrainingConfig,
    remote: RemoteConfig,
    paths: ProjectPaths,
) -> None:
    paths.ensure_runtime_dirs()
    commands = build_remote_command_set(
        paths=paths,
        training=training,
        remote=remote,
    )
    run_remote_training_commands(commands)

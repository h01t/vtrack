"""High-level workflows shared by the CLI and compatibility wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
    apply_ultralytics_datasets_dir,
    checkpoint_dir,
    resolve_dataset_config,
    validate_inference_device,
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
        confidence=inference.min_confidence,
        track_conf=inference.track_conf,
        tracker=inference.tracker,
        trace_length=inference.trace_length,
        analytics=analytics,
        device=inference.device,
        imgsz=inference.imgsz,
        iou=inference.iou,
        max_det=inference.max_det,
        half=inference.half,
        vid_stride=inference.vid_stride,
        stream_buffer=inference.stream_buffer,
        agnostic_nms=inference.agnostic_nms,
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
        confidence=inference.min_confidence,
        device=inference.device,
        imgsz=inference.imgsz,
        iou=inference.iou,
        max_det=inference.max_det,
        half=inference.half,
        agnostic_nms=inference.agnostic_nms,
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
        confidence=inference.min_confidence,
        device=inference.device,
        imgsz=inference.imgsz,
        iou=inference.iou,
        max_det=inference.max_det,
        half=inference.half,
        agnostic_nms=inference.agnostic_nms,
    )
    return detector.detect_video(source=source, save=save, stream=stream)


def run_tracking_benchmark(
    *,
    source: str | Path | int,
    inference: InferenceConfig,
    trackers: list[str] | None = None,
    analytics_factory: Callable[[], VehicleAnalytics] | None = None,
    max_frames: int | None = None,
    warmup_frames: int = 30,
    export_csv: str | None = None,
) -> dict[str, Any]:
    from vtrack.benchmarking import benchmark_trackers

    return benchmark_trackers(
        source=source,
        inference=inference,
        trackers=trackers,
        analytics_factory=analytics_factory,
        max_frames=max_frames,
        warmup_frames=warmup_frames,
        export_csv=export_csv,
    )


def run_training(
    *,
    training: TrainingConfig,
    paths: ProjectPaths,
    command: list[str],
) -> tuple[Any, ArtifactBundle]:
    paths.ensure_runtime_dirs()
    validate_inference_device(training.device)
    apply_ultralytics_datasets_dir()
    resolved_data = resolve_dataset_config(training.data)

    model = YOLO(training.model_path)
    metrics = model.train(
        amp=training.amp,
        data=resolved_data,
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
    weights_dir = raw_output_path / "weights"
    host_checkpoint_dir = checkpoint_dir()
    copied = sync_checkpoints_to_models(
        paths=paths,
        checkpoint_dir=weights_dir,
        run_name=training.name,
        mirror_dir=host_checkpoint_dir,
    )

    summary = extract_metrics_summary(metrics)
    summary["training"] = {
        "run_name": training.name,
        "device": training.device,
        "data": resolved_data,
        "epochs": training.epochs,
        "imgsz": training.imgsz,
        "batch": training.batch,
        "amp": training.amp,
        "published_checkpoints": {name: str(path) for name, path in copied.items()},
        "checkpoint_mirror": str(host_checkpoint_dir),
    }

    bundle = publish_artifact_bundle(
        paths=paths,
        run_type="train",
        run_id=training.name,
        summary=summary,
        command=command,
        raw_output_path=raw_output_path,
        dataset_path=resolved_data,
        checkpoint_path=str(copied.get("best.pt") or (weights_dir / "best.pt")),
    )
    return metrics, bundle


def run_evaluation(
    *,
    evaluation: EvaluationConfig,
    paths: ProjectPaths,
    command: list[str],
) -> dict[str, Any]:
    paths.ensure_runtime_dirs()
    apply_ultralytics_datasets_dir()
    resolved_data = resolve_dataset_config(evaluation.data)

    finetuned_model = YOLO(evaluation.model_path)
    finetuned_metrics = finetuned_model.val(
        data=resolved_data,
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
            data=resolved_data,
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
        dataset_path=resolved_data,
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


def run_export_onnx(
    *,
    model_path: str | Path,
    paths: ProjectPaths,
    imgsz: int = 640,
    output: str | Path | None = None,
) -> Path:
    """Export a checkpoint to ONNX and mirror it under the host checkpoint dir."""
    import shutil

    paths.ensure_runtime_dirs()
    source = Path(model_path)
    model = YOLO(str(source))
    export_result = model.export(format="onnx", imgsz=imgsz)
    exported = Path(str(export_result))
    if not exported.is_file():
        raise FileNotFoundError(f"ONNX export did not produce a file: {export_result}")

    target = Path(output) if output is not None else paths.models_dir / f"{source.stem}.onnx"
    target.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)

    mirror_root = checkpoint_dir()
    mirror_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, mirror_root / target.name)
    return target.resolve()


def run_export_benchmark(
    *,
    source: str | Path | int,
    pt_model: str | Path,
    onnx_model: str | Path,
    device: str = "cuda",
    imgsz: int = 640,
    max_frames: int = 150,
    warmup_frames: int = 30,
) -> dict[str, Any]:
    """Compare frame latency for a .pt checkpoint vs an ONNX export."""
    import time

    from vtrack.benchmarking import _p95

    def _time_model(model_path: str | Path) -> dict[str, Any]:
        model = YOLO(str(model_path))
        latencies_ms: list[float] = []
        frames = 0
        iterator = iter(
            model.predict(
                source=source,
                stream=True,
                device=device,
                imgsz=imgsz,
                verbose=False,
            )
        )
        while frames < max_frames:
            frame_start = time.perf_counter()
            try:
                next(iterator)
            except StopIteration:
                break
            elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
            frames += 1
            if frames > warmup_frames:
                latencies_ms.append(elapsed_ms)
        timed = len(latencies_ms)
        return {
            "model": str(model_path),
            "frames_processed": frames,
            "timed_frames": timed,
            "avg_fps": round(timed / (sum(latencies_ms) / 1000.0), 3) if latencies_ms else 0.0,
            "p95_frame_ms": round(_p95(latencies_ms), 3),
            "device": device,
            "imgsz": imgsz,
        }

    return {
        "pytorch": _time_model(pt_model),
        "onnx": _time_model(onnx_model),
    }

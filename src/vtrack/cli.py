"""Unified command-line interface for the vtrack project."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any

from vtrack.settings import (
    EvaluationConfig,
    InferenceConfig,
    InferenceDeviceError,
    ProjectPaths,
    RemoteConfig,
    TrainingConfig,
    default_remote_datasets_dir,
    normalize_remote_dir,
)

if TYPE_CHECKING:
    from vtrack.analytics import VehicleAnalytics


def parse_line(value: str) -> tuple[Any, Any]:
    """Parse 'x1,y1,x2,y2' into two supervision Points."""
    import supervision as sv

    coords = [int(x) for x in value.split(",")]
    if len(coords) != 4:
        raise argparse.ArgumentTypeError("Line must be x1,y1,x2,y2")
    return sv.Point(coords[0], coords[1]), sv.Point(coords[2], coords[3])


def parse_polygon(value: str) -> Any:
    """Parse 'x1,y1,x2,y2,...' into a polygon array."""
    import numpy as np

    coords = [int(x) for x in value.split(",")]
    if len(coords) % 2 != 0 or len(coords) < 6:
        raise argparse.ArgumentTypeError("Polygon needs at least 3 points: x1,y1,x2,y2,x3,y3")
    return np.array(coords).reshape(-1, 2)


def _parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def _analytics_geometry_from_args(
    args: argparse.Namespace,
) -> tuple[tuple[Any, Any] | None, Any | None]:
    line_geometry = parse_line(args.line) if getattr(args, "line", None) else None
    polygon_geometry = parse_polygon(args.zone) if getattr(args, "zone", None) else None
    return line_geometry, polygon_geometry


def _build_analytics_from_geometry(
    line_geometry: tuple[Any, Any] | None,
    polygon_geometry: Any | None,
) -> "VehicleAnalytics":
    import supervision as sv

    from vtrack.analytics import VehicleAnalytics

    line_zone = None
    polygon_zone = None
    if line_geometry is not None:
        start, end = line_geometry
        line_zone = sv.LineZone(start=start, end=end)
    if polygon_geometry is not None:
        polygon_zone = sv.PolygonZone(polygon=polygon_geometry.copy())
    return VehicleAnalytics(line_zone=line_zone, polygon_zone=polygon_zone)


def _build_analytics(
    args: argparse.Namespace,
    *,
    always: bool = False,
) -> "VehicleAnalytics | None":
    enabled = always or any(
        getattr(args, field, None)
        for field in ("analytics", "line", "zone", "export_csv", "export_json")
    )
    if not enabled:
        return None

    line_geometry, polygon_geometry = _analytics_geometry_from_args(args)
    return _build_analytics_from_geometry(line_geometry, polygon_geometry)


def _build_analytics_factory(args: argparse.Namespace):
    line_geometry, polygon_geometry = _analytics_geometry_from_args(args)

    def factory():
        return _build_analytics_from_geometry(line_geometry, polygon_geometry)

    return factory


def _inference_config_from_args(args: argparse.Namespace) -> InferenceConfig:
    defaults = InferenceConfig()
    return InferenceConfig(
        model_path=args.model,
        min_confidence=args.confidence,
        track_conf=getattr(args, "track_conf", defaults.track_conf),
        tracker=getattr(args, "tracker", defaults.tracker),
        trace_length=getattr(args, "trace_length", defaults.trace_length),
        device=getattr(args, "device", defaults.device),
        imgsz=getattr(args, "imgsz", defaults.imgsz),
        iou=getattr(args, "iou", defaults.iou),
        max_det=getattr(args, "max_det", defaults.max_det),
        half=getattr(args, "half", defaults.half),
        vid_stride=getattr(args, "vid_stride", defaults.vid_stride),
        stream_buffer=getattr(args, "stream_buffer", defaults.stream_buffer),
        agnostic_nms=getattr(args, "agnostic_nms", defaults.agnostic_nms),
    )


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        model_path=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        amp=not args.no_amp,
    )


def _evaluation_config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        model_path=args.model,
        data=args.data,
        name=args.name,
        baseline_model_path=args.baseline,
        compare=args.compare,
    )


def _remote_config_from_args(args: argparse.Namespace, *, paths: ProjectPaths) -> RemoteConfig:
    config = RemoteConfig.from_env(
        project_name=paths.root.name,
        project_root=paths.root,
    )
    remote_dir = (
        normalize_remote_dir(
            args.remote_dir,
            project_name=paths.root.name,
            project_root=paths.root,
        )
        if args.remote_dir
        else config.remote_dir
    )
    return RemoteConfig(
        host=args.host or config.host,
        remote_dir=remote_dir,
        datasets_dir=(
            normalize_remote_dir(
                args.remote_datasets_dir,
                project_name="datasets",
                project_root=paths.root.parent,
            )
            if args.remote_datasets_dir
            else config.datasets_dir
            or default_remote_datasets_dir(remote_dir, project_name=paths.root.name)
        ),
        remote_python=args.remote_python or config.remote_python,
    )


def _cmd_demo(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_demo

    run_demo(
        source=_parse_source(args.source),
        inference=_inference_config_from_args(args),
        analytics=_build_analytics(args),
        display=not args.no_display,
        save_path=args.save,
        export_csv=args.export_csv,
        export_json=args.export_json,
    )
    return 0


def _cmd_benchmark_track(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_tracking_benchmark

    report = run_tracking_benchmark(
        source=_parse_source(args.source),
        inference=_inference_config_from_args(args),
        trackers=args.trackers,
        analytics_factory=_build_analytics_factory(args),
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        export_csv=args.export_csv,
    )
    print(json.dumps(report, indent=2))
    return 0


def _cmd_detect_image(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_detect_image

    results = run_detect_image(
        source=args.source,
        inference=_inference_config_from_args(args),
        save=not args.no_save,
    )
    for result in results:
        boxes = result.boxes
        print(f"\nDetected {len(boxes)} object(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names.get(cls_id, f"class_{cls_id}")
            print(f"  - {name}: {conf:.2f}")
        if not args.no_show:
            result.show()
    return 0


def _cmd_detect_video(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_detect_video

    frame_count = 0
    for result in run_detect_video(
        source=args.source,
        inference=_inference_config_from_args(args),
        save=args.save,
        stream=True,
    ):
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(result.boxes)} detections")
    print(f"\nProcessed {frame_count} frames total.")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_training

    paths = ProjectPaths()
    training = _training_config_from_args(args)
    _, bundle = run_training(
        training=training,
        paths=paths,
        command=args.command_argv,
    )
    print("\n--- Training Complete ---")
    print(f"Artifact bundle: {bundle.bundle_dir}")
    print(f"Manifest: {bundle.manifest_path}")
    print(f"Summary: {bundle.summary_path}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_evaluation

    paths = ProjectPaths()
    evaluation = _evaluation_config_from_args(args)
    result = run_evaluation(
        evaluation=evaluation,
        paths=paths,
        command=args.command_argv,
    )

    summary = result["summary"]
    finetuned = summary["finetuned"]["overall"]
    print(f"\n--- {evaluation.name}_finetuned ---")
    print(f"mAP@0.5:      {finetuned['map50']:.4f}")
    print(f"mAP@0.5:0.95: {finetuned['map50_95']:.4f}")
    print(f"Precision:     {finetuned['precision']:.4f}")
    print(f"Recall:        {finetuned['recall']:.4f}")
    if evaluation.compare and "delta" in summary:
        delta = summary["delta"]
        print("\n--- Comparison ---")
        print(f"mAP@0.5 delta:      {delta['map50']:+.4f}")
        print(f"mAP@0.5:0.95 delta: {delta['map50_95']:+.4f}")

    bundle = result["bundle"]
    print(f"\nArtifact bundle: {bundle.bundle_dir}")
    print(f"Manifest: {bundle.manifest_path}")
    print(f"Summary: {bundle.summary_path}")
    return 0


def _cmd_train_remote(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_remote_training

    paths = ProjectPaths()
    training = _training_config_from_args(args)
    remote = _remote_config_from_args(args, paths=paths)
    run_remote_training(training=training, remote=remote, paths=paths)
    print(f"Remote training completed for run '{training.name}'.")
    print(f"Synced artifacts: {paths.train_artifacts_dir / training.name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vehicle detection, tracking, training, and evaluation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run tracking + analytics on a video source")
    demo.add_argument("source", help="Video file, camera index (0), RTSP URL, or YouTube URL")
    demo.add_argument("--model", default="yolo11n.pt", help="Model weights path")
    demo.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence kept for overlays/analytics after tracking",
    )
    demo.add_argument(
        "--track-conf",
        type=float,
        default=0.10,
        help="Detection confidence threshold passed into the tracker",
    )
    demo.add_argument(
        "--tracker",
        default="bytetrack",
        help="Tracker preset alias or explicit YAML path",
    )
    demo.add_argument("--trace-length", type=int, default=30, help="Trail length in frames")
    demo.add_argument("--device", default=None, help="Inference device (e.g. cpu, cuda, mps)")
    demo.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    demo.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    demo.add_argument("--max-det", type=int, default=300, help="Maximum detections per frame")
    demo.add_argument("--half", action="store_true", help="Enable half-precision inference")
    demo.add_argument("--vid-stride", type=int, default=1, help="Read every Nth video frame")
    demo.add_argument(
        "--stream-buffer",
        action="store_true",
        help="Buffer all stream frames instead of keeping the latest frame",
    )
    demo.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic non-maximum suppression",
    )
    demo.add_argument("--save", default=None, help="Output video path (e.g. outputs/demo.mp4)")
    demo.add_argument("--no-display", action="store_true", help="Disable live display window")
    demo.add_argument("--analytics", action="store_true", help="Enable vehicle analytics")
    demo.add_argument("--line", type=str, default=None, help="Counting line as x1,y1,x2,y2")
    demo.add_argument("--zone", type=str, default=None, help="Monitoring zone polygon as x1,y1,...")
    demo.add_argument("--export-csv", default=None, help="Export per-frame data to CSV")
    demo.add_argument("--export-json", default=None, help="Export summary to JSON")
    demo.set_defaults(handler=_cmd_demo)

    detect_image = subparsers.add_parser("detect-image", help="Run detection on a single image")
    detect_image.add_argument(
        "source",
        nargs="?",
        default="https://ultralytics.com/images/bus.jpg",
        help="Image path or URL",
    )
    detect_image.add_argument("--model", default="yolo11n.pt", help="Model weights path")
    detect_image.add_argument("--confidence", type=float, default=0.25, help="Detection confidence")
    detect_image.add_argument(
        "--no-save",
        action="store_true",
        help="Disable Ultralytics save output",
    )
    detect_image.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. cpu, cuda, mps)",
    )
    detect_image.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    detect_image.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    detect_image.add_argument("--max-det", type=int, default=300, help="Maximum detections")
    detect_image.add_argument("--half", action="store_true", help="Enable half-precision inference")
    detect_image.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic non-maximum suppression",
    )
    detect_image.add_argument("--no-show", action="store_true", help="Disable image preview window")
    detect_image.set_defaults(handler=_cmd_detect_image)

    detect_video = subparsers.add_parser("detect-video", help="Run detection on a video source")
    detect_video.add_argument("source", help="Video file path")
    detect_video.add_argument("--model", default="yolo11n.pt", help="Model weights path")
    detect_video.add_argument("--confidence", type=float, default=0.25, help="Detection confidence")
    detect_video.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. cpu, cuda, mps)",
    )
    detect_video.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    detect_video.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    detect_video.add_argument("--max-det", type=int, default=300, help="Maximum detections")
    detect_video.add_argument("--half", action="store_true", help="Enable half-precision inference")
    detect_video.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic non-maximum suppression",
    )
    detect_video.add_argument("--save", action="store_true", help="Save annotated video")
    detect_video.set_defaults(handler=_cmd_detect_video)

    benchmark_track = subparsers.add_parser(
        "benchmark-track",
        help="Benchmark tracking presets on a video source",
    )
    benchmark_track.add_argument(
        "source",
        help="Video file, camera index (0), RTSP URL, or YouTube URL",
    )
    benchmark_track.add_argument("--model", default="yolo11n.pt", help="Model weights path")
    benchmark_track.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence kept for analytics after tracking",
    )
    benchmark_track.add_argument(
        "--track-conf",
        type=float,
        default=0.10,
        help="Detection confidence threshold passed into the tracker",
    )
    benchmark_track.add_argument(
        "--tracker",
        dest="trackers",
        action="append",
        default=None,
        help="Tracker preset alias or explicit YAML path; repeat to compare multiple trackers",
    )
    benchmark_track.add_argument(
        "--device",
        default=None,
        help="Inference device (e.g. cpu, cuda, mps)",
    )
    benchmark_track.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    benchmark_track.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    benchmark_track.add_argument("--max-det", type=int, default=300, help="Maximum detections")
    benchmark_track.add_argument(
        "--half",
        action="store_true",
        help="Enable half-precision inference",
    )
    benchmark_track.add_argument(
        "--vid-stride",
        type=int,
        default=1,
        help="Read every Nth video frame",
    )
    benchmark_track.add_argument(
        "--stream-buffer",
        action="store_true",
        help="Buffer all stream frames instead of keeping the latest frame",
    )
    benchmark_track.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic non-maximum suppression",
    )
    benchmark_track.add_argument(
        "--line",
        type=str,
        default=None,
        help="Counting line as x1,y1,x2,y2",
    )
    benchmark_track.add_argument(
        "--zone",
        type=str,
        default=None,
        help="Monitoring zone polygon as x1,y1,...",
    )
    benchmark_track.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after processing this many frames",
    )
    benchmark_track.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Exclude the first N frames from latency stats",
    )
    benchmark_track.add_argument(
        "--export-csv",
        default=None,
        help="Optional CSV path for one summary row per tracker run",
    )
    benchmark_track.set_defaults(handler=_cmd_benchmark_track)

    train = subparsers.add_parser("train", help="Train a model locally")
    train.add_argument("--model", default="yolo11n.pt", help="Base model weights")
    train.add_argument("--data", default="kitti.yaml", help="Dataset config")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="mps", help="Device: mps, cpu, or cuda")
    train.add_argument("--name", default="vehicle_v1", help="Run name")
    train.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    train.set_defaults(handler=_cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a model checkpoint")
    evaluate.add_argument("--model", required=True, help="Model weights path")
    evaluate.add_argument("--data", default="kitti.yaml", help="Dataset config")
    evaluate.add_argument("--name", default="eval", help="Run name")
    evaluate.add_argument("--baseline", default="yolo11n.pt", help="Baseline model for comparison")
    evaluate.add_argument("--compare", action="store_true", help="Also evaluate the baseline model")
    evaluate.set_defaults(handler=_cmd_evaluate)

    train_remote = subparsers.add_parser(
        "train-remote",
        help="Run remote training and sync artifacts",
    )
    train_remote.add_argument("--model", default="yolo11n.pt", help="Base model weights")
    train_remote.add_argument("--data", default="kitti.yaml", help="Dataset config")
    train_remote.add_argument("--epochs", type=int, default=50)
    train_remote.add_argument("--imgsz", type=int, default=640)
    train_remote.add_argument("--batch", type=int, default=16)
    train_remote.add_argument("--device", default="cuda", help="Remote device")
    train_remote.add_argument("--name", default="vehicle_v1", help="Run name")
    train_remote.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    train_remote.add_argument(
        "--host",
        default=None,
        help="Remote host (defaults to VTRACK_REMOTE_HOST)",
    )
    train_remote.add_argument(
        "--remote-dir",
        default=None,
        help="Remote checkout directory (defaults to VTRACK_REMOTE_DIR or ~/<project-name>)",
    )
    train_remote.add_argument(
        "--remote-python",
        default=None,
        help="Remote Python executable (defaults to VTRACK_REMOTE_PYTHON or python3)",
    )
    train_remote.add_argument(
        "--remote-datasets-dir",
        default=None,
        help=(
            "Remote datasets root "
            "(defaults to VTRACK_REMOTE_DATASETS_DIR or a sibling ~/.../datasets path)"
        ),
    )
    train_remote.set_defaults(handler=_cmd_train_remote)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    command_argv = ["vtrack", *((argv if argv is not None else sys.argv[1:]))]
    args = parser.parse_args(argv)
    args.command_argv = command_argv
    try:
        return args.handler(args)
    except InferenceDeviceError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


def demo_main(argv: list[str] | None = None) -> int:
    return main(["demo", *(argv or sys.argv[1:])])


def detect_image_main(argv: list[str] | None = None) -> int:
    return main(["detect-image", *(argv or sys.argv[1:])])


def detect_video_main(argv: list[str] | None = None) -> int:
    return main(["detect-video", *(argv or sys.argv[1:])])


def train_main(argv: list[str] | None = None) -> int:
    return main(["train", *(argv or sys.argv[1:])])


def evaluate_main(argv: list[str] | None = None) -> int:
    return main(["evaluate", *(argv or sys.argv[1:])])


def train_remote_main(argv: list[str] | None = None) -> int:
    return main(["train-remote", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())

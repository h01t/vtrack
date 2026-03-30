"""CLI parser builder and subcommand registration."""

from __future__ import annotations

import argparse

from vtrack.cli_handlers import (
    cmd_benchmark_track,
    cmd_demo,
    cmd_detect_image,
    cmd_detect_video,
    cmd_evaluate,
    cmd_train,
    cmd_train_remote,
)


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
    demo.set_defaults(handler=cmd_demo)

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
    detect_image.set_defaults(handler=cmd_detect_image)

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
    detect_video.set_defaults(handler=cmd_detect_video)

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
    benchmark_track.set_defaults(handler=cmd_benchmark_track)

    train = subparsers.add_parser("train", help="Train a model locally")
    train.add_argument("--model", default="yolo11n.pt", help="Base model weights")
    train.add_argument("--data", default="kitti.yaml", help="Dataset config")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="mps", help="Device: mps, cpu, or cuda")
    train.add_argument("--name", default="vehicle_v1", help="Run name")
    train.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    train.set_defaults(handler=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a model checkpoint")
    evaluate.add_argument("--model", required=True, help="Model weights path")
    evaluate.add_argument("--data", default="kitti.yaml", help="Dataset config")
    evaluate.add_argument("--name", default="eval", help="Run name")
    evaluate.add_argument("--baseline", default="yolo11n.pt", help="Baseline model for comparison")
    evaluate.add_argument("--compare", action="store_true", help="Also evaluate the baseline model")
    evaluate.set_defaults(handler=cmd_evaluate)

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
    train_remote.set_defaults(handler=cmd_train_remote)

    return parser

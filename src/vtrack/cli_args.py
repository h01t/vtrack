"""CLI argument parsing helpers and typed config adapters."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from vtrack.settings import (
    EvaluationConfig,
    InferenceConfig,
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


def parse_source(value: str) -> str | int:
    """Parse source as webcam index when numeric, else string path/URL."""
    return int(value) if value.isdigit() else value


def analytics_geometry_from_args(
    args: argparse.Namespace,
) -> tuple[tuple[Any, Any] | None, Any | None]:
    """Build optional line/polygon geometry from CLI arguments."""
    line_geometry = parse_line(args.line) if getattr(args, "line", None) else None
    polygon_geometry = parse_polygon(args.zone) if getattr(args, "zone", None) else None
    return line_geometry, polygon_geometry


def build_analytics_from_geometry(
    line_geometry: tuple[Any, Any] | None,
    polygon_geometry: Any | None,
) -> "VehicleAnalytics":
    """Construct VehicleAnalytics from parsed geometry."""
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


def build_analytics(
    args: argparse.Namespace,
    *,
    always: bool = False,
) -> "VehicleAnalytics | None":
    """Build optional analytics object depending on CLI switches."""
    enabled = always or any(
        getattr(args, field, None)
        for field in ("analytics", "line", "zone", "export_csv", "export_json")
    )
    if not enabled:
        return None

    line_geometry, polygon_geometry = analytics_geometry_from_args(args)
    return build_analytics_from_geometry(line_geometry, polygon_geometry)


def build_analytics_factory(args: argparse.Namespace):
    """Build lazy analytics factory used by benchmark workflows."""
    line_geometry, polygon_geometry = analytics_geometry_from_args(args)

    def factory():
        return build_analytics_from_geometry(line_geometry, polygon_geometry)

    return factory


def inference_config_from_args(args: argparse.Namespace) -> InferenceConfig:
    """Map CLI args into InferenceConfig with defaults for missing fields."""
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


def training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Map CLI args into TrainingConfig."""
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


def evaluation_config_from_args(args: argparse.Namespace) -> EvaluationConfig:
    """Map CLI args into EvaluationConfig."""
    return EvaluationConfig(
        model_path=args.model,
        data=args.data,
        name=args.name,
        baseline_model_path=args.baseline,
        compare=args.compare,
    )


def remote_config_from_args(args: argparse.Namespace, *, paths: ProjectPaths) -> RemoteConfig:
    """Map CLI args + env defaults into RemoteConfig."""
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

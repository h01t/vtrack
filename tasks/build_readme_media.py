#!/usr/bin/env python3
"""Build lightweight README media assets for the vtrack repository."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEMO_TRACKER = "bytetrack"
BENCHMARK_TRACKERS = ["bytetrack", "bytetrack-occlusion", "botsort"]
BENCHMARK_DEVICE = "cpu"
BENCHMARK_MAX_FRAMES = 150
BENCHMARK_WARMUP_FRAMES = 30
DEMO_MIN_CONFIDENCE = 0.25
DEMO_TRACK_CONFIDENCE = 0.10
DEMO_DURATION_SEC = 10.0
DEFAULT_SEGMENT_START_SEC = 3.0
DEFAULT_SEGMENT_DURATION_SEC = 12.0
README_VIDEO_URL = "https://github.com/h01t/vtrack/releases/download/media/demo-short.mp4"


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Source video path used to build README assets",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model weights path used for demo + benchmark runs",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for tracked README media",
    )
    parser.add_argument(
        "--segment-start-sec",
        type=float,
        default=DEFAULT_SEGMENT_START_SEC,
        help="Start time for the working source segment",
    )
    parser.add_argument(
        "--segment-duration-sec",
        type=float,
        default=DEFAULT_SEGMENT_DURATION_SEC,
        help="Duration of the working source segment",
    )
    parser.add_argument(
        "--scene-crop-top-ratio",
        type=float,
        default=1.0,
        help="Optional top-of-frame crop ratio applied before demo composition",
    )
    return parser.parse_args()


def ensure_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required binary '{name}' was not found in PATH.")
    return path


def probe_video(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()

    duration_sec = (frame_count / fps) if frame_count and fps else 0.0
    return VideoInfo(
        width=width,
        height=height,
        fps=fps or 30.0,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    completed = subprocess.run(command, check=False, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{completed.returncode}: {' '.join(command)}"
        )


def trim_source_segment(
    *,
    source_path: Path,
    destination: Path,
    start_sec: float,
    duration_sec: float,
) -> None:
    info = probe_video(source_path)
    effective_start = max(0.0, min(start_sec, max(0.0, info.duration_sec - 1.0)))
    effective_duration = min(duration_sec, max(1.0, info.duration_sec - effective_start))

    run(
        [
            ensure_binary("ffmpeg"),
            "-y",
            "-ss",
            f"{effective_start:.3f}",
            "-i",
            str(source_path),
            "-t",
            f"{effective_duration:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )


def build_line_zone(info: VideoInfo):
    import supervision as sv

    line_y = int(info.height * 0.62)
    return sv.LineZone(
        start=sv.Point(int(info.width * 0.15), line_y),
        end=sv.Point(int(info.width * 0.85), line_y),
    )


def compose_demo_video(
    *,
    annotated_video: Path,
    destination: Path,
    crop_top_ratio: float,
    duration_sec: float = DEMO_DURATION_SEC,
) -> None:
    crop_top_ratio = min(max(crop_top_ratio, 0.2), 1.0)
    info = probe_video(annotated_video)
    working_duration = min(duration_sec, info.duration_sec or duration_sec)
    crop_height = max(2, int(info.height * crop_top_ratio))
    crop_height -= crop_height % 2

    filter_graph = (
        f"[0:v]crop=in_w:{crop_height}:0:0,split=2[fgsrc][bgsrc];"
        "[bgsrc]scale=960:540:force_original_aspect_ratio=increase,"
        "boxblur=24:4,crop=960:540[bg];"
        "[fgsrc]scale=912:514:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )

    run(
        [
            ensure_binary("ffmpeg"),
            "-y",
            "-i",
            str(annotated_video),
            "-t",
            f"{working_duration:.3f}",
            "-an",
            "-filter_complex",
            filter_graph,
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "29",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "24",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )

    size_bytes = destination.stat().st_size
    if size_bytes > 10 * 1024 * 1024:
        raise RuntimeError(
            f"{destination} is {size_bytes / (1024 * 1024):.2f} MB; expected a file under 10 MB."
        )


def extract_frame(video_path: Path, *, fraction: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for frame extraction: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = max(0, min(frame_count - 1, int(frame_count * fraction))) if frame_count else 0
    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {target} from {video_path}")
    return frame


def write_png(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Unable to write PNG image to {path}")


def add_gradient_overlay(
    canvas: np.ndarray,
    *,
    top_alpha: float,
    bottom_alpha: float,
) -> np.ndarray:
    height, width = canvas.shape[:2]
    alpha = np.linspace(top_alpha, bottom_alpha, height, dtype=np.float32).reshape(height, 1, 1)
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    blended = (canvas.astype(np.float32) * (1.0 - alpha)) + (overlay.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_social_preview(hero_poster: Path, destination: Path) -> None:
    image = cv2.imread(str(hero_poster))
    if image is None:
        raise RuntimeError(f"Unable to read hero poster from {hero_poster}")

    preview = cv2.resize(image, (1280, 640), interpolation=cv2.INTER_LINEAR)
    preview = add_gradient_overlay(preview, top_alpha=0.10, bottom_alpha=0.48)
    cv2.rectangle(preview, (48, 52), (510, 242), (15, 23, 42), -1)
    cv2.rectangle(preview, (48, 52), (510, 242), (255, 255, 255), 2)

    cv2.putText(
        preview,
        "vtrack",
        (78, 134),
        cv2.FONT_HERSHEY_DUPLEX,
        2.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "Vehicle Detection &",
        (80, 176),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (229, 231, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "Tracking Pipeline",
        (80, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (229, 231, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "YOLOv11 + ByteTrack analytics",
        (78, 586),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    write_png(destination, preview)


def build_demo_assets(args: argparse.Namespace) -> None:
    from vtrack.analytics import VehicleAnalytics
    from vtrack.benchmarking import benchmark_trackers
    from vtrack.readme_media import load_benchmark_rows, render_benchmark_svg
    from vtrack.settings import InferenceConfig
    from vtrack.workflows import run_demo

    source_path = Path(args.source).resolve()
    model_path = Path(args.model).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    ensure_binary("ffmpeg")

    with tempfile.TemporaryDirectory(prefix="vtrack-readme-media-") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        source_segment = temp_dir / "source-segment.mp4"
        annotated_video = temp_dir / "annotated-full.mp4"
        frames_csv = temp_dir / "frames.csv"
        summary_json = temp_dir / "summary.json"

        trim_source_segment(
            source_path=source_path,
            destination=source_segment,
            start_sec=args.segment_start_sec,
            duration_sec=args.segment_duration_sec,
        )

        segment_info = probe_video(source_segment)
        line_zone = build_line_zone(segment_info)
        analytics = VehicleAnalytics(line_zone=line_zone)
        inference = InferenceConfig(
            model_path=str(model_path),
            min_confidence=DEMO_MIN_CONFIDENCE,
            track_conf=DEMO_TRACK_CONFIDENCE,
            tracker=DEMO_TRACKER,
        )
        run_demo(
            source=source_segment,
            inference=inference,
            analytics=analytics,
            display=False,
            save_path=str(annotated_video),
            export_csv=str(frames_csv),
            export_json=str(summary_json),
        )

        demo_short = out_dir / "demo-short.mp4"
        compose_demo_video(
            annotated_video=annotated_video,
            destination=demo_short,
            crop_top_ratio=args.scene_crop_top_ratio,
        )

        hero_frame = extract_frame(demo_short, fraction=0.45)
        tracking_frame = extract_frame(demo_short, fraction=0.68)
        analytics_frame = extract_frame(demo_short, fraction=0.82)

        hero_poster = out_dir / "hero-poster.png"
        tracking_poster = out_dir / "tracking-frame.png"
        analytics_poster = out_dir / "analytics-frame.png"
        social_preview = out_dir / "social-preview.png"
        benchmark_csv = out_dir / "benchmark.csv"
        benchmark_svg = out_dir / "benchmark-trackers.svg"

        write_png(hero_poster, hero_frame)
        write_png(tracking_poster, tracking_frame)
        write_png(analytics_poster, analytics_frame)
        build_social_preview(hero_poster, social_preview)

        def analytics_factory() -> VehicleAnalytics:
            return VehicleAnalytics(line_zone=build_line_zone(segment_info))

        benchmark_report = benchmark_trackers(
            source=source_segment,
            inference=InferenceConfig(
                model_path=str(model_path),
                min_confidence=DEMO_MIN_CONFIDENCE,
                track_conf=DEMO_TRACK_CONFIDENCE,
                device=BENCHMARK_DEVICE,
            ),
            trackers=BENCHMARK_TRACKERS,
            analytics_factory=analytics_factory,
            max_frames=BENCHMARK_MAX_FRAMES,
            warmup_frames=BENCHMARK_WARMUP_FRAMES,
            export_csv=str(benchmark_csv),
        )
        subtitle = (
            f"model={model_path.name} · device={BENCHMARK_DEVICE} · "
            f"frames={benchmark_report['max_frames'] or 'all'}"
        )
        render_benchmark_svg(
            load_benchmark_rows(benchmark_csv),
            benchmark_svg,
            subtitle=subtitle,
        )

        print("\nBuilt README media:")
        print(f"  poster: {hero_poster}")
        print(f"  tracking frame: {tracking_poster}")
        print(f"  analytics frame: {analytics_poster}")
        print(f"  demo video (upload manually): {demo_short}")
        print(f"  benchmark csv: {benchmark_csv}")
        print(f"  benchmark svg: {benchmark_svg}")
        print(f"  social preview: {social_preview}")
        print(f"  expected release asset URL: {README_VIDEO_URL}")


def main() -> int:
    args = parse_args()
    build_demo_assets(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

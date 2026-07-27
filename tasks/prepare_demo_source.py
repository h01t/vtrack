#!/usr/bin/env python3
"""Prepare a continuous demo source video and link it into the repo.

KITTI val PNGs are independent scenes — stitching them produces a slideshow
where trackers and LineZone analytics cannot work. Prefer a real continuous
clip (default: Intel OpenVINO car-detection sample under /srv/ai).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_SOURCE = Path("/srv/ai/datasets/vtrack/samples/traffic-highway.mp4")
DEFAULT_OUT = Path("/srv/ai/datasets/vtrack/samples/demo-source.mp4")
DEFAULT_LINK = Path("data/test-video.mp4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=(
            "Continuous traffic / roadway video (not KITTI stills). "
            "Default: Roboflow supervision vehicles.mp4 downscaled to 1280x720"
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--link",
        type=Path,
        default=DEFAULT_LINK,
        help="Symlink path inside the repo (gitignored)",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=0.0,
        help="Optional trim start (seconds)",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="Optional trim duration (0 = copy full source)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output width (keeps aspect ratio; 0 = no scale)",
    )
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(
            f"Missing continuous source video: {args.source}\n"
            "Download Roboflow vehicles.mp4 and downscale, e.g.:\n"
            "  curl -L -o /srv/ai/datasets/vtrack/samples/vehicles-roboflow.mp4 \\\n"
            "    https://media.roboflow.com/supervision/video-examples/vehicles.mp4\n"
            "  ffmpeg -y -i .../vehicles-roboflow.mp4 -vf scale=1280:-2 -an \\\n"
            "    .../traffic-highway.mp4"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to prepare the demo source")

    command = [ffmpeg, "-y"]
    if args.start_sec > 0:
        command.extend(["-ss", f"{args.start_sec:.3f}"])
    command.extend(["-i", str(args.source)])
    if args.duration_sec > 0:
        command.extend(["-t", f"{args.duration_sec:.3f}"])
    if args.width > 0:
        command.extend(["-vf", f"scale={args.width}:-2"])
    command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.out),
        ]
    )
    print("+", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"ffmpeg failed with exit code {completed.returncode}")

    args.link.parent.mkdir(parents=True, exist_ok=True)
    if args.link.exists() or args.link.is_symlink():
        args.link.unlink()
    args.link.symlink_to(args.out.resolve())
    print(f"Wrote {args.out} -> {args.link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

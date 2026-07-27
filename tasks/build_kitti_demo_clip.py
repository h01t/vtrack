#!/usr/bin/env python3
"""Deprecated: KITTI val PNGs are not a temporal video.

Stitching sorted validation stills produces a slideshow — track IDs thrash and
LineZone crossings are meaningless. Use ``tasks/prepare_demo_source.py`` with a
continuous traffic clip instead.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "build_kitti_demo_clip.py is deprecated.\n"
        "KITTI val frames are independent scenes, not continuous video.\n"
        "Use:\n"
        "  uv run python tasks/prepare_demo_source.py\n"
        "  # default source: /srv/ai/datasets/vtrack/samples/traffic-highway.mp4",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

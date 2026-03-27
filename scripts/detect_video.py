"""Compatibility wrapper for the unified vtrack detect-video CLI."""

from vtrack.cli import detect_video_main


def main() -> int:
    return detect_video_main()


if __name__ == "__main__":
    raise SystemExit(main())

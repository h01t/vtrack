"""Compatibility wrapper for the unified vtrack detect-image CLI."""

from vtrack.cli import detect_image_main


def main() -> int:
    return detect_image_main()


if __name__ == "__main__":
    raise SystemExit(main())

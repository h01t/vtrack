"""Compatibility wrapper for the unified vtrack demo CLI."""

from vtrack.cli import demo_main


def main() -> int:
    return demo_main()


if __name__ == "__main__":
    raise SystemExit(main())

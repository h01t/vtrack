"""Compatibility wrapper for the unified vtrack train CLI."""

from vtrack.cli import train_main


def main() -> int:
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())

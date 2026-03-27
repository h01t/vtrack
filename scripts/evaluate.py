"""Compatibility wrapper for the unified vtrack evaluate CLI."""

from vtrack.cli import evaluate_main


def main() -> int:
    return evaluate_main()


if __name__ == "__main__":
    raise SystemExit(main())

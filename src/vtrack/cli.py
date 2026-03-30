"""Unified command-line interface for the vtrack project."""

from __future__ import annotations

import sys

from vtrack.cli_args import inference_config_from_args
from vtrack.cli_parser import build_parser
from vtrack.settings import InferenceDeviceError

_inference_config_from_args = inference_config_from_args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    command_argv = ["vtrack", *((argv if argv is not None else sys.argv[1:]))]
    args = parser.parse_args(argv)
    args.command_argv = command_argv
    try:
        return args.handler(args)
    except InferenceDeviceError as exc:
        parser.exit(status=2, message=f"error: {exc}\n")


def demo_main(argv: list[str] | None = None) -> int:
    return main(["demo", *(argv or sys.argv[1:])])


def detect_image_main(argv: list[str] | None = None) -> int:
    return main(["detect-image", *(argv or sys.argv[1:])])


def detect_video_main(argv: list[str] | None = None) -> int:
    return main(["detect-video", *(argv or sys.argv[1:])])


def train_main(argv: list[str] | None = None) -> int:
    return main(["train", *(argv or sys.argv[1:])])


def evaluate_main(argv: list[str] | None = None) -> int:
    return main(["evaluate", *(argv or sys.argv[1:])])


def train_remote_main(argv: list[str] | None = None) -> int:
    return main(["train-remote", *(argv or sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())

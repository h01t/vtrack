import subprocess
import sys
from pathlib import Path

from vtrack.cli import _inference_config_from_args, build_parser

ROOT = Path(__file__).resolve().parents[1]
VTRACK_BIN = Path(sys.executable).with_name("vtrack")


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_vtrack_root_help() -> None:
    assert VTRACK_BIN.exists()
    completed = run_command([str(VTRACK_BIN), "--help"])

    assert completed.returncode == 0
    assert "benchmark-track" in completed.stdout
    assert "detect-image" in completed.stdout
    assert "train-remote" in completed.stdout


def test_vtrack_subcommand_help() -> None:
    completed = run_command([str(VTRACK_BIN), "benchmark-track", "--help"])

    assert completed.returncode == 0
    assert "--warmup-frames" in completed.stdout
    assert "--tracker" in completed.stdout


def test_demo_parser_maps_runtime_inference_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "demo",
            "data/test-video.mp4",
            "--confidence",
            "0.33",
            "--track-conf",
            "0.12",
            "--tracker",
            "bytetrack-occlusion",
            "--device",
            "cpu",
            "--imgsz",
            "960",
            "--iou",
            "0.5",
            "--max-det",
            "42",
            "--half",
            "--vid-stride",
            "2",
            "--stream-buffer",
            "--agnostic-nms",
        ]
    )

    inference = _inference_config_from_args(args)

    assert inference.min_confidence == 0.33
    assert inference.confidence == 0.33
    assert inference.track_conf == 0.12
    assert inference.tracker == "bytetrack-occlusion"
    assert inference.device == "cpu"
    assert inference.imgsz == 960
    assert inference.iou == 0.5
    assert inference.max_det == 42
    assert inference.half is True
    assert inference.vid_stride == 2
    assert inference.stream_buffer is True
    assert inference.agnostic_nms is True


def test_benchmark_parser_collects_multiple_trackers() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-track",
            "data/test-video.mp4",
            "--tracker",
            "bytetrack",
            "--tracker",
            "botsort",
        ]
    )

    assert args.trackers == ["bytetrack", "botsort"]


def test_demo_wrapper_delegates_to_cli() -> None:
    completed = run_command([sys.executable, "scripts/demo.py", "--help"])

    assert completed.returncode == 0
    assert "--analytics" in completed.stdout


def test_train_wrapper_delegates_to_cli() -> None:
    completed = run_command([sys.executable, "scripts/train.py", "--help"])

    assert completed.returncode == 0
    assert "--epochs" in completed.stdout


def test_evaluate_wrapper_delegates_to_cli() -> None:
    completed = run_command([sys.executable, "scripts/evaluate.py", "--help"])

    assert completed.returncode == 0
    assert "--baseline" in completed.stdout


def test_remote_shell_wrapper_delegates_to_cli() -> None:
    completed = run_command(["bash", "scripts/train_remote.sh", "--help"])

    assert completed.returncode == 0
    assert "train-remote" in completed.stdout

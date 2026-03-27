import subprocess
import sys
from pathlib import Path

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
    assert "detect-image" in completed.stdout
    assert "train-remote" in completed.stdout


def test_vtrack_subcommand_help() -> None:
    completed = run_command([str(VTRACK_BIN), "evaluate", "--help"])

    assert completed.returncode == 0
    assert "--compare" in completed.stdout


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

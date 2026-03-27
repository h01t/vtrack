"""Remote training orchestration helpers."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass

from vtrack.settings import ProjectPaths, RemoteConfig, TrainingConfig


@dataclass(frozen=True)
class RemoteCommandSet:
    """Concrete commands for a remote training session."""

    prepare: list[str]
    push: list[str]
    train: list[str]
    pull_artifacts: list[str]
    pull_models: list[str]


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _remote_shell_path(remote_dir: str) -> str:
    remote_dir = remote_dir.rstrip("/")
    if remote_dir == "~":
        return "~"
    if remote_dir.startswith("~/"):
        return f"~/{shlex.quote(remote_dir[2:])}"
    return shlex.quote(remote_dir)


def _build_remote_train_shell_command(
    *,
    remote_shell_dir: str,
    remote_datasets_dir: str,
    remote_python: str,
    cli_args: list[str],
) -> str:
    quoted_python = shlex.quote(remote_python)
    quoted_args = _shell_join(cli_args)
    quoted_remote_datasets_dir = shlex.quote(remote_datasets_dir)
    resolve_datasets_dir = (
        'DATASETS_DIR=$("$REMOTE_PYTHON" -c '
        f"'import os, sys; print(os.path.expanduser(sys.argv[1]))' {quoted_remote_datasets_dir})"
    )
    apply_ultralytics_settings = (
        'PYTHONPATH=src "$REMOTE_PYTHON" -c '
        '\'from ultralytics import settings; import sys; '
        'settings.update({"datasets_dir": sys.argv[1]})\' '
        '"$DATASETS_DIR"'
    )
    return (
        f"cd {remote_shell_dir} && "
        f"REMOTE_PYTHON={quoted_python} && "
        'if [ -x .venv/bin/python ]; then REMOTE_PYTHON=.venv/bin/python; fi && '
        f"{resolve_datasets_dir} && "
        'mkdir -p "$DATASETS_DIR" && '
        f"{apply_ultralytics_settings} && "
        'PYTHONPATH=src "$REMOTE_PYTHON" '
        f"{quoted_args}"
    )


def build_remote_command_set(
    *,
    paths: ProjectPaths,
    training: TrainingConfig,
    remote: RemoteConfig,
) -> RemoteCommandSet:
    """Build rsync and ssh commands for a remote training run."""
    host = remote.require_host()
    remote_dir = remote.remote_dir or f"~/{paths.root.name}"
    remote_datasets_dir = remote.datasets_dir or "~/datasets"
    remote_shell_dir = _remote_shell_path(remote_dir)
    remote_datasets_shell_dir = _remote_shell_path(remote_datasets_dir)

    prepare = ["ssh", host, f"mkdir -p {remote_shell_dir} {remote_datasets_shell_dir}"]

    push = [
        "rsync",
        "-avz",
        "--exclude",
        ".venv",
        "--exclude",
        "data",
        "--exclude",
        "models",
        "--exclude",
        "outputs",
        "--exclude",
        "artifacts",
        "--exclude",
        "runs",
        "--exclude",
        ".pytest_cache",
        "--exclude",
        ".ruff_cache",
        "--exclude",
        "__pycache__",
        "--exclude",
        "*.egg-info",
        "--exclude",
        ".DS_Store",
        "--exclude",
        ".git",
        f"{paths.root}/",
        f"{host}:{remote_dir}/",
    ]

    train_args = [
        "-m",
        "vtrack.cli",
        "train",
        "--model",
        training.model_path,
        "--data",
        training.data,
        "--epochs",
        str(training.epochs),
        "--imgsz",
        str(training.imgsz),
        "--batch",
        str(training.batch),
        "--device",
        training.device,
        "--name",
        training.name,
    ]
    if not training.amp:
        train_args.append("--no-amp")
    train = [
        "ssh",
        host,
        _build_remote_train_shell_command(
            remote_shell_dir=remote_shell_dir,
            remote_datasets_dir=remote_datasets_dir,
            remote_python=remote.remote_python,
            cli_args=train_args,
        ),
    ]

    pull_artifacts = [
        "rsync",
        "-avz",
        f"{host}:{remote_dir}/artifacts/train/{training.name}/",
        f"{paths.train_artifacts_dir / training.name}/",
    ]
    pull_models = [
        "rsync",
        "-avz",
        f"{host}:{remote_dir}/models/",
        f"{paths.models_dir}/",
    ]

    return RemoteCommandSet(
        prepare=prepare,
        push=push,
        train=train,
        pull_artifacts=pull_artifacts,
        pull_models=pull_models,
    )


def run_remote_training_commands(commands: RemoteCommandSet) -> None:
    """Execute remote training commands sequentially."""
    for command in (
        commands.prepare,
        commands.push,
        commands.train,
        commands.pull_artifacts,
        commands.pull_models,
    ):
        subprocess.run(command, check=True)

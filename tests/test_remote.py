from pathlib import Path

from vtrack.remote import build_remote_command_set
from vtrack.settings import ProjectPaths, RemoteConfig, TrainingConfig


def test_build_remote_command_set_uses_normalized_artifact_paths(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    training = TrainingConfig(
        model_path="yolo11n.pt",
        data="/datasets/kitti.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda",
        name="vehicle_v1",
        amp=True,
    )
    remote = RemoteConfig(
        host="gpu-box",
        remote_dir="~/object-det",
        datasets_dir="~/datasets",
        remote_python="/opt/venv/bin/python",
    )

    commands = build_remote_command_set(paths=paths, training=training, remote=remote)

    assert commands.prepare == ["ssh", "gpu-box", "mkdir -p ~/object-det ~/datasets"]
    assert commands.push[:2] == ["rsync", "-avz"]
    assert "outputs" in commands.push
    assert ".pytest_cache" in commands.push
    assert ".ruff_cache" in commands.push
    assert "*.egg-info" in commands.push
    assert commands.train[0] == "ssh"
    assert "cd ~/object-det &&" in commands.train[2]
    assert (
        'if [ -x .venv/bin/python ]; then REMOTE_PYTHON=.venv/bin/python; fi'
        in commands.train[2]
    )
    assert 'DATASETS_DIR=$("$REMOTE_PYTHON" -c ' in commands.train[2]
    assert "'~/datasets')" in commands.train[2]
    assert 'settings.update({"datasets_dir": sys.argv[1]})' in commands.train[2]
    assert 'PYTHONPATH=src "$REMOTE_PYTHON" -m vtrack.cli train' in commands.train[2]
    assert f"artifacts/train/{training.name}/" in commands.pull_artifacts[2]
    assert commands.pull_models[2] == "gpu-box:~/object-det/models/"


def test_build_remote_command_set_preserves_home_relative_subdirs(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    training = TrainingConfig(name="vehicle_v2")
    remote = RemoteConfig(
        host="gpu-box",
        remote_dir="~/Dev/object-det",
        datasets_dir="~/Dev/datasets",
        remote_python="python3",
    )

    commands = build_remote_command_set(paths=paths, training=training, remote=remote)

    assert commands.prepare == ["ssh", "gpu-box", "mkdir -p ~/Dev/object-det ~/Dev/datasets"]
    assert commands.push[-1] == "gpu-box:~/Dev/object-det/"
    assert "cd ~/Dev/object-det &&" in commands.train[2]
    assert "'~/Dev/datasets')" in commands.train[2]
    assert "REMOTE_PYTHON=python3" in commands.train[2]

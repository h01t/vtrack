from pathlib import Path

from vtrack.settings import (
    ProjectPaths,
    RemoteConfig,
    default_remote_datasets_dir,
    normalize_remote_dir,
)


def test_project_paths_expose_normalized_layout(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)

    assert paths.models_dir == tmp_path / "models"
    assert paths.train_artifacts_dir == tmp_path / "artifacts" / "train"
    assert paths.eval_artifacts_dir == tmp_path / "artifacts" / "eval"
    assert paths.raw_training_dir == tmp_path / "runs" / "training"
    assert paths.raw_evaluation_dir == tmp_path / "runs" / "evaluation"


def test_remote_config_prefers_environment_values() -> None:
    config = RemoteConfig.from_env(
        {
            "VTRACK_REMOTE_HOST": "gpu-box",
            "VTRACK_REMOTE_DIR": "~/remote-project",
            "VTRACK_REMOTE_DATASETS_DIR": "~/remote-datasets",
            "VTRACK_REMOTE_PYTHON": "/opt/venv/bin/python",
        },
        project_name="ignored-name",
    )

    assert config.host == "gpu-box"
    assert config.remote_dir == "~/remote-project"
    assert config.datasets_dir == "~/remote-datasets"
    assert config.remote_python == "/opt/venv/bin/python"


def test_remote_config_defaults_remote_dir_from_project_name() -> None:
    config = RemoteConfig.from_env({}, project_name="object-det")

    assert config.host is None
    assert config.remote_dir == "~/object-det"
    assert config.datasets_dir == "~/datasets"
    assert config.remote_python == "python3"


def test_remote_config_defaults_to_home_relative_project_path(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "Dev" / "object-det"
    project_root.mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    config = RemoteConfig.from_env({}, project_name="object-det", project_root=project_root)

    assert config.remote_dir == "~/Dev/object-det"
    assert config.datasets_dir == "~/Dev/datasets"


def test_normalize_remote_dir_compresses_local_home_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    normalized = normalize_remote_dir(
        str(tmp_path / "object-det"),
        project_name="object-det",
    )

    assert normalized == "~/object-det"


def test_default_remote_datasets_dir_tracks_remote_checkout_parent() -> None:
    assert (
        default_remote_datasets_dir("~/Dev/object-det", project_name="object-det")
        == "~/Dev/datasets"
    )
    assert default_remote_datasets_dir("~/object-det", project_name="object-det") == "~/datasets"

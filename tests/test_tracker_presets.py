from pathlib import Path

from vtrack.tracker_presets import available_tracker_presets, resolve_tracker_config


def test_available_tracker_presets_include_occlusion_variant() -> None:
    assert available_tracker_presets() == ("bytetrack", "bytetrack-occlusion", "botsort")


def test_resolve_builtin_tracker_preset() -> None:
    resolved = resolve_tracker_config("bytetrack")

    assert resolved.is_builtin is True
    assert resolved.name == "bytetrack"
    assert Path(resolved.path).name == "bytetrack.yaml"


def test_resolve_tracker_compatibility_alias() -> None:
    resolved = resolve_tracker_config("botsort.yaml")

    assert resolved.is_builtin is True
    assert resolved.name == "botsort"
    assert Path(resolved.path).name == "botsort.yaml"


def test_resolve_explicit_tracker_path(tmp_path: Path) -> None:
    tracker_path = tmp_path / "custom-tracker.yaml"
    tracker_path.write_text("tracker_type: bytetrack\n", encoding="utf-8")

    resolved = resolve_tracker_config(str(tracker_path))

    assert resolved.is_builtin is False
    assert resolved.path == str(tracker_path.resolve())

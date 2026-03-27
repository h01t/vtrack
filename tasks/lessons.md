# Lessons Learned

_Updated as corrections and insights arise._

- When wrapping third-party CV annotation APIs, verify constructor and method signatures against the installed version before refactoring for reuse. In this project, `supervision.PolygonZoneAnnotator` binds the zone in `__init__`, so integration changes should carry a regression test for the zone-enabled pipeline path.
- For remote shell paths supplied through environment variables, assume `~` may already have been expanded by the local shell. Normalize local-home absolute paths back to `~/...`, and create the remote directory explicitly before `rsync` so mirrored paths like `~/Dev/object-det` work on fresh machines.
- When invoking code on a remote checkout copied by `rsync`, do not assume the package is installed there. Run from the checkout with `PYTHONPATH=src`, and prefer the repo-local `.venv/bin/python` when present so dependencies match the project environment.
- Remote training must not inherit stale Ultralytics global settings from previous checkouts. For builtin dataset aliases like `kitti.yaml`, explicitly set `datasets_dir` for the remote run based on the current remote workspace layout or an explicit override such as `VTRACK_REMOTE_DATASETS_DIR`.

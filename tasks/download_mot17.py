#!/usr/bin/env python3
"""Download a MOT17 PoC subset under /srv/ai/datasets/mot17.

Primary source is a Hugging Face mirror (motchallenge.net is often blocked).
Keeps only FRCNN folders for MOT17-02 / 04 / 09 by default.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_ROOT = Path("/srv/ai/datasets/mot17")
HF_REPO = "Lekim89/MOT17"
POC_SEQUENCES = ("MOT17-02", "MOT17-04", "MOT17-09")


def _keep_only_poc(train: Path) -> None:
    keep = {f"{seq}-FRCNN" for seq in POC_SEQUENCES}
    for path in list(train.iterdir()):
        if path.is_dir() and path.name not in keep:
            shutil.rmtree(path)
            print(f"removed {path.name}")


def download_from_huggingface(root: Path, *, keep_only_poc: bool) -> None:
    from huggingface_hub import snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    patterns = (
        [f"train/{seq}-FRCNN/**" for seq in POC_SEQUENCES]
        if keep_only_poc
        else ["train/**"]
    )
    print(f"Downloading {HF_REPO} patterns={patterns} -> {root}")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=patterns,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Existing MOT17.zip path (offline extract; skips Hugging Face)",
    )
    parser.add_argument(
        "--keep-only-poc",
        action="store_true",
        default=True,
        help="Keep only FRCNN folders for MOT17-02/04/09 (default: true)",
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Download full train split instead of PoC subset",
    )
    args = parser.parse_args()
    keep_only_poc = not args.all_train

    if args.zip is not None:
        import zipfile

        if not args.zip.is_file():
            raise SystemExit(f"Missing MOT17 zip: {args.zip}")
        args.root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip) as archive:
            archive.extractall(args.root)
        if (args.root / "MOT17" / "train").is_dir() and not (args.root / "train").is_dir():
            (args.root / "MOT17" / "train").rename(args.root / "train")
            shutil.rmtree(args.root / "MOT17", ignore_errors=True)
    else:
        download_from_huggingface(args.root, keep_only_poc=keep_only_poc)

    train = args.root / "train"
    if not train.is_dir():
        # Some HF layouts nest MOT17/train
        nested = args.root / "MOT17" / "train"
        if nested.is_dir():
            args.root.mkdir(parents=True, exist_ok=True)
            nested.rename(train)
            shutil.rmtree(args.root / "MOT17", ignore_errors=True)
        else:
            raise SystemExit(f"Expected train/ under {args.root} after download")

    if keep_only_poc:
        _keep_only_poc(train)

    print(f"MOT17 ready at {args.root}")
    print("Sequences:", ", ".join(sorted(p.name for p in train.iterdir() if p.is_dir())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

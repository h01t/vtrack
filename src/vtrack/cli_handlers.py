"""CLI command handlers."""

from __future__ import annotations

import argparse
import json

from vtrack.cli_args import (
    build_analytics,
    build_analytics_factory,
    evaluation_config_from_args,
    inference_config_from_args,
    parse_source,
    remote_config_from_args,
    training_config_from_args,
)
from vtrack.settings import ProjectPaths


def cmd_demo(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_demo

    run_demo(
        source=parse_source(args.source),
        inference=inference_config_from_args(args),
        analytics=build_analytics(args),
        display=not args.no_display,
        save_path=args.save,
        export_csv=args.export_csv,
        export_json=args.export_json,
    )
    return 0


def cmd_benchmark_track(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_tracking_benchmark

    report = run_tracking_benchmark(
        source=parse_source(args.source),
        inference=inference_config_from_args(args),
        trackers=args.trackers,
        analytics_factory=build_analytics_factory(args),
        max_frames=args.max_frames,
        warmup_frames=args.warmup_frames,
        export_csv=args.export_csv,
    )
    print(json.dumps(report, indent=2))
    return 0


def cmd_detect_image(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_detect_image

    results = run_detect_image(
        source=args.source,
        inference=inference_config_from_args(args),
        save=not args.no_save,
    )
    for result in results:
        boxes = result.boxes
        print(f"\nDetected {len(boxes)} object(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names.get(cls_id, f"class_{cls_id}")
            print(f"  - {name}: {conf:.2f}")
        if not args.no_show:
            result.show()
    return 0


def cmd_detect_video(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_detect_video

    frame_count = 0
    for result in run_detect_video(
        source=args.source,
        inference=inference_config_from_args(args),
        save=args.save,
        stream=True,
    ):
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(result.boxes)} detections")
    print(f"\nProcessed {frame_count} frames total.")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_training

    paths = ProjectPaths()
    training = training_config_from_args(args)
    _, bundle = run_training(
        training=training,
        paths=paths,
        command=args.command_argv,
    )
    print("\n--- Training Complete ---")
    print(f"Artifact bundle: {bundle.bundle_dir}")
    print(f"Manifest: {bundle.manifest_path}")
    print(f"Summary: {bundle.summary_path}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_evaluation

    paths = ProjectPaths()
    evaluation = evaluation_config_from_args(args)
    result = run_evaluation(
        evaluation=evaluation,
        paths=paths,
        command=args.command_argv,
    )

    summary = result["summary"]
    finetuned = summary["finetuned"]["overall"]
    print(f"\n--- {evaluation.name}_finetuned ---")
    print(f"mAP@0.5:      {finetuned['map50']:.4f}")
    print(f"mAP@0.5:0.95: {finetuned['map50_95']:.4f}")
    print(f"Precision:     {finetuned['precision']:.4f}")
    print(f"Recall:        {finetuned['recall']:.4f}")
    if evaluation.compare and "delta" in summary:
        delta = summary["delta"]
        print("\n--- Comparison ---")
        print(f"mAP@0.5 delta:      {delta['map50']:+.4f}")
        print(f"mAP@0.5:0.95 delta: {delta['map50_95']:+.4f}")

    bundle = result["bundle"]
    print(f"\nArtifact bundle: {bundle.bundle_dir}")
    print(f"Manifest: {bundle.manifest_path}")
    print(f"Summary: {bundle.summary_path}")
    return 0


def cmd_train_remote(args: argparse.Namespace) -> int:
    from vtrack.workflows import run_remote_training

    paths = ProjectPaths()
    training = training_config_from_args(args)
    remote = remote_config_from_args(args, paths=paths)
    run_remote_training(training=training, remote=remote, paths=paths)
    print(f"Remote training completed for run '{training.name}'.")
    print(f"Synced artifacts: {paths.train_artifacts_dir / training.name}")
    return 0

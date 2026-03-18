"""
Core computer vision pipeline wiring together:

1. Video ingestion (OpenCV, sample every 0.5s).
2. Object detection (YOLOv8) for people and vehicles.
3. Multi-object tracking (DeepSORT-style) to assign IDs and estimate suspects.
4. Event analysis engine to summarize detection events.

This module can be used directly as a script for local testing:

    python -m backend.pipeline path/to/video.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from backend.notifications.console_notifier import ConsoleNotifier
from backend.ml_intrusion.inference import predict_sequence, DEFAULT_THRESHOLD
from backend.ml_intrusion.data_loader import list_split_sequences


def run_pipeline(video_path: str, *, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Runs the end-to-end CV pipeline on a single video and returns
    a high-level analysis dictionary.

    Parameters
    ----------
    video_path:
        Path to the local video file to analyze.

    Returns
    -------
    dict
        High-level analysis, including estimated suspect count.
    """
    notifier = ConsoleNotifier()

    # Model-driven inference only (no rule-based intrusion logic).
    # For frame datasets, video_path should be a directory containing the frames for one sequence.
    if Path(video_path).is_dir():
        import os

        frame_paths = [
            os.path.join(video_path, f)
            for f in sorted(os.listdir(video_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        probability = predict_sequence(Path(video_path).name, frame_paths, threshold=threshold)
    else:
        # If a user passes a raw string id, attempt to locate it in the test split.
        seq_id = Path(video_path).stem
        sequences = list_split_sequences("test")
        match = next((s for s in sequences if s.sequence_id.lower() == seq_id.lower()), None)
        if match is None:
            raise FileNotFoundError(
                "Provide either a directory of frames for one sequence, "
                "or a sequence id that exists in the test split."
            )
        probability = predict_sequence(match.sequence_id, match.frame_paths, threshold=threshold)

    notifier.notify_intrusion_probability(
        video_path=video_path,
        probability=probability,
        threshold=threshold,
    )

    return {
        "video": video_path,
        "intrusion_probability": probability,
        "threshold": threshold,
        "is_intrusion": bool(probability > threshold),
    }


def main(argv: list[str] | None = None) -> None:
    """
    Simple CLI entry point for local experimentation.
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python -m backend.pipeline <video_path> [threshold]")
        raise SystemExit(1)

    video_path = argv[0]
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        raise SystemExit(1)

    thr = float(argv[1]) if len(argv) >= 2 else DEFAULT_THRESHOLD
    result = run_pipeline(video_path, threshold=thr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


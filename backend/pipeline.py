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
from backend.ml_intrusion.inference import DEFAULT_THRESHOLD, predict_frames, predict_sequence, predict_video
from backend.video.ingestion import VideoIngestion
from backend.detection.yolo_detector import YoloV8Detector


def _iter_frame_sequence(video_path: str) -> List:
    """
    Accept either:
      - a directory of image frames (sorted by filename)
      - a video file path

    Returns a bounded list of sampled frames.
    """
    path = Path(video_path)
    if path.is_dir():
        import os

        frame_paths = [
            os.path.join(video_path, f)
            for f in sorted(os.listdir(video_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]

        # Lightweight sampling: keep at most 20 frames from the directory.
        max_frames = 40
        stride = max(len(frame_paths) // max_frames, 1) if frame_paths else 1
        import cv2

        frames: List = []
        for i in range(0, len(frame_paths), stride):
            if len(frames) >= max_frames:
                break
            img = cv2.imread(frame_paths[i])
            if img is not None:
                frames.append(img)
        return frames

    # Video file: use OpenCV sampling via VideoIngestion.
    ingestion = VideoIngestion(video_path=str(video_path), sample_interval=0.5)
    frames: List = []
    for frame_data in ingestion.iter_frames():
        frames.append(frame_data.image)
        if len(frames) >= 40:
            break
    return frames


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

    detector = YoloV8Detector()

    print(f"Processing video: {video_path}")
    # Sample frames once (bounded to max_frames=20) and reuse them for both:
    # 1) YOLO activity detection
    # 2) intrusion classification
    sampled_frames = _iter_frame_sequence(video_path)

    # Build "activity" summary from YOLO detections (no intrusion rules here).
    activity: Dict[str, Any] = {
        "frames_analyzed": len(sampled_frames),
        "people_detections": 0,
        "vehicle_detections": 0,
        "per_frame": [],
    }

    # For directory-based inputs we don't have real timestamps; use indices.
    for idx, frame in enumerate(sampled_frames):
        timestamp = idx * 0.5
        det_event = detector.detect_frame(frame=frame, timestamp=timestamp)
        objects = det_event.get("objects", [])

        people_count = sum(1 for o in objects if o.get("type") == "person")
        vehicle_count = sum(1 for o in objects if o.get("type") == "vehicle")

        activity["people_detections"] += people_count
        activity["vehicle_detections"] += vehicle_count

        activity["per_frame"].append(
            {
                "timestamp": det_event.get("timestamp"),
                "objects": objects,
            }
        )

    # Model-driven intrusion decision ONLY:
    probability = predict_frames(
        sampled_frames,
        threshold=threshold,
        print_alert=False,
        verbose=False,
    )

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
        "activity": activity,
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


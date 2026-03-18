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

from backend.video.ingestion import VideoIngestion
from backend.detection.yolo_detector import YoloV8Detector
from backend.tracking.deepsort_tracker import DeepSortTracker
from backend.analysis.event_engine import EventAnalysisEngine


def run_pipeline(video_path: str) -> Dict[str, Any]:
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
    ingestion = VideoIngestion(video_path=video_path, sample_interval=.5)
    detector = YoloV8Detector()
    tracker = DeepSortTracker()
    analysis_engine = EventAnalysisEngine()

    tracking_events: List[Dict[str, Any]] = []

    # 1) Ingest sampled frames from the video.
    for frame_data in ingestion.iter_frames():
        frame = frame_data.image
        timestamp = frame_data.timestamp

        # 2) Run YOLOv8 detection on the current frame.
        detection_event = detector.detect_frame(frame=frame, timestamp=timestamp)

        # 3) Update DeepSORT tracker with detections and get tracking event.
        tracking_event = tracker.update(frame=frame, detection_event=detection_event)

        tracking_events.append(tracking_event)

    # 4) Pass all tracking events to the event analysis engine.
    analysis_result = analysis_engine.analyze_events(tracking_events)
    return analysis_result


def main(argv: list[str] | None = None) -> None:
    """
    Simple CLI entry point for local experimentation.
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python -m backend.pipeline <video_path>")
        raise SystemExit(1)

    video_path = argv[0]
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        raise SystemExit(1)

    result = run_pipeline(video_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


"""Core streaming CV pipeline for learned intrusion inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

from backend.analysis.learning_event_inference import LearningBasedEventInference
from backend.notifications.console_notifier import ConsoleNotifier
from backend.video.ingestion import VideoIngestion
from backend.detection.yolo_detector import YoloV8Detector
from backend.tracking.deepsort_tracker import DeepSortTracker


def run_pipeline(
    video_path: str,
    *,
    sample_interval: float = 0.5,
    event_model_path: str = "intrusion_event_model.pt",
) -> Dict[str, Any]:
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
        Learned intrusion analysis with unique tracked counts.
    """
    notifier = ConsoleNotifier()
    detector = YoloV8Detector()
    tracker = DeepSortTracker()
    event_inference = LearningBasedEventInference(model_path=event_model_path)

    print(f"Processing video: {video_path}")
    ingestion = VideoIngestion(video_path=str(video_path), sample_interval=sample_interval)
    frames_analyzed = 0
    last_timestamp = 0.0

    for frame_data in ingestion.iter_frames():
        frames_analyzed += 1
        frame = frame_data.image
        timestamp = float(frame_data.timestamp)
        last_timestamp = timestamp

        det_event = detector.detect_frame(frame=frame, timestamp=timestamp)
        track_event = tracker.update(frame=frame, detection_event=det_event)
        event_inference.update(
            frame_shape=frame.shape,
            timestamp=timestamp,
            tracked_objects=track_event["objects"],
        )

    inference_result = event_inference.finalize()
    probability = float(inference_result["intrusion_probability"])
    threshold = float(inference_result["threshold"])
    is_intrusion = bool(inference_result["is_intrusion"])

    notifier.notify_intrusion_probability(video_path=video_path, probability=probability, threshold=threshold)

    return {
        "video": video_path,
        "frames_analyzed": frames_analyzed,
        "last_timestamp": last_timestamp,
        "intrusion_probability": probability,
        "threshold": threshold,
        "is_intrusion": is_intrusion,
        "activity": {
            "people_detections": int(inference_result["activity"]["people_detections"]),
            "vehicle_detections": int(inference_result["activity"]["vehicle_detections"]),
            "sequence_length": int(inference_result["sequence_length"]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    """
    Simple CLI entry point for local experimentation.
    """
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python -m backend.pipeline <video_path> [event_model_path]")
        raise SystemExit(1)

    video_path = argv[0]
    if not Path(video_path).exists():
        print(f"Video file not found: {video_path}")
        raise SystemExit(1)

    model_path = argv[1] if len(argv) >= 2 else "intrusion_event_model.pt"
    result = run_pipeline(video_path, event_model_path=model_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


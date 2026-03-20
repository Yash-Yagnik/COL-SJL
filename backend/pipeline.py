"""
Context-aware event intelligence pipeline:

Frame extraction → YOLO → tracking → semantic zones → track repository
→ timeline → contextual sequence model → reasoning → law-enforcement summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.analysis.contextual_features import zone_counts_from_tracks
from backend.analysis.intrusion_reasoning import build_intrusion_reasoning, merge_reasoning_with_model_notes
from backend.analysis.learning_event_inference import LearningBasedEventInference
from backend.analysis.semantic_zones import default_semantic_zones, primary_semantic_zone, resolve_zones
from backend.analysis.timeline_builder import build_timeline
from backend.analysis.track_repository import TrackRepository
from backend.notifications.console_notifier import ConsoleNotifier
from backend.reporting.law_enforcement_report import generate_law_enforcement_summary
from backend.video.ingestion import VideoIngestion
from backend.detection.yolo_detector import YoloV8Detector
from backend.tracking.deepsort_tracker import DeepSortTracker


def _compact_tracks(repo: TrackRepository) -> Dict[str, Any]:
    out: Dict[str, Any] = {"people": {}, "vehicles": {}}
    for pid, rec in repo.all_people().items():
        out["people"][pid] = {
            "person_id": pid,
            "entry_timestamp_sec": rec.entry_timestamp_sec,
            "exit_timestamp_sec": rec.exit_timestamp_sec,
            "entry_frame_index": rec.entry_frame_index,
            "exit_frame_index": rec.exit_frame_index,
            "trajectory": [
                {
                    "t": p["t"],
                    "frame_index": p["frame_index"],
                    "cx": p["cx"],
                    "cy": p["cy"],
                    "zone": p.get("zone"),
                    "velocity_px_s": p.get("velocity_px_s"),
                    "velocity_vector_px_s": p.get("velocity_vector_px_s"),
                }
                for p in rec.trajectory
            ],
        }
    for vid, rec in repo.all_vehicles().items():
        out["vehicles"][vid] = {
            "vehicle_id": vid,
            "entry_timestamp_sec": rec.entry_timestamp_sec,
            "exit_timestamp_sec": rec.exit_timestamp_sec,
            "entry_frame_index": rec.entry_frame_index,
            "exit_frame_index": rec.exit_frame_index,
            "trajectory": [
                {
                    "t": p["t"],
                    "frame_index": p["frame_index"],
                    "cx": p["cx"],
                    "cy": p["cy"],
                    "zone": p.get("zone"),
                    "velocity_px_s": p.get("velocity_px_s"),
                    "velocity_vector_px_s": p.get("velocity_vector_px_s"),
                }
                for p in rec.trajectory
            ],
        }
    return out


def run_pipeline(
    video_path: str,
    *,
    sample_interval: float = 0.5,
    event_model_path: str = "intrusion_event_model.pt",
) -> Dict[str, Any]:
    """
    Analyze a video file and return a law-enforcement-ready structured result.
    """
    notifier = ConsoleNotifier()
    detector = YoloV8Detector()
    tracker = DeepSortTracker()
    repo = TrackRepository()
    event_inference = LearningBasedEventInference(model_path=event_model_path)

    zones_def = default_semantic_zones()
    zones_px: Optional[Dict[str, Any]] = None

    print(f"Processing video: {video_path}")
    ingestion = VideoIngestion(video_path=str(video_path), sample_interval=sample_interval)
    frames_analyzed = 0
    last_timestamp = 0.0
    frame_width = 0
    frame_height = 0

    for frame_data in ingestion.iter_frames():
        frames_analyzed += 1
        frame = frame_data.image
        timestamp = float(frame_data.timestamp)
        last_timestamp = timestamp
        frame_idx = int(frame_data.frame_index)
        h, w = frame.shape[:2]
        frame_height, frame_width = h, w
        if zones_px is None:
            zones_px = resolve_zones(zones_def, w, h)

        def zone_for_bbox(bbox: List[float]) -> Optional[str]:
            return primary_semantic_zone(tuple(float(x) for x in bbox), zones_px)

        det_event = detector.detect_frame(frame=frame, timestamp=timestamp)
        track_event = tracker.update(frame=frame, detection_event=det_event)
        objs = track_event["objects"]

        repo.update(
            timestamp_sec=timestamp,
            frame_index=frame_idx,
            tracked_objects=objs,
            zone_for_bbox=zone_for_bbox,
        )

        assert zones_px is not None
        zc = zone_counts_from_tracks(objs, zones_px)
        n = sum(1 for o in objs if o.get("id") is not None)
        event_inference.update(
            frame_shape=frame.shape,
            timestamp=timestamp,
            tracked_objects=objs,
            zone_counts=zc,
            total_tracks=n,
        )

    inference_result = event_inference.finalize()
    probability = float(inference_result["intrusion_probability"])
    unique_people = len(repo.all_people())
    unique_vehicles = len(repo.all_vehicles())

    timeline = build_timeline(
        people=repo.all_people(),
        vehicles=repo.all_vehicles(),
        frame_width=frame_width or 1920,
        frame_height=frame_height or 1080,
    )

    reasoning = build_intrusion_reasoning(
        timeline=timeline,
        unique_people=unique_people,
        unique_vehicles=unique_vehicles,
        intrusion_probability=probability,
    )
    if inference_result.get("used_learned_model"):
        reasoning = merge_reasoning_with_model_notes(
            reasoning,
            [
                "A trained sequence model (GRU over contextual features) contributed to the risk estimate.",
            ],
        )
    else:
        reasoning = merge_reasoning_with_model_notes(
            reasoning,
            [
                "No trained event weights were loaded; the estimate uses data-relative normalization over this clip.",
            ],
        )

    summary = generate_law_enforcement_summary(
        timeline=timeline,
        unique_people=unique_people,
        unique_vehicles=unique_vehicles,
        intrusion_probability=probability,
        reasoning_bullets=reasoning,
    )

    notifier.notify_intrusion_probability(
        video_path=video_path,
        probability=probability,
        threshold=float(inference_result.get("decision_threshold", 0.5)),
    )

    return {
        "video": video_path,
        "frames_analyzed": frames_analyzed,
        "last_timestamp_sec": last_timestamp,
        "intrusion_probability": probability,
        "intrusion_confidence_reasoning": reasoning,
        "summary": summary,
        "activity": {
            "unique_people": unique_people,
            "unique_vehicles": unique_vehicles,
        },
        "timeline": timeline,
        "tracks": _compact_tracks(repo),
        "model": {
            "used_learned_weights": bool(inference_result.get("used_learned_model", False)),
            "sequence_feature_length": int(inference_result.get("sequence_length", 0)),
            "decision_threshold": float(inference_result.get("decision_threshold", 0.5)),
        },
    }


def main(argv: Optional[List[str]] = None) -> None:
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

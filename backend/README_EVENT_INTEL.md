# Context-Aware Event Intelligence Pipeline

End-to-end flow for **video → detect → track → semantic zones → timeline → learned risk → law-enforcement narrative**.

## Stages

| Stage | Module | Role |
|-------|--------|------|
| Frame extraction | `backend/video/ingestion.py` | Sample frames at a fixed interval until EOF |
| Detection | `backend/detection/yolo_detector.py` | YOLOv8 people / vehicles |
| Tracking | `backend/tracking/deepsort_tracker.py` | Persistent IDs (DeepSORT or IoU fallback) |
| Semantic zones | `backend/analysis/semantic_zones.py` | `vehicle_zone`, `transition_zone`, `entry_zone` (normalized geometry) |
| Track store | `backend/analysis/track_repository.py` | Trajectories, entry/exit frame & time, velocity |
| Timeline | `backend/analysis/timeline_builder.py` | Observable events (arrivals, zone transitions, proximity) |
| Contextual features | `backend/analysis/contextual_features.py` | Zone occupancy + trajectory features |
| Sequence model | `backend/analysis/learning_event_inference.py` | GRU classifier or data-relative fallback |
| Reasoning | `backend/analysis/intrusion_reasoning.py` | Bullet explanations from facts + model |
| NL report | `backend/reporting/law_enforcement_report.py` | Dispatch-ready summary |

## Run

```bash
python -m backend.pipeline path/to/video.mp4 intrusion_event_model.pt
```

Train the event model (optional; otherwise fallback scoring applies):

```bash
python -m backend.analysis.train_event_model
```

## Output shape

- `activity.unique_people` / `unique_vehicles`
- `timeline` — structured events with `timestamp` (`MM:SS`) and `person_id` / `vehicle_id` where applicable
- `intrusion_probability` — learned or data-normalized
- `intrusion_confidence_reasoning` — bullet list
- `summary` — narrative for law enforcement / emergency review
- `tracks` — per-ID trajectories with zones and velocity

## Zones

Default rectangles are **normalized** to the frame; replace `default_semantic_zones()` with camera-specific calibration in production (still geometry, not rule thresholds).

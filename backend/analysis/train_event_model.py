from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from backend.analysis.contextual_features import append_zone_features, zone_counts_from_tracks
from backend.analysis.learning_event_inference import TrajectoryEventClassifier, TrajectoryFeatureExtractor
from backend.analysis.semantic_zones import default_semantic_zones, resolve_zones
from backend.detection.yolo_detector import YoloV8Detector
from backend.ml_intrusion.data_loader import list_split_sequences, load_sequence_frames
from backend.tracking.deepsort_tracker import DeepSortTracker


def _build_feature_sequence(frames: List) -> List[List[float]]:
    detector = YoloV8Detector()
    tracker = DeepSortTracker()
    extractor = TrajectoryFeatureExtractor()
    feats: List[List[float]] = []
    zones_px = None

    for i, frame in enumerate(frames):
        ts = float(i)
        h, w = frame.shape[:2]
        if zones_px is None:
            zones_px = resolve_zones(default_semantic_zones(), w, h)
        det_event = detector.detect_frame(frame=frame, timestamp=ts)
        track_event = tracker.update(frame=frame, detection_event=det_event)
        objs = track_event["objects"]
        base = extractor.update(
            frame_shape=frame.shape,
            timestamp=ts,
            tracked_objects=objs,
        )
        zc = zone_counts_from_tracks(objs, zones_px)
        n = sum(1 for o in objs if o.get("id") is not None)
        feats.append(append_zone_features(base, zc, n))

    return feats


def _to_batch(
    sequences: List[List[List[float]]],
    labels: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in sequences)
    feat_dim = len(sequences[0][0])
    x = torch.zeros((len(sequences), max_len, feat_dim), dtype=torch.float32, device=device)
    for i, seq in enumerate(sequences):
        x[i, : len(seq)] = torch.tensor(seq, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
    return x, y


def train_event_model(
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    sample_max_frames: Optional[int] = 80,
    model_out: str = "intrusion_event_model.pt",
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_items = list_split_sequences("train")

    feature_sequences: List[List[List[float]]] = []
    labels: List[int] = []
    for item in train_items:
        frames = load_sequence_frames(item.frame_paths, max_frames=sample_max_frames)
        if len(frames) < 2:
            continue
        seq = _build_feature_sequence(frames)
        if not seq:
            continue
        feature_sequences.append(seq)
        labels.append(int(item.label))

    if not feature_sequences:
        raise RuntimeError("No valid feature sequences were extracted for training.")

    feat_dim = len(feature_sequences[0][0])
    model = TrajectoryEventClassifier(input_dim=feat_dim, hidden_dim=64, num_layers=1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        x, y = _to_batch(feature_sequences, labels, device=device)
        logit = model(x)
        loss = loss_fn(logit, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"Epoch {epoch}: loss={float(loss.item()):.4f}")

    payload = {
        "state_dict": model.state_dict(),
        "meta": {
            "input_dim": feat_dim,
            "hidden_dim": 64,
            "num_layers": 1,
            "decision_threshold": 0.5,
        },
    }
    torch.save(payload, model_out)
    print(f"Saved event model: {model_out}")


if __name__ == "__main__":
    train_event_model()

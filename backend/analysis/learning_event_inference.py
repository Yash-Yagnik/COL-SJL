from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn

from backend.analysis.contextual_features import append_zone_features


@dataclass
class _TrackState:
    history: Deque[Tuple[float, float, float]]
    last_bbox: Tuple[float, float, float, float]
    first_seen: float
    last_seen: float


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


class TrajectoryFeatureExtractor:
    """
    Streaming trajectory / spatial / temporal feature extractor.

    Produces 12 base features; pipeline combines with 3 zone-occupancy
    features for a 15-dim contextual vector (see contextual_features).
    """

    def __init__(self, history_size: int = 64) -> None:
        self.history_size = history_size
        self.people_states: Dict[int, _TrackState] = {}
        self.vehicle_states: Dict[int, _TrackState] = {}
        self.unique_people_ids: set[int] = set()
        self.unique_vehicle_ids: set[int] = set()
        self.disappearance_points: List[Tuple[float, float]] = []
        self._prev_timestamp: Optional[float] = None

    def _update_states(
        self,
        *,
        current_ids: set[int],
        states: Dict[int, _TrackState],
        seen_ids: set[int],
        objects: List[Dict[str, Any]],
        timestamp: float,
    ) -> Tuple[int, int]:
        new_ids = current_ids - set(states.keys())
        seen_ids.update(current_ids)

        for obj in objects:
            tid = int(obj["id"])
            cx, cy = _bbox_center(obj["bbox"])
            bbox = tuple(float(v) for v in obj["bbox"])
            if tid not in states:
                states[tid] = _TrackState(
                    history=deque(maxlen=self.history_size),
                    last_bbox=bbox,
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
            st = states[tid]
            st.history.append((timestamp, cx, cy))
            st.last_bbox = bbox
            st.last_seen = timestamp

        disappeared_ids = set(states.keys()) - current_ids
        for tid in disappeared_ids:
            st = states[tid]
            if st.history:
                _, cx, cy = st.history[-1]
                self.disappearance_points.append((cx, cy))
            del states[tid]

        return (len(new_ids), len(disappeared_ids))

    def _infer_entry_point(
        self,
        frame_shape: Tuple[int, int, int],
        current_people: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        if self.disappearance_points:
            sx = sum(p[0] for p in self.disappearance_points)
            sy = sum(p[1] for p in self.disappearance_points)
            n = float(len(self.disappearance_points))
            return (sx / n, sy / n)

        if current_people:
            xs, ys = [], []
            for obj in current_people:
                cx, cy = _bbox_center(obj["bbox"])
                xs.append(cx)
                ys.append(cy)
            return (sum(xs) / len(xs), sum(ys) / len(ys))

        h, w = frame_shape[:2]
        return (w * 0.5, h * 0.5)

    def update(
        self,
        *,
        frame_shape: Tuple[int, int, int],
        timestamp: float,
        tracked_objects: List[Dict[str, Any]],
    ) -> List[float]:
        people = [o for o in tracked_objects if o.get("type") == "person" and o.get("id") is not None]
        vehicles = [o for o in tracked_objects if o.get("type") == "vehicle" and o.get("id") is not None]

        curr_people_ids = {int(o["id"]) for o in people}
        curr_vehicle_ids = {int(o["id"]) for o in vehicles}

        new_people, disappeared_people = self._update_states(
            current_ids=curr_people_ids,
            states=self.people_states,
            seen_ids=self.unique_people_ids,
            objects=people,
            timestamp=timestamp,
        )
        new_vehicles, disappeared_vehicles = self._update_states(
            current_ids=curr_vehicle_ids,
            states=self.vehicle_states,
            seen_ids=self.unique_vehicle_ids,
            objects=vehicles,
            timestamp=timestamp,
        )

        dt = 0.0 if self._prev_timestamp is None else max(timestamp - self._prev_timestamp, 1e-6)
        self._prev_timestamp = timestamp

        h, w = frame_shape[:2]
        diag = max((w * w + h * h) ** 0.5, 1e-6)
        entry_x, entry_y = self._infer_entry_point(frame_shape, people)

        speeds: List[float] = []
        dists_to_entry: List[float] = []
        for st in self.people_states.values():
            if len(st.history) >= 2:
                t0, x0, y0 = st.history[-2]
                t1, x1, y1 = st.history[-1]
                dt_local = max(t1 - t0, 1e-6)
                speeds.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt_local / diag)
            _, x, y = st.history[-1]
            dists_to_entry.append((((x - entry_x) ** 2 + (y - entry_y) ** 2) ** 0.5) / diag)

        prox_vals: List[float] = []
        for p in people:
            px, py = _bbox_center(p["bbox"])
            for v in vehicles:
                vx, vy = _bbox_center(v["bbox"])
                prox_vals.append((((px - vx) ** 2 + (py - vy) ** 2) ** 0.5) / diag)

        mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
        mean_entry_dist = sum(dists_to_entry) / len(dists_to_entry) if dists_to_entry else 1.0
        mean_person_vehicle_dist = sum(prox_vals) / len(prox_vals) if prox_vals else 1.0

        time_scale = dt if dt > 0 else 1.0
        return [
            float(len(curr_people_ids)),
            float(len(curr_vehicle_ids)),
            float(len(self.unique_people_ids)),
            float(len(self.unique_vehicle_ids)),
            float(new_people / time_scale),
            float(new_vehicles / time_scale),
            float(disappeared_people / time_scale),
            float(disappeared_vehicles / time_scale),
            float(mean_speed),
            float(mean_entry_dist),
            float(mean_person_vehicle_dist),
            float(math.log1p(len(self.disappearance_points))),
        ]


class TrajectoryEventClassifier(nn.Module):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 64, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.classifier(last)


def _fallback_probability(feature_sequence: List[List[float]]) -> float:
    """Data-relative score: min-max normalize over the clip, then sigmoid of mean(last)."""
    if not feature_sequence:
        return 0.0
    import numpy as np

    arr = np.array(feature_sequence, dtype=np.float64)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    last = (arr[-1] - mins) / span
    z = float(np.clip(last.mean(), 0.0, 1.0))
    return float(1.0 / (1.0 + math.exp(-(z * 4.0 - 2.0))))


class LearningBasedEventInference:
    """
    Learned sequence model over contextual trajectory features.
    Falls back to data-normalized scoring if weights are missing.
    """

    def __init__(
        self,
        *,
        model_path: str = "intrusion_event_model.pt",
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = TrajectoryFeatureExtractor()
        self._feature_sequence: List[List[float]] = []
        self.model: Optional[TrajectoryEventClassifier] = None
        self.input_dim = 15
        self.decision_threshold = 0.5
        self._used_learned_weights = False

        payload = None
        try:
            try:
                payload = torch.load(model_path, map_location=self.device, weights_only=False)
            except TypeError:
                payload = torch.load(model_path, map_location=self.device)
        except FileNotFoundError:
            payload = None
        except Exception:
            payload = None

        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            meta = payload.get("meta", {})
        elif payload is not None:
            state_dict = payload
            meta = {}
        else:
            state_dict = None
            meta = {}

        if state_dict is not None:
            self.input_dim = int(meta.get("input_dim", 15))
            hidden_dim = int(meta.get("hidden_dim", 64))
            num_layers = int(meta.get("num_layers", 1))
            self.decision_threshold = float(meta.get("decision_threshold", 0.5))
            self.model = TrajectoryEventClassifier(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
            ).to(self.device)
            try:
                self.model.load_state_dict(state_dict, strict=True)
                self.model.eval()
                self._used_learned_weights = True
            except Exception:
                self.model = None
                self._used_learned_weights = False

    def update(
        self,
        *,
        frame_shape: Tuple[int, int, int],
        timestamp: float,
        tracked_objects: List[Dict[str, Any]],
        zone_counts: Optional[Dict[str, int]] = None,
        total_tracks: int = 0,
    ) -> None:
        base = self.extractor.update(
            frame_shape=frame_shape,
            timestamp=timestamp,
            tracked_objects=tracked_objects,
        )
        zc = zone_counts or {"vehicle_zone": 0, "transition_zone": 0, "entry_zone": 0}
        if self.input_dim <= len(base):
            feat = base[: self.input_dim]
        else:
            feat = append_zone_features(base, zc, total_tracks)
            while len(feat) < self.input_dim:
                feat.append(0.0)
            feat = feat[: self.input_dim]
        self._feature_sequence.append(feat[: self.input_dim])

    def finalize(self) -> Dict[str, Any]:
        if not self._feature_sequence:
            prob = 0.0
        elif self.model is not None:
            x = torch.tensor(self._feature_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            if x.shape[2] != self.input_dim:
                x = x[:, :, : self.input_dim]
            with torch.no_grad():
                logit = self.model(x).squeeze()
                prob = float(torch.sigmoid(logit).item())
        else:
            prob = _fallback_probability(self._feature_sequence)

        return {
            "intrusion_probability": prob,
            "sequence_length": len(self._feature_sequence),
            "used_learned_model": self._used_learned_weights,
            "decision_threshold": self.decision_threshold,
            "activity": {
                "people_detections": len(self.extractor.unique_people_ids),
                "vehicle_detections": len(self.extractor.unique_vehicle_ids),
            },
        }

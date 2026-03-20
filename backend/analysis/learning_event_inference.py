from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import math
import torch
import torch.nn as nn


@dataclass
class _TrackState:
    history: Deque[Tuple[float, float, float]]  # (timestamp, cx, cy)
    last_bbox: Tuple[float, float, float, float]
    first_seen: float
    last_seen: float


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)


class TrajectoryFeatureExtractor:
    """
    Streaming trajectory/spatial/temporal feature extractor.

    Features are computed only from tracker outputs and frame geometry,
    so this remains memory-efficient for long videos.
    """

    def __init__(self, history_size: int = 64) -> None:
        self.history_size = history_size
        self.people_states: Dict[int, _TrackState] = {}
        self.vehicle_states: Dict[int, _TrackState] = {}
        self.unique_people_ids: set[int] = set()
        self.unique_vehicle_ids: set[int] = set()
        self.disappearance_points: List[Tuple[float, float]] = []
        self._prev_people: set[int] = set()
        self._prev_vehicles: set[int] = set()
        self._prev_timestamp: float | None = None

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
        self._prev_people = curr_people_ids
        self._prev_vehicles = curr_vehicle_ids

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
        feature_vec = [
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
        return feature_vec


class TrajectoryEventClassifier(nn.Module):
    def __init__(self, input_dim: int = 12, hidden_dim: int = 64, num_layers: int = 1) -> None:
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


class LearningBasedEventInference:
    """
    Model-only event inference from tracked object trajectories.
    """

    def __init__(
        self,
        *,
        model_path: str = "intrusion_event_model.pt",
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = TrajectoryFeatureExtractor()
        self._feature_sequence: List[List[float]] = []

        payload = torch.load(model_path, map_location=self.device)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            meta = payload.get("meta", {})
        else:
            state_dict = payload
            meta = {}

        input_dim = int(meta.get("input_dim", 12))
        hidden_dim = int(meta.get("hidden_dim", 64))
        num_layers = int(meta.get("num_layers", 1))
        self.decision_threshold = float(meta.get("decision_threshold", 0.5))

        self.model = TrajectoryEventClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def update(
        self,
        *,
        frame_shape: Tuple[int, int, int],
        timestamp: float,
        tracked_objects: List[Dict[str, Any]],
    ) -> None:
        feat = self.extractor.update(
            frame_shape=frame_shape,
            timestamp=timestamp,
            tracked_objects=tracked_objects,
        )
        self._feature_sequence.append(feat)

    def finalize(self) -> Dict[str, Any]:
        if not self._feature_sequence:
            prob = 0.0
        else:
            x = torch.tensor(self._feature_sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                logit = self.model(x).squeeze()
                prob = float(torch.sigmoid(logit).item())

        return {
            "intrusion_probability": prob,
            "is_intrusion": bool(prob >= self.decision_threshold),
            "threshold": self.decision_threshold,
            "sequence_length": len(self._feature_sequence),
            "activity": {
                "people_detections": len(self.extractor.unique_people_ids),
                "vehicle_detections": len(self.extractor.unique_vehicle_ids),
            },
        }


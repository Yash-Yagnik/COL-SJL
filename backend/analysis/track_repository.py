"""
Persistent track state: trajectories, entry/exit frames, deduplicated IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

TrackKind = Literal["person", "vehicle"]


@dataclass
class TrackRecord:
    """One tracked object across the video (sampled frames)."""

    track_id: int
    kind: TrackKind
    entry_timestamp_sec: float
    exit_timestamp_sec: float
    entry_frame_index: int
    exit_frame_index: int
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    # Each point: { "t": float, "frame_index": int, "cx": float, "cy": float,
    #              "bbox": [x1,y1,x2,y2], "zone": str | None }


class TrackRepository:
    """
    Accumulates per-track trajectories and lifecycle bounds.
    Works with tracker outputs that include person_id / vehicle_id as "id".
    """

    def __init__(self) -> None:
        self._people: Dict[int, TrackRecord] = {}
        self._vehicles: Dict[int, TrackRecord] = {}

    def update(
        self,
        *,
        timestamp_sec: float,
        frame_index: int,
        tracked_objects: List[Dict[str, Any]],
        zone_for_bbox,
    ) -> None:
        """
        zone_for_bbox: callable taking bbox list -> optional zone name str.
        """
        for obj in tracked_objects:
            oid = obj.get("id")
            if oid is None:
                continue
            tid = int(oid)
            otype = str(obj.get("type", ""))
            if otype == "person":
                kind: TrackKind = "person"
            elif otype == "vehicle":
                kind = "vehicle"
            else:
                continue

            bbox = [float(x) for x in obj["bbox"]]
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            zname = zone_for_bbox(bbox)

            dct = self._people if kind == "person" else self._vehicles
            if tid not in dct:
                dct[tid] = TrackRecord(
                    track_id=tid,
                    kind=kind,
                    entry_timestamp_sec=timestamp_sec,
                    exit_timestamp_sec=timestamp_sec,
                    entry_frame_index=frame_index,
                    exit_frame_index=frame_index,
                    trajectory=[],
                )
            rec = dct[tid]
            rec.exit_timestamp_sec = timestamp_sec
            rec.exit_frame_index = frame_index
            point: Dict[str, Any] = {
                "t": timestamp_sec,
                "frame_index": frame_index,
                "cx": cx,
                "cy": cy,
                "bbox": bbox,
                "zone": zname,
            }
            if rec.trajectory:
                prev = rec.trajectory[-1]
                dt = max(timestamp_sec - float(prev["t"]), 1e-6)
                vx = (cx - float(prev["cx"])) / dt
                vy = (cy - float(prev["cy"])) / dt
                point["velocity_px_s"] = (vx * vx + vy * vy) ** 0.5
                point["velocity_vector_px_s"] = [vx, vy]
            rec.trajectory.append(point)

    def all_people(self) -> Dict[int, TrackRecord]:
        return dict(self._people)

    def all_vehicles(self) -> Dict[int, TrackRecord]:
        return dict(self._vehicles)

    def summary(self) -> Dict[str, Any]:
        return {
            "unique_people": len(self._people),
            "unique_vehicles": len(self._vehicles),
            "people": {pid: self._serialize_track(r) for pid, r in self._people.items()},
            "vehicles": {vid: self._serialize_track(r) for vid, r in self._vehicles.items()},
        }

    @staticmethod
    def _serialize_track(r: TrackRecord) -> Dict[str, Any]:
        return {
            "person_id" if r.kind == "person" else "vehicle_id": r.track_id,
            "kind": r.kind,
            "entry_timestamp_sec": r.entry_timestamp_sec,
            "exit_timestamp_sec": r.exit_timestamp_sec,
            "entry_frame_index": r.entry_frame_index,
            "exit_frame_index": r.exit_frame_index,
            "trajectory": r.trajectory,
        }

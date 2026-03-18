from __future__ import annotations

from dataclasses import dataclass
from time import gmtime, strftime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from backend.analysis.zones import (
    Zone,
    bbox_center,
    default_zones,
    intersection_over_area_of_bbox,
    is_in_zone,
)


HistoryPoint = Tuple[int, float, float, float]  # (frame_idx, x_center, y_center, timestamp_sec)
BBox = Tuple[float, float, float, float]  # xyxy


def get_direction(history: List[HistoryPoint], lookback: int = 5) -> Tuple[float, float]:
    """
    Estimate movement direction as a vector between two points in history.

    Returns (dx, dy) in pixels over the lookback window.
    """
    if len(history) < 2:
        return (0.0, 0.0)

    i0 = max(0, len(history) - 1 - lookback)
    _, x0, y0, _ = history[i0]
    _, x1, y1, _ = history[-1]
    return (x1 - x0, y1 - y0)


def get_speed(history: List[HistoryPoint], lookback: int = 5) -> float:
    """
    Estimate speed in pixels/second using the lookback window.
    """
    if len(history) < 2:
        return 0.0

    i0 = max(0, len(history) - 1 - lookback)
    _, x0, y0, t0 = history[i0]
    _, x1, y1, t1 = history[-1]

    dt = max(t1 - t0, 1e-6)
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return dist / dt


def is_stationary(
    history: List[HistoryPoint],
    *,
    threshold_frames: int = 10,
    movement_threshold_px: float = 8.0,
) -> bool:
    """
    Returns True if an object has not moved meaningfully over the last
    `threshold_frames` history points.

    This helps detect loitering and "lingering at the door".
    """
    if len(history) < threshold_frames:
        return False

    recent = history[-threshold_frames:]
    _, x0, y0, _ = recent[0]
    _, x1, y1, _ = recent[-1]
    dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return dist < movement_threshold_px


def _format_timestamp(seconds_from_start: float) -> str:
    """
    Convert seconds-from-start into a string like "02:41 AM".
    """
    return strftime("%I:%M %p", gmtime(seconds_from_start))


@dataclass
class EventEngineConfig:
    """
    Tunable thresholds for the behavior engine.

    These values are conservative defaults for an MVP; you should tune them
    for your camera angle and sampling rate.
    """

    # Group arrival: N people within window seconds.
    group_arrival_min_people: int = 3
    group_arrival_window_sec: float = 8.0

    # Vehicle->person correlation window for "dropoff".
    vehicle_recent_window_sec: float = 12.0

    # Loitering detection near the door.
    loitering_min_sec_in_door: float = 10.0

    # Attempted entry heuristics.
    attempted_entry_min_sec_in_door: float = 12.0
    attempted_entry_overlap_threshold: float = 0.35
    attempted_entry_reentry_count: int = 3
    attempted_entry_reentry_window_sec: float = 15.0

    # Approach detection.
    approaching_min_speed_px_per_sec: float = 15.0
    approaching_alignment_threshold: float = 0.65  # cosine similarity

    # Risk scoring weights.
    score_group_arrival: int = 2
    score_vehicle_dropoff: int = 2
    score_approaching_entry: int = 3
    score_loitering: int = 2
    score_attempted_entry: int = 3

    # Final "high risk intrusion" threshold (sum of weights).
    high_risk_score_threshold: int = 7


class EventEngine:
    """
    Temporal behavior event engine.

    Consumes per-frame tracking outputs and produces:
      - inferred events (approach/loiter/attempted entry/group arrival/dropoff)
      - a per-frame risk score
      - a final structured intrusion decision

    This engine avoids frame-by-frame intrusion decisions and relies on
    temporal aggregation.
    """

    def __init__(
        self,
        *,
        zones: Optional[Dict[str, Zone]] = None,
        config: Optional[EventEngineConfig] = None,
    ) -> None:
        self.config = config or EventEngineConfig()

        # Zones are resolved to pixel coordinates once we know the frame size.
        self._zones_def = zones or default_zones()
        self._zones_px: Optional[Dict[str, BBox]] = None

        # Track history: track_id -> [(frame_idx, cx, cy, tsec), ...]
        self.track_history: Dict[int, List[HistoryPoint]] = {}

        # Zone history: track_id -> ["street", "driveway", "door", ...]
        self.zone_history: Dict[int, List[str]] = {}

        # First seen times.
        self.first_seen_time: Dict[int, float] = {}

        # Door zone entry time: track_id -> entry timestamp in seconds.
        self.zone_entry_time: Dict[int, Dict[str, float]] = {}

        # Presence history for "re-entry" detection (door boundary interactions).
        self._door_presence_events: Dict[int, List[Tuple[float, bool]]] = {}

        # Global vehicle observations.
        self.last_vehicle_seen_time: Optional[float] = None

        # Derived events emitted over time.
        self.events: List[Dict[str, Any]] = []

        # Track active IDs.
        self.active_ids: set[int] = set()

        # For group arrival detection: list of (track_id, first_seen_time) for persons.
        self._person_first_seen_log: List[Tuple[int, float]] = []

        # Intrusion latch (avoid spamming repeated high-risk intrusion events).
        self._high_risk_intrusion_emitted = False

    def _ensure_zones_resolved(self, frame: Any) -> None:
        if self._zones_px is not None:
            return

        h, w = frame.shape[:2]
        self._zones_px = {name: z.resolve(w, h) for name, z in self._zones_def.items()}

    def _current_zone(self, bbox_xyxy: BBox) -> Optional[str]:
        """
        Returns the name of the zone the bbox center lies in (first match),
        or None if it is outside configured zones.
        """
        if not self._zones_px:
            return None

        # Priority: door first, then driveway, then street.
        for name in ("door", "driveway", "street"):
            zone = self._zones_px.get(name)
            if zone and is_in_zone(bbox_xyxy, zone):
                return name

        # Any other zones.
        for name, zone in self._zones_px.items():
            if name in ("door", "driveway", "street"):
                continue
            if is_in_zone(bbox_xyxy, zone):
                return name

        return None

    def _record_zone_transition(self, track_id: int, zone: Optional[str]) -> None:
        if zone is None:
            return
        hist = self.zone_history.setdefault(track_id, [])
        if not hist or hist[-1] != zone:
            hist.append(zone)

    def _update_door_presence(self, track_id: int, timestamp_sec: float, in_door: bool) -> None:
        """
        Track door presence boundary crossings to detect repeated "in/out" behavior.
        """
        events = self._door_presence_events.setdefault(track_id, [])
        if not events or events[-1][1] != in_door:
            events.append((timestamp_sec, in_door))

        # Keep only recent window.
        window = self.config.attempted_entry_reentry_window_sec
        cutoff = timestamp_sec - window
        self._door_presence_events[track_id] = [(t, v) for (t, v) in events if t >= cutoff]

    def _count_door_reentries(self, track_id: int) -> int:
        """
        Count how many times the person entered the door zone in the recent window.
        """
        events = self._door_presence_events.get(track_id, [])
        # Count False->True transitions.
        reentries = 0
        for i in range(1, len(events)):
            if events[i - 1][1] is False and events[i][1] is True:
                reentries += 1
        return reentries

    def update(
        self,
        *,
        frame: Any,
        frame_idx: int,
        timestamp_sec: float,
        tracked_objects: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process one frame worth of tracking results and return a per-frame summary:

        {
          "events": [...],
          "risk_score": int,
          "active_ids": [...]
        }
        """
        self._ensure_zones_resolved(frame)
        assert self._zones_px is not None

        frame_events: List[Dict[str, Any]] = []
        risk_score = 0

        # Update vehicle observation time (global).
        vehicles_present_now = any(o.get("type") == "vehicle" for o in tracked_objects)
        if vehicles_present_now:
            self.last_vehicle_seen_time = timestamp_sec

        # Update per-object histories.
        current_person_ids: List[int] = []

        for obj in tracked_objects:
            if obj.get("type") != "person":
                continue

            tid = obj.get("id")
            if tid is None:
                continue
            track_id = int(tid)
            self.active_ids.add(track_id)
            current_person_ids.append(track_id)

            x1, y1, x2, y2 = obj["bbox"]
            bbox: BBox = (float(x1), float(y1), float(x2), float(y2))
            cx, cy = bbox_center(bbox)

            # Track movement history.
            self.track_history.setdefault(track_id, []).append((frame_idx, cx, cy, timestamp_sec))

            # First seen bookkeeping (for group arrival).
            if track_id not in self.first_seen_time:
                self.first_seen_time[track_id] = timestamp_sec
                self._person_first_seen_log.append((track_id, timestamp_sec))

            # Zone tracking.
            zone = self._current_zone(bbox)
            self._record_zone_transition(track_id, zone)

            # Zone entry time tracking for dwell/loitering.
            zet = self.zone_entry_time.setdefault(track_id, {})
            if zone is not None:
                if zone not in zet:
                    zet[zone] = timestamp_sec
            # If they are NOT in a zone now, we do not clear entry time; we keep the last entry
            # time per zone for simple MVP logic.

            # Door presence boundary crossings for attempted entry detection.
            in_door = zone == "door"
            self._update_door_presence(track_id, timestamp_sec, in_door)

        # 1) Group arrival detection.
        # If >= N unique people appear within the last window seconds, emit group_arrival once.
        window = self.config.group_arrival_window_sec
        cutoff = timestamp_sec - window
        recent_people = {tid for tid, t0 in self._person_first_seen_log if t0 >= cutoff}

        if len(recent_people) >= self.config.group_arrival_min_people:
            frame_events.append(
                {
                    "event_type": "group_arrival",
                    "suspect_count": len(self.first_seen_time),
                    "vehicles_detected": [],
                    "timestamp": _format_timestamp(timestamp_sec),
                    "actors": sorted(list(recent_people)),
                }
            )
            risk_score += self.config.score_group_arrival

        # 2) Vehicle -> person correlation ("dropoff").
        if self.last_vehicle_seen_time is not None:
            vehicle_recent = (timestamp_sec - self.last_vehicle_seen_time) <= self.config.vehicle_recent_window_sec
        else:
            vehicle_recent = False

        # "New people detected" in recent window: people first seen in last few seconds.
        new_people_cutoff = timestamp_sec - self.config.vehicle_recent_window_sec
        new_people = {tid for tid, t0 in self._person_first_seen_log if t0 >= new_people_cutoff}

        if vehicle_recent and len(new_people) > 0:
            frame_events.append(
                {
                    "event_type": "vehicle_dropoff",
                    "suspect_count": len(self.first_seen_time),
                    "vehicles_detected": ["vehicle"],
                    "timestamp": _format_timestamp(timestamp_sec),
                    "actors": sorted(list(new_people)),
                }
            )
            risk_score += self.config.score_vehicle_dropoff

        # 3) Approach, dwell/loitering, and attempted entry near door.
        door_zone = self._zones_px.get("door")
        door_center = bbox_center(door_zone) if door_zone else (0.0, 0.0)

        approaching_ids: List[int] = []
        loitering_ids: List[int] = []
        attempted_entry_ids: List[int] = []

        for track_id in current_person_ids:
            history = self.track_history.get(track_id, [])
            if len(history) < 2:
                continue

            # Determine if in door zone now (based on last known zone).
            zhist = self.zone_history.get(track_id, [])
            in_door = bool(zhist and zhist[-1] == "door")

            # Approach detection: moving toward door.
            dx, dy = get_direction(history, lookback=5)
            speed = get_speed(history, lookback=5)

            # Vector to door from current position.
            _, cx, cy, _ = history[-1]
            to_door = (door_center[0] - cx, door_center[1] - cy)

            # Compute cosine similarity between movement vector and door direction.
            mv_norm = (dx * dx + dy * dy) ** 0.5
            td_norm = (to_door[0] * to_door[0] + to_door[1] * to_door[1]) ** 0.5
            if mv_norm > 1e-6 and td_norm > 1e-6:
                cos_sim = (dx * to_door[0] + dy * to_door[1]) / (mv_norm * td_norm)
            else:
                cos_sim = 0.0

            approaching_entry = (
                speed >= self.config.approaching_min_speed_px_per_sec
                and cos_sim >= self.config.approaching_alignment_threshold
            )

            if approaching_entry:
                approaching_ids.append(track_id)

            # Dwell/loitering in the door zone.
            loitering = False
            attempted_entry = False

            if in_door:
                entry_t = self.zone_entry_time.get(track_id, {}).get("door")
                if entry_t is not None:
                    time_in_door = timestamp_sec - entry_t
                    if time_in_door >= self.config.loitering_min_sec_in_door and is_stationary(
                        history, threshold_frames=10
                    ):
                        loitering = True

                    # Attempted entry: long dwell AND strong overlap with door region
                    # OR repeated re-entry boundary crossings.
                    if door_zone is not None:
                        # Approximate overlap from the last known bbox by reconstructing from tracking event.
                        # We can read bbox from last tracked object by scanning current frame objects.
                        last_bbox = None
                        for o in tracked_objects:
                            if o.get("type") == "person" and o.get("id") == track_id:
                                x1, y1, x2, y2 = o["bbox"]
                                last_bbox = (float(x1), float(y1), float(x2), float(y2))
                                break

                        overlap = (
                            intersection_over_area_of_bbox(last_bbox, door_zone) if last_bbox is not None else 0.0
                        )
                    else:
                        overlap = 0.0

                    reentries = self._count_door_reentries(track_id)

                    if (
                        time_in_door >= self.config.attempted_entry_min_sec_in_door
                        and overlap >= self.config.attempted_entry_overlap_threshold
                    ) or (reentries >= self.config.attempted_entry_reentry_count):
                        attempted_entry = True

            if loitering:
                loitering_ids.append(track_id)
            if attempted_entry:
                attempted_entry_ids.append(track_id)

        if approaching_ids:
            frame_events.append(
                {
                    "event_type": "approaching_entry",
                    "suspect_count": len(self.first_seen_time),
                    "vehicles_detected": ["vehicle"] if vehicles_present_now else [],
                    "timestamp": _format_timestamp(timestamp_sec),
                    "actors": sorted(list(set(approaching_ids))),
                }
            )
            risk_score += self.config.score_approaching_entry

        if loitering_ids:
            frame_events.append(
                {
                    "event_type": "loitering",
                    "suspect_count": len(self.first_seen_time),
                    "vehicles_detected": ["vehicle"] if vehicles_present_now else [],
                    "timestamp": _format_timestamp(timestamp_sec),
                    "actors": sorted(list(set(loitering_ids))),
                }
            )
            risk_score += self.config.score_loitering

        if attempted_entry_ids:
            frame_events.append(
                {
                    "event_type": "attempted_entry",
                    "suspect_count": len(self.first_seen_time),
                    "vehicles_detected": ["vehicle"] if vehicles_present_now else [],
                    "timestamp": _format_timestamp(timestamp_sec),
                    "actors": sorted(list(set(attempted_entry_ids))),
                }
            )
            risk_score += self.config.score_attempted_entry

        # 4) Zone transition pattern: street -> driveway -> door.
        suspicious_progression_ids: List[int] = []
        for tid, zhist in self.zone_history.items():
            if len(zhist) < 3:
                continue
            # Look for a subsequence: street -> driveway -> door (in order).
            joined = ",".join(zhist[-6:])
            if "street" in joined and "driveway" in joined and "door" in joined:
                # Ensure order by scanning.
                try:
                    i_st = zhist.index("street")
                    i_dr = zhist.index("driveway", i_st + 1)
                    i_do = zhist.index("door", i_dr + 1)
                    if i_st < i_dr < i_do:
                        suspicious_progression_ids.append(tid)
                except ValueError:
                    pass

        if suspicious_progression_ids:
            frame_events.append(
                {
                    "event_type": "suspicious_progression",
                    "suspect_count": len(self.first_seen_time),
                    "vehicles_detected": ["vehicle"] if vehicles_present_now else [],
                    "timestamp": _format_timestamp(timestamp_sec),
                    "actors": sorted(list(set(suspicious_progression_ids))),
                }
            )
            # This event is informational; don't add risk weight by default.

        # Intrusion logic (multi-condition):
        # True intrusion requires approaching entry AND (loitering OR attempted_entry).
        approaching_entry_now = len(approaching_ids) > 0
        loitering_now = len(loitering_ids) > 0
        attempted_entry_now = len(attempted_entry_ids) > 0

        is_intrusion = approaching_entry_now and (loitering_now or attempted_entry_now)

        # "High risk intrusion" should be emitted when risk is strong.
        if (
            is_intrusion
            and not self._high_risk_intrusion_emitted
            and risk_score >= self.config.high_risk_score_threshold
        ):
            actors = sorted(list(set(approaching_ids + loitering_ids + attempted_entry_ids)))
            intrusion_event = {
                "event": "high_risk_intrusion",
                "confidence": 0.85,  # MVP fixed confidence; can be learned later.
                "actors": actors,
                "risk_score": risk_score,
                "timestamp": _format_timestamp(timestamp_sec),
            }
            frame_events.append(intrusion_event)
            self.events.append(intrusion_event)
            self._high_risk_intrusion_emitted = True

        # Persist frame events to global log for debugging/analysis.
        for ev in frame_events:
            self.events.append(ev)

        return {
            "events": frame_events,
            "risk_score": risk_score,
            "active_ids": sorted(list(self.active_ids)),
        }

    def finalize(self) -> Dict[str, Any]:
        """
        Return a final summary after processing the video.
        """
        last_ts = self.events[-1]["timestamp"] if self.events else None
        high_risk = [e for e in self.events if e.get("event") == "high_risk_intrusion"]

        return {
            "event_summary": {
                "high_risk_intrusion": bool(high_risk),
                "suspect_count": len(self.first_seen_time),
                "timestamp": last_ts,
            },
            "events": self.events,
        }


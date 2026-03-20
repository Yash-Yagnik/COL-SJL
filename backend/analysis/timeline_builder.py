"""
Builds a structured temporal event timeline from track records.

Events are derived from observable state changes (appearances, zone transitions,
proximity changes), not from fixed intrusion rule thresholds.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from backend.analysis.track_repository import TrackRecord


def format_timestamp_mmss(seconds: float) -> str:
    s = max(0.0, float(seconds))
    m = int(s // 60)
    sec = int(round(s - m * 60))
    if sec >= 60:
        m += 1
        sec = 0
    return f"{m:02d}:{sec:02d}"


def _frame_diag(w: float, h: float) -> float:
    return max(math.sqrt(w * w + h * h), 1e-6)


def _pairwise_min_dist_person_vehicle(
    traj_p: List[Dict[str, Any]],
    traj_v: List[Dict[str, Any]],
    frame_w: float,
    frame_h: float,
) -> List[Tuple[float, float]]:
    """Return list of (t_person, min_dist_norm) for each person sample."""
    if not traj_v:
        return []
    diag = _frame_diag(frame_w, frame_h)
    out: List[Tuple[float, float]] = []
    for pp in traj_p:
        t = float(pp["t"])
        px, py = float(pp["cx"]), float(pp["cy"])
        best = min(
            math.hypot(px - float(v["cx"]), py - float(v["cy"])) for v in traj_v
        )
        out.append((t, best / diag))
    return out


def build_timeline(
    *,
    people: Dict[int, TrackRecord],
    vehicles: Dict[int, TrackRecord],
    frame_width: int,
    frame_height: int,
) -> List[Dict[str, Any]]:
    """
    Construct ordered timeline events with timestamps as MM:SS strings.
    Uses person_id / vehicle_id keys for law-enforcement clarity.
    """
    events: List[Dict[str, Any]] = []
    fw, fh = float(frame_width), float(frame_height)

    # Vehicle arrivals (first observation)
    for vid, vr in sorted(vehicles.items(), key=lambda x: x[1].entry_timestamp_sec):
        events.append(
            {
                "timestamp": format_timestamp_mmss(vr.entry_timestamp_sec),
                "timestamp_sec": vr.entry_timestamp_sec,
                "event": "vehicle_arrival",
                "vehicle_id": vid,
            }
        )
        z0 = vr.trajectory[0].get("zone") if vr.trajectory else None
        if z0:
            events.append(
                {
                    "timestamp": format_timestamp_mmss(vr.entry_timestamp_sec),
                    "timestamp_sec": vr.entry_timestamp_sec,
                    "event": "vehicle_in_zone",
                    "vehicle_id": vid,
                    "zone": z0,
                }
            )

    # People: first seen + zone transitions
    for pid, pr in sorted(people.items(), key=lambda x: x[1].entry_timestamp_sec):
        events.append(
            {
                "timestamp": format_timestamp_mmss(pr.entry_timestamp_sec),
                "timestamp_sec": pr.entry_timestamp_sec,
                "event": "person_detected",
                "person_id": pid,
            }
        )
        prev_zone: Optional[str] = None
        for pt in pr.trajectory:
            z = pt.get("zone")
            if z != prev_zone and z is not None:
                events.append(
                    {
                        "timestamp": format_timestamp_mmss(float(pt["t"])),
                        "timestamp_sec": float(pt["t"]),
                        "event": "zone_entered",
                        "person_id": pid,
                        "zone": z,
                    }
                )
            if prev_zone is not None and z != prev_zone and prev_zone is not None:
                events.append(
                    {
                        "timestamp": format_timestamp_mmss(float(pt["t"])),
                        "timestamp_sec": float(pt["t"]),
                        "event": "zone_exited",
                        "person_id": pid,
                        "zone": prev_zone,
                    }
                )
            prev_zone = z

    # Proximity: person near vehicle then separation (soft "exit vehicle" narrative)
    for pid, pr in people.items():
        for vid, vr in vehicles.items():
            dists = _pairwise_min_dist_person_vehicle(pr.trajectory, vr.trajectory, fw, fh)
            if len(dists) < 3:
                continue
            # Near = normalized distance in lower part of this pair's observed range
            vals = [d[1] for d in dists]
            lo, hi = min(vals), max(vals)
            span = max(hi - lo, 1e-6)
            near_thr = lo + 0.25 * span
            far_thr = lo + 0.65 * span
            was_near = False
            for t_sec, dnorm in dists:
                if dnorm <= near_thr:
                    was_near = True
                elif was_near and dnorm >= far_thr:
                    events.append(
                        {
                            "timestamp": format_timestamp_mmss(t_sec),
                            "timestamp_sec": t_sec,
                            "event": "person_exit_vehicle",
                            "person_id": pid,
                            "vehicle_id": vid,
                        }
                    )
                    was_near = False

    # Approach entry: moving from transition toward entry (zone sequence)
    for pid, pr in people.items():
        zones_seq = [pt.get("zone") for pt in pr.trajectory if pt.get("zone")]
        for i in range(1, len(zones_seq)):
            if zones_seq[i - 1] == "transition_zone" and zones_seq[i] == "entry_zone":
                t = pr.trajectory[i]["t"]
                events.append(
                    {
                        "timestamp": format_timestamp_mmss(float(t)),
                        "timestamp_sec": float(t),
                        "event": "approach_entry_point",
                        "person_id": pid,
                    }
                )

    # Disappearance near entry: last zone was entry_zone
    for pid, pr in people.items():
        if not pr.trajectory:
            continue
        last = pr.trajectory[-1]
        last_z = last.get("zone")
        if last_z == "entry_zone":
            events.append(
                {
                    "timestamp": format_timestamp_mmss(float(pr.exit_timestamp_sec)),
                    "timestamp_sec": pr.exit_timestamp_sec,
                    "event": "person_disappears_near_entry",
                    "person_id": pid,
                }
            )

    events.sort(key=lambda e: (float(e.get("timestamp_sec", 0)), str(e.get("event", ""))))
    return events

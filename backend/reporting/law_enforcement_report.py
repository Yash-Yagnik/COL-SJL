"""
Natural-language incident narrative for law enforcement / emergency review.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _fmt_range(t0: str, t1: str) -> str:
    if t0 == t1:
        return f"approximately {t0}"
    return f"between {t0} and {t1}"


def generate_law_enforcement_summary(
    *,
    timeline: List[Dict[str, Any]],
    unique_people: int,
    unique_vehicles: int,
    intrusion_probability: float,
    reasoning_bullets: List[str],
) -> str:
    """
    Produce a concise narrative suitable for dispatch / investigator handoff.
    """
    paragraphs: List[str] = []

    # Opening
    if unique_vehicles >= 1:
        va = next((e for e in timeline if e.get("event") == "vehicle_arrival"), None)
        ts = va.get("timestamp", "00:00") if va else "the start of the clip"
        paragraphs.append(
            f"At approximately {ts}, a vehicle was first observed in the camera view."
        )
    else:
        paragraphs.append("No clearly tracked vehicle was present for the analyzed segment.")

    # People
    if unique_people >= 1:
        t_first = next(
            (e.get("timestamp") for e in timeline if e.get("event") == "person_detected"),
            None,
        )
        t_last = None
        for e in reversed(timeline):
            if e.get("event") in ("person_disappears_near_entry", "zone_entered"):
                t_last = e.get("timestamp")
                break
        if t_first:
            paragraphs.append(
                f"Individuals appeared on camera ({unique_people} unique track(s)). "
                f"Initial person activity was noted around {_fmt_range(t_first, t_last or t_first)}."
            )
    else:
        paragraphs.append("No persons were tracked reliably in this segment.")

    # Separation / entry
    if any(e.get("event") == "person_exit_vehicle" for e in timeline):
        paragraphs.append(
            "Tracked persons were observed separating from a vehicle after close proximity, "
            "consistent with exiting or moving away from the vehicle."
        )
    if any(e.get("event") == "approach_entry_point" for e in timeline):
        paragraphs.append(
            "Movement toward the entry area was observed for at least one tracked person."
        )
    if any(e.get("event") == "person_disappears_near_entry" for e in timeline):
        paragraphs.append(
            "At least one person was last seen near the entry area before no longer being visible "
            "in frame; return to the vehicle was not observed in the tracked sequence."
        )

    # Assessment
    if intrusion_probability >= 0.7:
        qual = "high"
    elif intrusion_probability >= 0.4:
        qual = "moderate"
    else:
        qual = "lower"

    paragraphs.append(
        f"Based on observed trajectories, zone activity, and the sequence model output "
        f"(estimated likelihood {intrusion_probability:.0%}), the behavioral assessment is "
        f"a {qual} concern relative to typical benign activity. "
        "This is an analytical aid; investigators should review the source video."
    )

    paragraphs.append(
        "Supporting analytical notes:\n- "
        + "\n- ".join(reasoning_bullets[:12])
    )

    return "\n\n".join(paragraphs)

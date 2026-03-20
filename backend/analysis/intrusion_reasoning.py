"""
Natural-language reasoning lines derived from timeline facts + model output.

No fixed intrusion thresholds: bullets describe observed behavior; probability
comes from the learned classifier (or data-normalized fallback).
"""

from __future__ import annotations

from typing import Any, Dict, List, Set


def build_intrusion_reasoning(
    *,
    timeline: List[Dict[str, Any]],
    unique_people: int,
    unique_vehicles: int,
    intrusion_probability: float,
) -> List[str]:
    """Produce law-enforcement style reasoning bullets (English)."""
    lines: List[str] = []
    event_types: Set[str] = {str(e.get("event", "")) for e in timeline}

    if unique_vehicles >= 1:
        lines.append(
            "At least one vehicle was observed in the scene during the recording period."
        )
    else:
        lines.append("No vehicle was clearly tracked for the full sequence.")

    if unique_people >= 1:
        lines.append(
            f"{unique_people} distinct individual(s) were tracked (deduplicated across frames)."
        )
    else:
        lines.append("No individuals were tracked as persons in this clip.")

    if "vehicle_arrival" in event_types or "vehicle_in_zone" in event_types:
        lines.append("Vehicle-related activity was observed in the timeline.")

    if "person_exit_vehicle" in event_types:
        lines.append(
            "One or more tracked persons moved away from a vehicle after being in close proximity."
        )

    if "approach_entry_point" in event_types:
        lines.append("Tracked persons showed movement toward the designated entry area.")

    if "person_disappears_near_entry" in event_types:
        lines.append(
            "At least one person was last observed near the entry zone before leaving the frame."
        )

    if "zone_entered" in event_types:
        lines.append("Zone transitions were recorded, indicating movement across the property layout.")

    lines.append(
        f"The behavioral risk estimate from the sequence model is {intrusion_probability:.0%} "
        "(this is a statistical estimate, not a legal determination)."
    )
    return lines


def merge_reasoning_with_model_notes(
    base: List[str],
    extra: List[str],
) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in base + extra:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

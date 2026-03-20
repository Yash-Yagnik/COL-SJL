"""Per-frame contextual features (zone occupancy) for sequence models."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.analysis.semantic_zones import primary_semantic_zone


def zone_counts_from_tracks(
    tracked_objects: List[Dict[str, Any]],
    zones_px: Dict[str, Any],
) -> Dict[str, int]:
    counts = {"vehicle_zone": 0, "transition_zone": 0, "entry_zone": 0}
    for obj in tracked_objects:
        if obj.get("id") is None:
            continue
        bbox = obj.get("bbox")
        if not bbox:
            continue
        z = primary_semantic_zone(tuple(float(x) for x in bbox), zones_px)
        if z in counts:
            counts[z] += 1
    return counts


def append_zone_features(
    base_twelve: List[float],
    zone_counts: Dict[str, int],
    total_tracks: int,
) -> List[float]:
    n = max(total_tracks, 1)
    vz = zone_counts.get("vehicle_zone", 0) / n
    tz = zone_counts.get("transition_zone", 0) / n
    ez = zone_counts.get("entry_zone", 0) / n
    return base_twelve + [vz, tz, ez]

"""
Semantic scene zones for context-aware event intelligence.

Zones are geometric regions (normalized 0–1 or pixels). They define *where*
things happen, not intrusion rules. Membership uses soft scores for downstream
learned / probabilistic models.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from backend.analysis.zones import BBox, Zone, bbox_center

SemanticZoneName = Literal["vehicle_zone", "entry_zone", "transition_zone"]


def default_semantic_zones() -> Dict[str, Zone]:
    """
    Default layout for a typical front-property camera:
    - vehicle_zone: street / approach where vehicles appear
    - transition_zone: driveway / path between street and door
    - entry_zone: door / porch (high interest for entry behavior)
    """
    return {
        "vehicle_zone": Zone("vehicle_zone", 0.0, 0.0, 1.0, 0.42, normalized=True),
        "transition_zone": Zone("transition_zone", 0.15, 0.35, 0.85, 0.92, normalized=True),
        "entry_zone": Zone("entry_zone", 0.38, 0.42, 0.62, 0.98, normalized=True),
    }


def resolve_zones(
    zones_def: Dict[str, Zone],
    frame_width: int,
    frame_height: int,
) -> Dict[str, BBox]:
    return {name: z.resolve(frame_width, frame_height) for name, z in zones_def.items()}


def zone_name_at_point(
    px: float,
    py: float,
    zones_px: Dict[str, BBox],
    priority: Tuple[str, ...] = ("entry_zone", "transition_zone", "vehicle_zone"),
) -> Optional[str]:
    """Return the highest-priority zone whose rectangle contains the point."""
    for name in priority:
        z = zones_px.get(name)
        if z is None:
            continue
        x1, y1, x2, y2 = z
        if x1 <= px <= x2 and y1 <= py <= y2:
            return name
    return None


def soft_zone_scores(
    px: float,
    py: float,
    zones_px: Dict[str, BBox],
) -> Dict[str, float]:
    """
    Soft membership in [0, 1] using distance to rectangle interior.
    Useful for probabilistic models without hard thresholds.
    """
    scores: Dict[str, float] = {}
    for name, z in zones_px.items():
        x1, y1, x2, y2 = z
        cx = min(max(px, x1), x2)
        cy = min(max(py, y1), y2)
        dist = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        scale = (w * w + h * h) ** 0.5
        scores[name] = float(max(0.0, 1.0 - dist / max(scale, 1e-6)))
    return scores


def primary_semantic_zone(
    bbox_xyxy: BBox,
    zones_px: Dict[str, BBox],
) -> Optional[str]:
    """Highest-priority zone whose center lies inside the rectangle."""
    cx, cy = bbox_center(bbox_xyxy)
    return zone_name_at_point(cx, cy, zones_px)

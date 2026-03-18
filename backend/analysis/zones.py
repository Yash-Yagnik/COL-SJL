"""
Zone configuration and helpers.

Zones are used by the behavior event engine to reason about movement
patterns (street -> driveway -> door), dwell time near entry points, etc.

This MVP supports zones defined in:
  - absolute pixel coordinates, or
  - normalized coordinates (0..1) relative to frame width/height
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


BBox = Tuple[float, float, float, float]  # xyxy


@dataclass(frozen=True)
class Zone:
    """
    Axis-aligned rectangular zone.

    If `normalized=True`, coordinates are fractions of the frame size:
      x in [0, 1], y in [0, 1]
    """

    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    normalized: bool = True

    def resolve(self, frame_width: int, frame_height: int) -> BBox:
        """
        Return this zone as absolute pixel coordinates in xyxy format.
        """
        if not self.normalized:
            return (self.x1, self.y1, self.x2, self.y2)

        return (
            self.x1 * frame_width,
            self.y1 * frame_height,
            self.x2 * frame_width,
            self.y2 * frame_height,
        )


def default_zones() -> Dict[str, Zone]:
    """
    Default zones in normalized coordinates.

    IMPORTANT: You should tune these values for your camera view.
    These defaults are just a starting point.
    """
    return {
        # Bottom-center area typical for a front door region in many camera views.
        "door": Zone("door", 0.40, 0.45, 0.60, 0.95, normalized=True),
        # Area closer to the property where cars may stop.
        "driveway": Zone("driveway", 0.20, 0.40, 0.80, 0.98, normalized=True),
        # Area outside property bounds.
        "street": Zone("street", 0.0, 0.0, 1.0, 0.45, normalized=True),
    }


def bbox_center(bbox_xyxy: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def is_in_zone(bbox_xyxy: BBox, zone_xyxy: BBox) -> bool:
    """
    Returns True if the bbox center point lies within the zone.
    """
    cx, cy = bbox_center(bbox_xyxy)
    zx1, zy1, zx2, zy2 = zone_xyxy
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def intersection_over_area_of_bbox(bbox_xyxy: BBox, zone_xyxy: BBox) -> float:
    """
    How much of the bbox area overlaps the zone (0..1).

    This is useful for detecting strong interaction with the door zone.
    """
    bx1, by1, bx2, by2 = bbox_xyxy
    zx1, zy1, zx2, zy2 = zone_xyxy

    ix1 = max(bx1, zx1)
    iy1 = max(by1, zy1)
    ix2 = min(bx2, zx2)
    iy2 = min(by2, zy2)

    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_bbox = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    return inter / area_bbox if area_bbox > 0 else 0.0


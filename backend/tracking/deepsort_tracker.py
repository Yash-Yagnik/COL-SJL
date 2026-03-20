from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    # Preferred: real DeepSORT implementation.
    from deep_sort_realtime.deepsort_tracker import DeepSort

    _HAS_DEEPSORT = True
except Exception:
    # If the dependency is not installed, we fall back to a lightweight IoU tracker.
    DeepSort = None  # type: ignore
    _HAS_DEEPSORT = False


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """
    Convert a bounding box from [x1, y1, x2, y2] to [x, y, w, h].
    """
    return (x1, y1, x2 - x1, y2 - y1)


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """
    Intersection-over-Union for boxes in xyxy format.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class _IoUTrack:
    """
    Very small fallback tracker that keeps IDs stable using IoU matching.

    This is NOT DeepSORT, but it prevents the "new ID every frame" problem
    when DeepSORT isn't installed.
    """

    def __init__(self, track_id: int, bbox_xyxy: Tuple[float, float, float, float]) -> None:
        self.track_id = track_id
        self.bbox_xyxy = bbox_xyxy
        self.missed = 0


class DeepSortTracker:
    """
    Wraps a DeepSORT-style tracker to:

    - Take YOLO detections per frame.
    - Maintain identities (track IDs) across frames.
    - Estimate the total number of unique suspects (persons).
    """

    def __init__(
        self,
        *,
        max_age: int = 50,
        n_init: int = 2,
        max_iou_distance: float = 0.7,
        iou_match_threshold: float = 0.3,
    ) -> None:
        """
        Parameters
        ----------
        max_age:
            How many *sampled frames* a track is kept "alive" without matching
            a detection. Because you sample every 0.5s, ``max_age=50`` keeps
            tracks for ~25s of occlusion (camera covered / temporary loss).
        n_init:
            Number of hits needed before a DeepSORT track is considered confirmed.
        max_iou_distance:
            DeepSORT gating parameter (depends on implementation).
        iou_match_threshold:
            Fallback IoU tracker threshold when DeepSORT is unavailable.
        """
        self._seen_person_ids: set[int] = set()
        self._seen_vehicle_ids: set[int] = set()
        self._iou_match_threshold = iou_match_threshold

        if _HAS_DEEPSORT:
            # DeepSORT will use appearance + motion to maintain IDs through occlusion.
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                max_iou_distance=max_iou_distance,
            )
        else:
            # Fallback: simple IoU matcher (keeps IDs stable for smooth motion).
            self.tracker = None
            self._next_id = 1
            self._tracks_by_type: Dict[str, List[_IoUTrack]] = {"person": [], "vehicle": []}

    def _update_iou_tracks(
        self,
        *,
        detections: List[Tuple[Tuple[float, float, float, float], float, str]],
        object_type: str,
        tracked_objects: List[Dict[str, Any]],
    ) -> None:
        tracks = self._tracks_by_type.setdefault(object_type, [])

        for tr in tracks:
            tr.missed += 1

        for bbox, conf, obj_type in detections:
            best_track = None
            best_iou = 0.0
            for tr in tracks:
                iou_val = _iou(tr.bbox_xyxy, bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_track = tr

            if best_track is not None and best_iou >= self._iou_match_threshold:
                best_track.bbox_xyxy = bbox
                best_track.missed = 0
                track_id = best_track.track_id
            else:
                track_id = self._next_id
                self._next_id += 1
                tracks.append(_IoUTrack(track_id=track_id, bbox_xyxy=bbox))

            if obj_type == "person":
                self._seen_person_ids.add(track_id)
            elif obj_type == "vehicle":
                self._seen_vehicle_ids.add(track_id)

            tracked_objects.append(
                {"id": track_id, "type": obj_type, "confidence": conf, "bbox": list(bbox)}
            )

        self._tracks_by_type[object_type] = [tr for tr in tracks if tr.missed <= 50]

    def update(
        self,
        frame: np.ndarray,
        detection_event: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Updates tracks using detections from a single frame and returns
        a structured tracking event dictionary.

        Parameters
        ----------
        frame:
            Current video frame as a NumPy array.
        detection_event:
            Output from ``YoloV8Detector.detect_frame``.

        Returns
        -------
        dict
            Example structure:

            {
              "timestamp": 1.23,
              "objects": [
                {"id": 5, "type": "person", "confidence": 0.93, "bbox": [...]},
                {"id": 2, "type": "vehicle", "confidence": 0.87, "bbox": [...]}
              ],
              "total_suspects": 3
            }
        """
        timestamp = float(detection_event["timestamp"])
        objects = detection_event["objects"]

        tracked_objects: List[Dict[str, Any]] = []

        if _HAS_DEEPSORT and self.tracker is not None:
            # Build detections in the format expected by deep_sort_realtime:
            # [ ([x1,y1,x2,y2], confidence, class_name), ... ]
            # This allows DeepSORT to preserve IDs across frames and short occlusions.
            dets_for_tracker: List[Tuple[List[float], float, str]] = []
            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                conf = float(obj["confidence"])
                cls_name = str(obj["type"])  # "person" or "vehicle"
                dets_for_tracker.append(([float(x1), float(y1), float(x2), float(y2)], conf, cls_name))

            tracks = self.tracker.update_tracks(dets_for_tracker, frame=frame)

            for t in tracks:
                # Ignore tentative/unconfirmed tracks to avoid counting noise as suspects.
                if hasattr(t, "is_confirmed") and not t.is_confirmed():
                    continue

                ltrb = t.to_ltrb()
                x1, y1, x2, y2 = [float(v) for v in ltrb]

                track_id = int(getattr(t, "track_id"))

                # deep_sort_realtime provides class name if we pass it in detections.
                det_class = None
                if hasattr(t, "get_det_class"):
                    det_class = t.get_det_class()
                elif hasattr(t, "det_class"):
                    det_class = getattr(t, "det_class")

                obj_type = str(det_class) if det_class else "unknown"

                # Confidence may not be available for every track; keep best-effort.
                conf = None
                if hasattr(t, "get_det_conf"):
                    conf = t.get_det_conf()

                if obj_type == "person":
                    self._seen_person_ids.add(track_id)
                elif obj_type == "vehicle":
                    self._seen_vehicle_ids.add(track_id)

                tracked_objects.append(
                    {
                        "id": track_id,
                        "type": obj_type,
                        "confidence": float(conf) if conf is not None else None,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        else:
            # Fallback: stable IDs via IoU matching (better than "new id every frame").
            person_dets: List[Tuple[Tuple[float, float, float, float], float, str]] = []
            vehicle_dets: List[Tuple[Tuple[float, float, float, float], float, str]] = []

            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                bbox = (float(x1), float(y1), float(x2), float(y2))
                conf = float(obj["confidence"])
                obj_type = str(obj["type"])
                if obj_type == "person":
                    person_dets.append((bbox, conf, obj_type))
                else:
                    vehicle_dets.append((bbox, conf, obj_type))

            self._update_iou_tracks(
                detections=person_dets,
                object_type="person",
                tracked_objects=tracked_objects,
            )
            self._update_iou_tracks(
                detections=vehicle_dets,
                object_type="vehicle",
                tracked_objects=tracked_objects,
            )

        return {
            "timestamp": timestamp,
            "objects": tracked_objects,
            "total_unique_people": len(self._seen_person_ids),
            "total_unique_vehicles": len(self._seen_vehicle_ids),
            "total_suspects": len(self._seen_person_ids),
        }


from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO

import torch

_orig_load = torch.load

def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)

torch.load = _torch_load

# YOLO class names that correspond to people and vehicles (COCO-style).
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "bicycle", "motorcycle"}


class YoloV8Detector:
    """
    Thin wrapper around a YOLOv8 model (via ``ultralytics``) that:

    - Runs inference on a single frame.
    - Filters detections to people and vehicles.
    - Returns a structured event-like dictionary per frame.
    """

    def __init__(self, model_path: str = "yolov8n.pt", device: str = "cpu") -> None:
        """
        Parameters
        ----------
        model_path:
            Path to the YOLOv8 weights file (e.g. ``yolov8n.pt``).
        device:
            Inference device identifier (e.g. ``"cpu"`` or ``"cuda"``).
        """
        self.model = YOLO(model_path)
        self.device = device

    def detect_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """
        Runs detection on one frame and returns a structured dictionary.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array.
        timestamp:
            Time in seconds from the start of the video for this frame.

        Returns
        -------
        dict
            Example structure:

            {
              "timestamp": 1.23,
              "objects": [
                  {"type": "person", "confidence": 0.92, "bbox": [x1, y1, x2, y2]},
                  {"type": "vehicle", "confidence": 0.87, "bbox": [x1, y1, x2, y2]}
              ]
            }
        """
        # Run YOLO inference. ``verbose=False`` to keep logs clean.
        results = self.model(frame, device=self.device, verbose=False)

        objects: List[Dict[str, Any]] = []

        # YOLO may return multiple result objects; iterate through boxes.
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]

                # Filter: only keep people and vehicles.
                if label in PERSON_CLASSES:
                    obj_type = "person"
                elif label in VEHICLE_CLASSES:
                    obj_type = "vehicle"
                else:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                objects.append(
                    {
                        "type": obj_type,
                        "label": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return {
            "timestamp": timestamp,
            "objects": objects,
        }


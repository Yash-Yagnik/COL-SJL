from typing import List, Optional

import torch
import cv2

from backend.ml_intrusion.model import CnnLstmIntrusionModel
from backend.video.ingestion import VideoIngestion


DEFAULT_THRESHOLD = 0.8  # configurable default per requirements


def _resolve_model_path(model_path: str) -> str:
    """
    Be tolerant to model filename differences.

    If the requested path doesn't exist, try common alternates in the
    current working directory.
    """
    import os

    if os.path.exists(model_path):
        return model_path

    alternates = ["intrusion_model.pt", "intrusion_detection.pt"]
    for alt in alternates:
        if os.path.exists(alt):
            return alt

    raise FileNotFoundError(
        f"Model file not found: {model_path}. Also tried {alternates} in the current directory."
    )


def predict_frames(
    frames: List,
    *,
    model_path: str = "intrusion_model.pt",
    threshold: float = DEFAULT_THRESHOLD,
    backbone_name: str = "resnet18",
    print_alert: bool = True,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> float:
    """
    Run model inference from an already-extracted frame sequence.

    Model-driven only: the ONLY alert decision is (probability > threshold).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = _resolve_model_path(model_path)

    model = CnnLstmIntrusionModel(backbone_name=backbone_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if verbose:
        print(f"Frames extracted: {len(frames)}")

    if len(frames) < 2:
        if verbose:
            print("Intrusion probability: 0.00 (insufficient frames)")
        return 0.0

    with torch.no_grad():
        feats = model.cnn.frames_to_features(frames, device=device).unsqueeze(0)  # [1,T,F]
        logit = model(feats).squeeze()
        prob = torch.sigmoid(logit).item()

    if verbose:
        print(f"Intrusion probability: {prob:.2f}")

    if print_alert and prob > threshold:
        print("=== SECURITY ALERT ===")
        print(f"INTRUSION DETECTED (p={prob:.2f}, threshold={threshold:.2f})")
        print("======================")

    return float(prob)


def predict_sequence(
    sequence_id: str,
    frame_paths: list[str],
    *,
    model_path: str = "intrusion_model.pt",
    threshold: float = DEFAULT_THRESHOLD,
    max_frames: Optional[int] = None,
    backbone_name: str = "resnet18",
    print_alert: bool = True,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> float:
    """
    Run model-driven intrusion inference for a single video.

    Prints logs:
    - Processing video: X
    - Frames extracted: N
    - Intrusion probability: 0.82
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep this helper for image-sequence testing.
    from backend.ml_intrusion.data_loader import load_sequence_frames

    print(f"Processing video/sequence: {sequence_id}")
    frames = load_sequence_frames(frame_paths, max_frames=max_frames)
    return predict_frames(
        frames,
        model_path=model_path,
        threshold=threshold,
        backbone_name=backbone_name,
        print_alert=print_alert,
        verbose=verbose,
        device=device,
    )


def predict_video(
    video_path: str,
    *,
    model_path: str = "intrusion_model.pt",
    threshold: float = DEFAULT_THRESHOLD,
    sample_interval: float = 0.5,
    backbone_name: str = "resnet18",
    print_alert: bool = True,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> float:
    """
    Run intrusion inference on a real video file (.mp4, .avi, ...).

    Sampling:
    - uses OpenCV
    - samples frames every `sample_interval` seconds
    - processes frames until the video ends
    """
    # For compatibility with your existing frame sampling logic, we reuse VideoIngestion.
    ingestion = VideoIngestion(video_path=video_path, sample_interval=sample_interval)
    frames: List = []
    for frame_data in ingestion.iter_frames():
        frames.append(frame_data.image)

    print(f"Processing video: {video_path}")
    return predict_frames(
        frames,
        model_path=model_path,
        threshold=threshold,
        backbone_name=backbone_name,
        print_alert=print_alert,
        verbose=verbose,
        device=device,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "  python -m backend.ml_intrusion.inference <video_path.mp4> [threshold]\n"
            "  (or) pass a frames directory to use predict_sequence separately."
        )
        raise SystemExit(1)

    path = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_THRESHOLD

    # If the path is a directory, treat it as an image-sequence dir.
    import os
    if os.path.isdir(path):
        frame_paths = [
            os.path.join(path, f)
            for f in sorted(os.listdir(path))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        predict_sequence(os.path.basename(path), frame_paths, threshold=thr)
    else:
        predict_video(path, threshold=thr)


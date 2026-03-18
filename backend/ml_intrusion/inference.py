from typing import Optional

import torch

from backend.ml_intrusion.data_loader import load_sequence_frames
from backend.ml_intrusion.model import CnnLstmIntrusionModel


DEFAULT_THRESHOLD = 0.8  # configurable default per requirements


def predict_sequence(
    sequence_id: str,
    frame_paths: list[str],
    *,
    model_path: str = "intrusion_model.pt",
    threshold: float = DEFAULT_THRESHOLD,
    max_frames: int = 20,
    backbone_name: str = "resnet18",
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

    model = CnnLstmIntrusionModel(backbone_name=backbone_name).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Processing video: {sequence_id}")
    frames = load_sequence_frames(frame_paths, max_frames=max_frames)
    print(f"Frames extracted: {len(frames)}")

    if len(frames) < 2:
        print("Intrusion probability: 0.00 (insufficient frames)")
        return 0.0

    with torch.no_grad():
        feats = model.cnn.frames_to_features(frames, device=device).unsqueeze(0)  # [1,T,F]
        logit = model(feats).squeeze()
        prob = torch.sigmoid(logit).item()

    print(f"Intrusion probability: {prob:.2f}")

    # Model-driven only: threshold is configurable and is the ONLY alert trigger.
    if prob > threshold:
        print("=== SECURITY ALERT ===")
        print(f"INTRUSION DETECTED (p={prob:.2f})")
        print("======================")

    return float(prob)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m backend.ml_intrusion.inference <sequence_frames_dir> [threshold]")
        raise SystemExit(1)

    seq_dir = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) >= 3 else DEFAULT_THRESHOLD
    # Load all images in the directory as a sequence.
    import os

    frame_paths = [
        os.path.join(seq_dir, f)
        for f in sorted(os.listdir(seq_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    predict_sequence(os.path.basename(seq_dir), frame_paths, threshold=thr)


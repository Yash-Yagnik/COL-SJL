import random

import torch

from backend.ml_intrusion.data_loader import list_split_sequences, load_sequence_frames
from backend.ml_intrusion.model import CnnLstmIntrusionModel


def train(
    *,
    epochs: int = 5,
    lr: float = 1e-4,
    max_frames: int = 20,
    backbone_name: str = "resnet18",
) -> None:
    """
    Minimal working training loop:
    - Loads bounded sampled frames per video
    - Extracts CNN features (frozen CNN)
    - Trains LSTM + classifier with BCEWithLogitsLoss
    - Prints training loss per epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_items = list_split_sequences("train")
    test_items = list_split_sequences("test")

    model = CnnLstmIntrusionModel(backbone_name=backbone_name).to(device)
    model.train()

    # Only train parameters that require gradients (CNN is frozen).
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n = 0

        random.shuffle(train_items)
        for item in train_items:
            print(f"Processing video: {item.sequence_id}")
            frames = load_sequence_frames(item.frame_paths, max_frames=max_frames)
            print(f"Frames extracted: {len(frames)}")

            # Skip empty videos safely.
            if len(frames) < 2:
                continue

            with torch.no_grad():
                feats = model.cnn.frames_to_features(frames, device=device)  # [T,F]

            feats = feats.unsqueeze(0)  # [1,T,F]
            y = torch.tensor([[float(item.label)]], device=device)  # [1,1]

            logit = model(feats)  # [1,1]
            loss = loss_fn(logit, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            n += 1

        print(f"Epoch {epoch} - Training loss: {total_loss / max(n, 1):.4f}")

    torch.save(model.state_dict(), "intrusion_model.pt")
    print("Saved model: intrusion_model.pt")

    # Simple evaluation pass.
    if test_items:
        from backend.ml_intrusion.inference import predict_sequence

        print("=== Evaluation ===")
        for item in test_items:
            prob = predict_sequence(
                item.sequence_id,
                item.frame_paths,
                model_path="intrusion_model.pt",
                threshold=0.8,
                max_frames=max_frames,
                backbone_name=backbone_name,
                device=device,
            )
            print(f"Label: {item.label} (0=Normal,1=Burglary) | Prob: {prob:.2f}")


if __name__ == "__main__":
    train()


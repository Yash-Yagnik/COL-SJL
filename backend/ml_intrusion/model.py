from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class CnnBackbone(nn.Module):
    """
    Wrapper for a pretrained CNN backbone that outputs a feature vector per frame.

    Per requirements:
    - Uses a pretrained CNN
    - CNN is frozen (not trained from scratch)
    """

    def __init__(self, backbone_name: str = "resnet18") -> None:
        super().__init__()

        # Import torchvision lazily to keep module import errors obvious.
        import torchvision.models as models

        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
            feature_dim = 2048
        else:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            feature_dim = 512

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.feature_dim = feature_dim

        # Freeze backbone parameters.
        for p in self.backbone.parameters():
            p.requires_grad = False

        import torchvision.transforms as T

        self.preprocess = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def frames_to_features(self, frames_bgr: List, device: torch.device) -> torch.Tensor:
        """
        Convert a list of OpenCV BGR frames to a tensor of CNN features [T, F].
        """
        imgs = []
        for f in frames_bgr:
            # OpenCV gives BGR; convert to RGB.
            rgb = f[:, :, ::-1].copy()
            imgs.append(self.preprocess(rgb))

        if not imgs:
            # Return an empty feature tensor for safety.
            return torch.empty((0, self.feature_dim), device=device)

        x = torch.stack(imgs, dim=0).to(device)  # [T,3,224,224]
        feats = self.backbone(x)  # [T,F]
        return feats


class CnnLstmIntrusionModel(nn.Module):
    """
    CNN feature extractor + LSTM sequence model -> binary logit output.

    Output is a logit for BCEWithLogitsLoss; apply sigmoid for probability.
    """

    def __init__(
        self,
        *,
        backbone_name: str = "resnet18",
        lstm_hidden_dim: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.cnn = CnnBackbone(backbone_name=backbone_name)
        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, seq_features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        seq_features:
            Tensor shape [B, T, F]

        Returns
        -------
        Tensor
            Logits shape [B, 1]
        """
        out, _ = self.lstm(seq_features)
        last = out[:, -1, :]  # [B,H]
        logit = self.classifier(last)  # [B,1]
        return logit


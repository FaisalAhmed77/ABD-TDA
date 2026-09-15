"""
Model definitions for the ABD-TDA fusion framework:
  - TDANetwork: MLP side-branch that embeds topological (Betti-0/1) features
  - ViTTDAFusion: pretrained ViT + TDANetwork, fused for final classification
"""

import torch
import torch.nn as nn
from transformers import ViTModel

import config


class TDANetwork(nn.Module):
    """Side-branch MLP that projects raw TDA/HOG features into a compact embedding."""

    def __init__(self, input_dim, hidden_dim=config.SIDE_HIDDEN_DIM):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class ViTTDAFusion(nn.Module):
    """
    Fuses a pretrained ViT's [CLS] representation with the TDANetwork's
    embedding of topological features, then classifies the concatenation.
    """

    def __init__(self, side_dim, num_classes=config.NUM_CLASSES,
                 vit_model_name=config.VIT_MODEL_NAME):
        super().__init__()

        self.vit = ViTModel.from_pretrained(vit_model_name)
        vit_feature_dim = config.VIT_FEATURE_DIM

        self.side_branch = TDANetwork(side_dim)

        self.classifier = nn.Sequential(
            nn.Linear(vit_feature_dim + config.SIDE_HIDDEN_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, images, side_features):
        vit_outputs = self.vit(pixel_values=images)
        vit_features = vit_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        side_embedding = self.side_branch(side_features)
        combined = torch.cat((vit_features, side_embedding), dim=1)
        return self.classifier(combined)

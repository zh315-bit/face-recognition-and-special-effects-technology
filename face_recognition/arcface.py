"""ArcFace additive angular-margin classification head."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=64.0, margin=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, embeddings, labels):
        if labels.ndim != 1 or labels.numel() != embeddings.size(0):
            raise ValueError("labels must contain one value per embedding")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.weight.size(0)):
            raise ValueError("labels are out of range")
        # Keep the angular-margin math in float32 under CUDA autocast.
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).float().clamp(-1 + 1e-7, 1 - 1e-7)
        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        logits = cosine.clone()
        logits[torch.arange(labels.size(0), device=labels.device), labels] = phi[
            torch.arange(labels.size(0), device=labels.device), labels
        ]
        return logits * self.scale

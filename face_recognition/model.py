"""ResNet50-IR encoder for 112x112 aligned face crops."""

import torch
from torch import nn
from torch.nn import functional as F


class IRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.PReLU(out_channels), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.body(x) + self.shortcut(x)


class ResNet50IR(nn.Module):
    def __init__(self, embedding_size=512, dropout=0.4):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))
        channels, depths = [64, 128, 256, 512], [3, 4, 14, 3]
        blocks, current = [], 64
        for out, depth in zip(channels, depths):
            # Four stage boundaries reduce 112x112 features to 7x7.
            blocks.append(IRBlock(current, out, 2)); current = out
            blocks.extend(IRBlock(current, current) for _ in range(depth - 1))
        self.body = nn.Sequential(*blocks)
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(dropout), nn.Flatten(), nn.Linear(512 * 7 * 7, embedding_size), nn.BatchNorm1d(embedding_size))

    def forward(self, images):
        return F.normalize(self.output_layer(self.body(self.input_layer(images))), dim=1)

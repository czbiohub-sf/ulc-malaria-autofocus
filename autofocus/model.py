import torch
from torch import nn


class AutoFocus(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 16, 7, stride=2, padding=5, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 5, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 5, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.GELU(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 1, 3, stride=2, padding=1),
                nn.Flatten(start_dim=1),
                nn.Linear(520, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            ),
        )

    def forward(self, x):
        return self.model(x)

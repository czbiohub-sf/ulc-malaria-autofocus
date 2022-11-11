import torch
from torch import nn


class AutoFocus(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=5),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(5, stride=5),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(5, stride=2),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(5, stride=2),
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(32, 1, 3, padding=1), nn.Flatten(start_dim=1), nn.Linear(300, 1)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        return x

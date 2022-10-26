import torch
from torch import nn


class AutoFocus(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, 11, padding=(5, 5)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(5),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(32, 32, 11, padding=(5, 5)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(5),
            nn.Flatten(start_dim=1),
        )
        self.fc_block = nn.Sequential(
            nn.Linear(1536, 1024), nn.LeakyReLU(), nn.Dropout(0.5), nn.Linear(1024, 1)
        )

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.fc_block(x)
        return x

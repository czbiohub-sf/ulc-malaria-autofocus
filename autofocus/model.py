import torch
from torch import nn


class AutoFocus(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 16, 5, padding=5, bias=False),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.Conv2d(16, 16, 5, stride=2, bias=False),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=3, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 5, stride=2, bias=False),
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=2, bias=False),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
                nn.Conv2d(32, 32, 5, stride=2, bias=False),
            ),
            nn.Sequential(
                nn.Conv2d(32, 1, 3, padding=1),
                nn.Flatten(start_dim=1),
                nn.Linear(1850, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 1),
            ),
        )

        self.model.apply(self.init_network_weights)

    @staticmethod
    def init_network_weights(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            # init weights to default leaky relu neg slope, biases to 0
            torch.nn.init.kaiming_normal_(
                module.weight, a=0.01, nonlinearity="leaky_relu"
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)

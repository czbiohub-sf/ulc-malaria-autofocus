import torch
from torch import nn


AutoFocus = nn.Sequential(
    nn.Sequential(
        nn.Conv2d(1, 16, 5, padding=5),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(),
        nn.MaxPool2d(5, stride=5),
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, 3, padding=3),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(5, stride=2),
    ),
    nn.Sequential(
        nn.Conv2d(32, 32, 3, padding=2),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.MaxPool2d(5, stride=2),
    ),
    nn.Sequential(
        nn.Conv2d(32, 1, 3, padding=1), nn.Flatten(start_dim=1), nn.Linear(300, 1)
    ),
)

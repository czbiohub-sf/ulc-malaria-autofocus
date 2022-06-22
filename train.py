#! /usr/bin/env python3

# import wandb
# wandb.init("autofocus")

import torch

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from model import AutoFocus


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)


EPOCHS = 10



DATA_DIRS = "/Volumes/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"

transforms = Compose(
    [Resize([150, 200]), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)]
)
# full_dataset = FocusDataSet(DATA_DIRS)
full_dataset = datasets.ImageFolder(root=DATA_DIRS, transform=transforms, loader=read_image)


test_size = int(0.9999 * len(full_dataset))
train_size = len(full_dataset) - test_size
testing_dataset, training_dataset = random_split(full_dataset, [test_size, train_size])

test_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)
# test_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)
# train_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)


device = torch.device("mps")


# wandb.config = {
#     "learning_rate": 3e-4,
#     "epochs": 10,
#     "training_set_size": train_size,
#     "device": "cpu"
# }


def train():
    net = AutoFocus()
    L2 = nn.MSELoss()
    optimizer = Adam(net.parameters(), lr=3e-4)

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):

            imgs, labels = data

            print(imgs.dtype)
            print(net.parameters)
            optimizer.zero_grad()

            outputs = net(imgs)
            loss = L2(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"loss: {loss}")

            # wandb.log({"test_loss": loss})


if __name__ == "__main__":
    train()

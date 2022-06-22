#! /usr/bin/env python3

import wandb

wandb.init("autofocus")

import torch

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

from model import AutoFocus

import matplotlib.pyplot as plt
import time


EPOCHS = 20
BATCH_SIZE = 32
device = torch.device("mps")

DATA_DIRS = "/Volumes/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"

transforms = Compose(
    [Resize([150, 200]), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)]
)
full_dataset = datasets.ImageFolder(
    root=DATA_DIRS, transform=transforms, loader=read_image
)

test_size = int(0.2 * len(full_dataset))
validation_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - test_size - validation_size

testing_dataset, validation_dataset, training_dataset = random_split(
    full_dataset, [test_size, validation_size, train_size]
)

test_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=BATCH_SIZE, shuffle=True
)
train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)


wandb.config = {
    "learning_rate": 3e-4,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "training_set_size": train_size,
    "device": str(device),
}


def train(dev):
    net = AutoFocus().to(dev)
    L2 = nn.MSELoss().to(dev)
    optimizer = Adam(net.parameters(), lr=3e-4)
    losses = []

    for epoch in range(EPOCHS):
        running_loss = 0.0
        for data in train_dataloader:
            imgs, labels = data
            imgs = imgs.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()

            outputs = net(imgs).reshape(-1)
            loss = L2(outputs, labels.float())

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"({epoch}) loss: {loss}")
            wandb.log({"test_loss": loss})

        val_loss = 0.0
        for data in validation_dataloader:
            imgs, labels = data
            imgs = imgs.to(dev)
            labels = labels.to(dev)

            with torch.no_grad():
                outputs = net(imgs).reshape(-1)
                loss = L2(outputs, labels.float())
                val_loss += loss.item()

        wandb.log(
            {
                "avg_test_loss": running_loss / len(train_dataloader),
                "avg_val_loss": val_loss / len(validation_dataloader),
            }
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_test_loss": running_loss / len(train_dataloader),
                "avg_val_loss": val_loss / len(validation_dataloader),
            },
            "trained_models/model_{time.time()}.pth",
        )

    test_loss = 0.0
    for data in test_dataloader:
        imgs, labels = data
        imgs = imgs.to(dev)
        labels = labels.to(dev)

        with torch.no_grad():
            outputs = net(imgs).reshape(-1)
            loss = L2(outputs, labels.float())
            test_loss += loss.item()

    print(f"Final test loss: {test_loss / len(test_dataloader)}")


if __name__ == "__main__":
    train(device)

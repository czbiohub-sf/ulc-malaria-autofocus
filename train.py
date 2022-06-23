#! /usr/bin/env python3


import torch
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

import wandb
from model import AutoFocus
from dataloader import get_dataloader

from copy import deepcopy

EPOCHS = 10
ADAM_LR = 3e-4
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIRS = "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"


test_dataloader, validate_dataloader, train_dataloader = get_dataloader(
    DATA_DIRS, BATCH_SIZE, [0.2, 0.05, 0.75]
)


wandb.init(
    "autofocus",
    config={
        "learning_rate": ADAM_LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_set_size": len(train_dataloader),
        "device": str(device),
    },
)


def train(dev):
    net = AutoFocus().to(dev)
    L2 = nn.MSELoss().to(dev)
    optimizer = Adam(net.parameters(), lr=ADAM_LR)

    for epoch in range(EPOCHS):
        for i, data in enumerate(train_dataloader, 0):
            imgs, labels = data
            imgs = imgs.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()

            outputs = net(imgs).reshape(-1)
            loss = L2(outputs, labels.float())
            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss})

            if i % 100 == 0:
                val_loss = 0.0
                for data in validate_dataloader:
                    imgs, labels = data
                    imgs = imgs.to(dev)
                    labels = labels.to(dev)

                    with torch.no_grad():
                        outputs = net(imgs).reshape(-1)
                        loss = L2(outputs, labels.float())
                        val_loss += loss.item()

                wandb.log(
                    {
                        "avg_val_loss": val_loss / len(validate_dataloader),
                    }
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": deepcopy(net.state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "avg_val_loss": val_loss / len(validate_dataloader),
                    },
                    f"trained_models/{wandb.run.name}_{epoch}_{i}.pth",
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

    wandb.log({"test loss": test_loss})
    print(f"final average test loss: {test_loss / len(test_dataloader)}")


if __name__ == "__main__":
    print(f"using device {device}")
    train(device)

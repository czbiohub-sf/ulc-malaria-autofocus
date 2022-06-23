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
from copy import deepcopy


class ImageFolderWithLabels(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.target_transform = lambda idx: int(self.idx_to_class[idx])


EPOCHS = 10
ADAM_LR = 3e-4
BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIRS = "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"

transforms = Compose(
    [Resize([150, 200]), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)]
)

full_dataset = ImageFolderWithLabels(
    root=DATA_DIRS, transform=transforms, loader=read_image
)

test_size = int(0.2 * len(full_dataset))
validation_size = int(0.05 * len(full_dataset))
train_size = len(full_dataset) - test_size - validation_size

testing_dataset, validate_dataset, training_dataset = random_split(
    full_dataset, [test_size, validation_size, train_size]
)

test_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=True)
validate_dataloader = DataLoader(validate_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)


wandb.init(
    "autofocus",
    config={
        "learning_rate": ADAM_LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_set_size": train_size,
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

#! /usr/bin/env python3


import torch
from torch import nn
from torch.optim import AdamW

import wandb
from model import AutoFocus
from dataloader import get_dataloader
from nn_analysis import get_confusion_data

from pathlib import Path
from copy import deepcopy
from typing import List


EPOCHS = 256
ADAM_LR = 3e-4
BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATA_DIRS = "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"
DATA_DIRS = "/tmp/training_data"

exclude_classes: List[str] = []
test_dataloader, validate_dataloader, train_dataloader = get_dataloader(
    DATA_DIRS,
    BATCH_SIZE,
    [0.2, 0.03, 0.77],
    exclude_classes=exclude_classes,
)

wandb.init(
    "autofocus",
    config={
        "learning_rate": ADAM_LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_set_size": len(train_dataloader),
        "device": str(device),
        "exclude_classes": exclude_classes,
        "classes": train_dataloader.dataset.dataset.classes,
    },
)


def train(dev):
    net = AutoFocus().to(dev)
    L2 = nn.MSELoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)
    confusion_tbl = wandb.Table(columns=["confusion_data", "confusion_stddev"])
    model_save_dir = Path(f"trained_models/{wandb.run.name}")
    model_save_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(EPOCHS):
        for i, data in enumerate(train_dataloader, 1):
            imgs, labels = data
            imgs = imgs.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()

            outputs = net(imgs).reshape(-1)
            loss = L2(outputs, labels.float())
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item(), "epoch": epoch}, commit=(i % 10 == 0), step=i)

            if i % 100 == 0:
                val_loss = 0.0

                net.eval()
                for data in validate_dataloader:
                    imgs, labels = data
                    imgs = imgs.to(dev)
                    labels = labels.to(dev)

                    with torch.no_grad():
                        outputs = net(imgs).reshape(-1)
                        loss = L2(outputs, labels.float())
                        val_loss += loss.item()

                _, confusion_outputs, confusion_stddev = get_confusion_data(
                    net,
                    validate_dataloader.dataset.dataset,
                    sample_size=BATCH_SIZE,
                    device=device,
                )
                confusion_tbl.add_data(confusion_outputs, confusion_stddev)
                wandb.log(
                    {
                        "val_loss": val_loss / len(validate_dataloader),
                    }
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": deepcopy(net.state_dict()),
                        "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                        "avg_val_loss": val_loss / len(validate_dataloader),
                    },
                    str(model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"),
                )
                net.train()

    net.eval()
    test_loss = 0.0
    for data in test_dataloader:
        imgs, labels = data
        imgs = imgs.to(dev)
        labels = labels.to(dev)

        with torch.no_grad():
            outputs = net(imgs).reshape(-1)
            loss = L2(outputs, labels.float())
            test_loss += loss.item()

    wandb.log(
        {
            "test_loss": test_loss / len(test_dataloader),
            "confusion_table": confusion_tbl,
        }
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": deepcopy(net.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            "average_test_loss": test_loss / len(test_dataloader),
        },
        str(model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"),
    )


if __name__ == "__main__":
    print(f"using device {device}")
    train(device)

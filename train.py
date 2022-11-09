#! /usr/bin/env python3

import sys

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

import wandb
from model import AutoFocus
from dataloader import get_dataloader
from alrc import AdaptiveLRClipping
from argparsers import train_parser

from pathlib import Path
from copy import deepcopy
from typing import List

torch.backends.cuda.matmul.allow_tf32 = True


EPOCHS = 128
ADAM_LR = 3e-4
BATCH_SIZE = 256


def checkpoint_model(model, epoch, optimizer, name):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": deepcopy(model.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
        },
        str(name),
    )


def init_dataloaders(config):
    dataloaders = get_dataloader(
        config["dataset_descriptor_file"],
        BATCH_SIZE,
    )

    test_dataloader = dataloaders["test"]
    validate_dataloader = dataloaders["val"]
    train_dataloader = dataloaders["train"]

    wandb.config.update(
        {
            "training set size": f"{len(train_dataloader) * config['batch_size']} images",
            "validation set size": f"{len(validate_dataloader) * config['batch_size']} images",
            "testing set size": f"{len(test_dataloader) * config['batch_size']} images",
        }
    )

    if wandb.run.name is not None:
        model_save_dir = Path(f"trained_models/{wandb.run.name}")
    else:
        model_save_dir = Path(
            f"trained_models/unnamed_run_{torch.randint(100, size=(1,)).item()}"
        )
    model_save_dir.mkdir(exist_ok=True, parents=True)

    return model_save_dir, train_dataloader, validate_dataloader, test_dataloader


def train(dev):
    net = AutoFocus().to(dev)
    L2 = nn.MSELoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)

    (
        model_save_dir,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
    ) = init_dataloaders(wandb.config)

    anneal_period = wandb.config["epochs"] * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=anneal_period, eta_min=3e-5)

    global_step = 0
    for epoch in range(wandb.config["epochs"]):
        for i, (imgs, labels) in enumerate(train_dataloader, 1):
            global_step += 1
            imgs = imgs.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs).reshape(-1)
            loss = L2(outputs, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log(
                {
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "LR": scheduler.get_last_lr()[0],
                },
                commit=False,
                step=global_step,
            )

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

        wandb.log(
            {"val_loss": val_loss / len(validate_dataloader)},
        )
        checkpoint_model(
            net, epoch, optimizer, model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"
        )
        net.train()

    print("done training")
    net.eval()
    test_loss = 0.0
    for data in test_dataloader:
        imgs, labels = data
        imgs = imgs.to(dev)
        labels = labels.to(dev)

        with torch.no_grad():
            outputs = net(imgs).reshape(-1)
            loss = L2(outputs, labels.half())
            test_loss += loss.item()

    wandb.log(
        {
            "test_loss": test_loss / len(test_dataloader),
        },
    )
    checkpoint_model(
        net, epoch, optimizer, model_save_dir / f"{wandb.run.name}_{epoch}_{i}.pth"
    )


def do_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        "autofocus",
        entity="bioengineering",
        config={
            "learning_rate": ADAM_LR,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "device": str(device),
            "dataset_descriptor_file": args.dataset_descriptor_file,
            "run group": args.group,
        },
        notes=args.note,
        tags=["v0.0.2"],
    )

    train(device)


if __name__ == "__main__":
    parser = train_parser()
    args = parser.parse_args()

    do_training(args)

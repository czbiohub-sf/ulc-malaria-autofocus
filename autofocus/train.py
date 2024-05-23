#! /usr/bin/env python3

import os
import wandb
import torch

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from autofocus.model import AutoFocus
from autofocus.argparsers import train_parser
from autofocus.dataloader import get_dataloader

from pathlib import Path
from copy import deepcopy


def checkpoint_model(model, epoch, optimizer, name, img_size):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": deepcopy(model.state_dict()),
            "optimizer_state_dict": deepcopy(optimizer.state_dict()),
            "img_size": img_size,
        },
        str(name),
    )


def init_dataloaders(config):
    dataloaders = get_dataloader(
        config["dataset_descriptor_file"],
        config["batch_size"],
        img_size=config["resize_shape"],
        color_jitter=config["color_jitter"],
        random_erasing=config["random_erasing"],
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
    config = wandb.config

    net = AutoFocus().to(dev)
    net = torch.jit.script(net)

    L2 = nn.MSELoss().to(dev)
    optimizer = AdamW(
        net.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    (
        model_save_dir,
        train_dataloader,
        validate_dataloader,
        test_dataloader,
    ) = init_dataloaders(config)

    anneal_period = config["epochs"] * len(train_dataloader)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=anneal_period, eta_min=config["learning_rate"] / 3
    )

    best_val_loss = 1e10
    global_step = 0
    for epoch in range(config["epochs"]):
        for i, (imgs, labels) in enumerate(train_dataloader, 1):
            global_step += 1
            imgs = imgs.to(dev, dtype=torch.float, non_blocking=True)
            labels = labels.to(dev, dtype=torch.float, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs).view(-1)
            loss = L2(outputs, labels)
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
        for imgs, labels in validate_dataloader:
            imgs = imgs.to(dev, dtype=torch.float, non_blocking=True)
            labels = labels.to(dev, dtype=torch.float, non_blocking=True)

            with torch.no_grad():
                outputs = net(imgs).view(-1)
                loss = L2(outputs, labels)
                val_loss += loss.item()

        wandb.log(
            {"val_loss": val_loss / len(validate_dataloader)},
        )

        if val_loss < best_val_loss:
            checkpoint_model(
                net,
                epoch,
                optimizer,
                model_save_dir / "best.pth",
                config["resize_shape"],
            )
            best_val_loss = val_loss

        checkpoint_model(
            net, epoch, optimizer, model_save_dir / "latest.pth", config["resize_shape"]
        )

        net.train()

    # load best!
    net = AutoFocus.from_pth(model_save_dir / "best.pth")
    net = net.to(dev)
    net = torch.jit.script(net)

    print("done training")
    net.eval()
    test_loss = torch.tensor(0.0, device=dev)
    for data in test_dataloader:
        imgs, labels = data
        imgs = imgs.to(dev, dtype=torch.float, non_blocking=True)
        labels = labels.to(dev, dtype=torch.float, non_blocking=True)

        with torch.no_grad():
            outputs = net(imgs).view(-1)
            loss = L2(outputs, labels)
            test_loss += loss

    wandb.log(
        {
            "test_loss": test_loss / len(test_dataloader),
        },
    )


def do_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 192
    BATCH_SIZE = 256

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    wandb.init(
        project="ulc-malaria-autofocus",
        entity="bioengineering",
        config={
            "learning_rate": args.lr,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "weight_decay": 0.01,
            "device": str(device),
            "resize_shape": args.resize,
            "dataset_descriptor_file": args.dataset_descriptor_file,
            "run group": args.group,
            "slurm-job-id": os.getenv("SLURM_JOB_ID", default=None),
            "torch.backends.cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "color_jitter": args.color_jitter,
            "random_erasing": args.random_erasing,
        },
        notes=args.note,
        tags=["v0.0.2"],
    )

    train(device)


if __name__ == "__main__":
    parser = train_parser()
    args = parser.parse_args()

    do_training(args)

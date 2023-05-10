#! /usr/bin/env python3


import sys
import torch

import torch
from torch import nn
from torch.optim import AdamW
from torch.multiprocessing import set_start_method
from torch.profiler import profile, ProfilerActivity, record_function
from torch.optim.lr_scheduler import CosineAnnealingLR

from autofocus.model import AutoFocus
from autofocus.dataloader import get_dataloader


WARMUP = 2
ADAM_LR = 3e-4
BATCH_SIZE = 256


class MockedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoFocus()
        self.model = torch.jit.script(self.model)

    def forward(self, *args, **kwargs):
        with record_function("MODEL FORWARD"):
            return self.model(*args, **kwargs)


def profile_run(
    dev,
    train_dataloader,
    img_size,
):
    print("dev is ", dev)
    net = MockedModel().to(dev)
    L2 = nn.MSELoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)
    scheduler = CosineAnnealingLR(
        optimizer, T_max=8 * len(train_dataloader), eta_min=ADAM_LR / 10
    )

    print("here we goooooo!")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        schedule=torch.profiler.schedule(
            wait=6,
            warmup=12,
            active=48,
        ),
    ) as prof:
        for i, (imgs, labels) in enumerate(train_dataloader):
            imgs = imgs.to(dev, dtype=torch.float, non_blocking=True)
            labels = labels.to(dev, dtype=torch.float, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs = net(imgs).view(-1)
            loss = L2(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            prof.step()

            if i > 6 + 12 + 48:
                break

        return prof


if __name__ == "__main__":
    device = "cuda"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloaders = get_dataloader(
        sys.argv[1],
        BATCH_SIZE,
    )

    test_dataloader, validate_dataloader, train_dataloader = (
        dataloaders["test"],
        dataloaders["val"],
        dataloaders["test"],
    )

    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    torch.backends.cudnn.benchmark = True
    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    torch.backends.cuda.matmul.allow_tf32 = True

    # TODO: EPOCH and BATCH_SIZE and img_size in yml file?
    resize_target_size = (300, 400)

    prof = profile_run(
        device,
        train_dataloader,
        resize_target_size,
    )

    prof.export_chrome_trace("chrome_profile.json")
    print("I am done!")

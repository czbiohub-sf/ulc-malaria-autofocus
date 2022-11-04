#! /usr/bin/env python3


import torch

import torch
from torch import nn
from torch.optim import AdamW
from torch.multiprocessing import set_start_method
from torch.profiler import profile, ProfilerActivity, record_function

from model import AutoFocus
from dataloader import get_dataloader

from pathlib import Path
from copy import deepcopy
from typing import List

WARMUP = 2
ADAM_LR = 3e-4
BATCH_SIZE = 32


class MockedModel(AutoFocus):
    def forward(self, *args, **kwargs):
        with record_function("MODEL FORWARD"):
            return super().forward(*args, **kwargs)


def profile_run(
    dev,
    train_dataloader,
    validate_dataloader,
    test_dataloader,
    img_size,
):
    print('dev is ', dev)
    net = MockedModel().to(dev)
    L2 = nn.MSELoss().to(dev)
    optimizer = AdamW(net.parameters(), lr=ADAM_LR)

    print("warming up")
    for epoch in range(WARMUP):
        outputs = net(torch.randn(1, 1, *img_size, device=dev))
    net.zero_grad()

    print("here we goooooo!")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for i, (imgs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad(set_to_none=True)
            print('is cuda available?', torch.cuda.is_available())
            print('in loop dev is', dev)
            print(imgs.device)
            imgs.cuda()
            labels.cuda()
            print(imgs.device)

            outputs = net(imgs)
            loss = L2(outputs, labels)
            loss.backward()
            optimizer.step()

            if i == 5:
                break

        return prof


if __name__ == "__main__":
    set_start_method("spawn")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


    test_dataloader, validate_dataloader, train_dataloader = get_dataloader(
        "/tmp/training_data",
        BATCH_SIZE,
        [0.2, 0.05, 0.75],
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
        validate_dataloader,
        test_dataloader,
        resize_target_size,
    )

    prof.export_chrome_trace("chrome_profile.json")
    print("I am done!")

#! /usr/bin/env python3

import sys
import time

import numpy as np

import torch
from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

import matplotlib.pyplot as plt

from model import AutoFocus
from dataloader import get_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIRS = "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"


def get_confusion_data(net, dataset, sample_size=100, device=torch.device("cpu")):
    prev_net_state = net.training

    net.eval()

    outputs, std_devs = [], []
    classes_in_order = sorted([int(c) for c in dataset.classes])
    for clss in classes_in_order:
        samples = dataset.sample_from_class(int(clss), sample_size).to(device)
        with torch.no_grad():
            out = net(samples)
            output_stddev, output_mean = torch.std_mean(out, unbiased=True)
        outputs.append(output_mean.item())
        std_devs.append(output_stddev.item())

    net.train(prev_net_state)

    return classes_in_order, outputs, std_devs


if __name__ == "__main__":
    assert len(sys.argv) == 3, "usage: ./visualize_trained.py <PATH TO PTH> <PATH TO IMAGE>"

    dev = torch.device("cpu")

    model_save = torch.load(sys.argv[1], map_location=dev)

    net = AutoFocus()
    net.eval()
    net.load_state_dict(model_save["model_state_dict"])
    net.to(dev)

    img = read_image(sys.argv[2])
    transforms = Compose(
        [Resize([150, 200])]
    )
    preprocessed = transforms(img)
    preprocessed.unsqueeze_(dim=0)
    preprocessed.to(dev)

    with torch.no_grad():
        t0 = time.perf_counter()
        res = net(preprocessed)
        t1 = time.perf_counter()
    print(f"got {res} in {t1 - t0} sec")

#! /usr/bin/env python3

import sys

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

from model import AutoFocus
from dataloader import get_dataset

from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device("mps")

DATA_DIRS = "training_data"


def get_confusion_data(net, data_dir, sample_size=100):
    [full_dataset_subset] = get_dataset(data_dir, sample_size)
    full_dataset = full_dataset_subset.dataset

    outputs = []
    classes_in_order = sorted([int(c) for c in full_dataset.classes])
    for clss in tqdm(classes_in_order):
        samples = full_dataset.sample_from_class(int(clss), sample_size)
        out = net(samples).mean()
        outputs.append(out.item())

    return classes_in_order, outputs


if __name__ == "__main__":
    assert len(sys.argv) == 2, "usage: ./visualize_trained.py <PATH TO PTH>"

    dev = torch.device("cpu")

    model_save = torch.load(sys.argv[1], map_location=torch.device("cpu"))

    net = AutoFocus()
    net.train(False)
    net.load_state_dict(model_save["model_state_dict"])

    classes_in_order, outputs = get_confusion_data(net, DATA_DIRS, sample_size=10)
    plt.plot(classes_in_order, classes_in_order)
    plt.plot(classes_in_order, outputs)
    plt.show()

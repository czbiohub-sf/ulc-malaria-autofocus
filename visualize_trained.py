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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIRS = "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"


def get_confusion_data(net, dataset, sample_size=100, device=torch.device("cpu")):
    prev_net_state = net.training

    net.eval()

    outputs, variances = [], []
    classes_in_order = sorted([int(c) for c in dataset.classes])
    for clss in classes_in_order:
        samples = dataset.sample_from_class(int(clss), sample_size).to(device)
        with torch.no_grad():
            out = net(samples)
            output_num = out.mean().item()
            output_var = out.var(unbiased=True)
        outputs.append(output_num)
        variances.append(output_var)

    net.train(prev_net_state)

    return classes_in_order, outputs, variances


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    assert len(sys.argv) == 2, "usage: ./visualize_trained.py <PATH TO PTH>"

    dev = torch.device("cpu")

    model_save = torch.load(sys.argv[1], map_location=torch.device("cpu"))

    net = AutoFocus()
    net.eval()
    net.load_state_dict(model_save["model_state_dict"])

    sample_size = 5
    [full_dataset_subset] = get_dataset("training_data", sample_size)
    full_dataset = full_dataset_subset.dataset
    classes_in_order, outputs, variances = get_confusion_data(
        net, full_dataset, sample_size=sample_size
    )
    outputs = np.asarray(outputs)
    variances = np.asarray(variances)
    plt.scatter(classes_in_order, classes_in_order, s=2)
    plt.scatter(classes_in_order, [np.round(o) for o in outputs], c="red", s=2)
    plt.fill_between(
        classes_in_order,
        outputs - variances,
        outputs + variances,
        alpha=0.2,
        edgecolor="#1B2ACC",
        facecolor="#089FFF",
    )
    plt.show()

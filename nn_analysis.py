#! /usr/bin/env python3

import re
import sys
import time

import numpy as np

from pathlib import Path

import torch
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (
    ToTensor,
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

import matplotlib.pyplot as plt

from model import AutoFocus
from dataloader import get_dataset


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


def load_image_data(path_to_data: str, dev=torch.device("cpu")):
    "takes a path to either a single png image or a folder of pngs"
    datapath = Path(path_to_data)
    data = [datapath] if datapath.is_file() else datapath.glob("*.png")
    transforms = Resize([150, 200])
    for img_name in sorted(data):
        if img_name == "":
            continue
        image = read_image(str(img_name), mode=ImageReadMode.GRAY)
        preprocessed = transforms(image)
        preprocessed.unsqueeze_(dim=0)
        preprocessed.to(dev)
        yield img_name, preprocessed


def load_model_for_inference(path_to_pth: str, dev=torch.device("cpu")):
    model_save = torch.load(path_to_pth, map_location=dev)

    net = AutoFocus()
    net.eval()
    net.load_state_dict(model_save["model_state_dict"])
    net.to(dev)

    return net


if __name__ == "__main__":
    assert len(sys.argv) == 3, f"usage: {sys.argv[0]} <PATH TO PTH> <PATH TO IMAGE>"

    net = load_model_for_inference(sys.argv[1])

    motor_steps = []
    preds = []

    for img_name, img in load_image_data(sys.argv[2]):
        with torch.no_grad():
            t0 = time.perf_counter()
            res = net(img)
            t1 = time.perf_counter()

        matches = re.search("(\d+)\.png", str(img_name))
        if matches is not None:
            num = int(matches.group(1))
            motor_steps.append(num)
            preds.append(res.item())

            print(f"{num} got {res.item()} in {t1 - t0} sec")

    plt.plot(motor_steps, preds)
    plt.title("Motor position from home vs. predicted steps from focus (on my Mac)")
    plt.show()

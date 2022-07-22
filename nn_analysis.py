#! /usr/bin/env python3

import re
import csv
import sys
import time
import zarr

import numpy as np

from pathlib import Path

import torch
from torchvision.transforms import Resize
from torchvision.io import read_image, ImageReadMode

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


def load_metadata_csv(fname: str):
    with open(fname, "r") as f:
        data = csv.DictReader(f)
        return {
            im_count: row
            for im_count, row in enumerate(data)
        }


def open_zarr_data(fname: str):
    return zarr.open(fname)


def infer_image(path_to_pth, path_to_image_data):
    net = load_model_for_inference(path_to_pth)

    motor_steps = []
    preds = []

    for img_name, img in load_image_data(path_to_image_data):
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


if __name__ == "__main__":
    model = load_model_for_inference("trained_models/efficient-donkey-final.pth")
    transforms = Resize([150, 200])

    image_data = open_zarr_data("testing_data/2022-07-21-161530_SSAF_start_at_peak_focus.zip")
    metadata = load_metadata_csv("testing_data/2022-07-21-161530_SSAF_start_at_peak_focus_metadata.csv")
    start_pos = metadata[0]["motor_pos"]
    print(f"start pos is {start_pos}")

    preds, Ys = [], []
    print("starting inference...")
    for step in sorted(image_data, key=int):
        img = torch.tensor(np.array(image_data[step]))
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        preprocessed = transforms(img)
        metadata_row = metadata[int(step)]

        preds.append(model(preprocessed).item())
        Ys.append(float(metadata_row["motor_pos"]) - float(start_pos))

    print("starting plotting...")
    ax = plt.gca()
    print(len(Ys))
    times = [f"{int((f / 30) // 60)}:{int((f / 30) % 60)}" for f in range(len(preds))]
    plt.plot(range(len(times)), preds)
    plt.plot(range(len(times)), Ys)
    posses = [(i, t) for i, t in enumerate(times) if i % (30 * 30) == 0]
    xposses = [t[0] for t in posses]
    labels = [t[1] for t in posses]
    ax.set(xticks=xposses, xticklabels=labels)
    plt.legend(["predictions", "actuals"])
    plt.show()

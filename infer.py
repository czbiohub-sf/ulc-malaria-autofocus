#! /usr/bin/env python3

import sys
import zarr

import torch
import numpy as np

from pathlib import Path
from model import AutoFocus
from argparsers import infer_parser
from dataloader import get_datasets, read_grayscale

from torchvision.transforms import Resize, Compose, ToTensor


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_data(path_to_data: str):
    "takes a path to either a single png image or a folder of pngs"
    device = choose_device()
    transforms = Resize([300, 400])

    datapath = Path(path_to_data)
    data = [datapath] if datapath.is_file() else datapath.glob("*.png")

    for img_name in sorted(data):
        image = read_grayscale(img_name)
        preprocessed = transforms(image)
        preprocessed.unsqueeze_(dim=0)
        preprocessed.to(device)
        yield img_name, preprocessed


def load_zarr_data(path_to_zarr: str):
    device = choose_device()
    data = zarr.open(path_to_zarr)
    transform = Compose([ToTensor(), Resize([300, 400])])
    for i in range(len(data)):
        img = transform(data[i][:])
        img.unsqueeze_(dim=0)
        img.to(device)
        yield img


def load_model_for_inference(path_to_pth: str):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_save = torch.load(path_to_pth, map_location=device)

    net = AutoFocus()
    net.load_state_dict(model_save["model_state_dict"])
    net.eval()
    net.to(device)

    return net


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()

    no_imgs = args.images is None
    no_zarr = args.zarr is None
    if (no_imgs and no_zarr) or (not no_imgs and not no_zarr):
        print("you must supply a value for only one of --images or --zarr")
        sys.exit(1)

    model = load_model_for_inference(args.pth_path)
    image_loader = (
        load_image_data(args.images) if no_zarr else load_zarr_data(args.zarr)
    )

    with torch.no_grad():
        for image in image_loader:
            res = model(image)
            print(res.item())

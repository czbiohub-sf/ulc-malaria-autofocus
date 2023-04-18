#! /usr/bin/env python3

import sys
import zarr

import torch
import allantools as at

from pathlib import Path

from torchvision.transforms import Resize, Compose, ToTensor

from autofocus.model import AutoFocus
from autofocus.argparsers import infer_parser
from autofocus.dataloader import read_grayscale


def _tqdm(iterable, **kwargs):
    return iterable


try:
    from tqdm import tqdm
except ImportError:
    tqdm = _tqdm


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_inference(path_to_pth: str, device: torch.device):
    model_save = torch.load(path_to_pth, map_location=device)

    net = AutoFocus()
    net.load_state_dict(model_save["model_state_dict"])
    net.eval()
    net.to(device)

    return net


def infer(model, image_loader):
    for image in image_loader:
        with torch.no_grad():
            res = model(image)
            yield res.item()


def calculate_allan_dev(model, image_loader):
    ds = at.Dataset(data=[v for v in infer(model, tqdm(image_loader))])
    res = ds.compute("tdev")
    res["taus"]
    res["stat"]

    with open("allan_dev_calc.txt", "w") as f:
        f.write("\n".join(f"{t},{s}" for t, s in zip(res["taus"], res["stat"])))

    pl = at.Plot()
    pl.plot(ds, errorbars=True, grid=True)
    pl.ax.set_xlabel("frames")
    pl.ax.set_ylabel("Allan Deviation")
    pl.show()


class ImageLoader:
    def __init__(self, _iter, _num_els):
        self._iter = _iter
        self._num_els = _num_els

    def __iter__(self):
        return self._iter()

    def __len__(self):
        if self._iter is None:
            raise RuntimeError(
                "instantiate ImageLoader with `load_image_data` or `load_zarr_data`"
            )

        return self._num_els

    @classmethod
    def load_image_data(cls, path_to_data: str):
        "takes a path to either a single png image or a folder of pngs"
        device = choose_device()
        transforms = Resize([300, 400])

        datapath = Path(path_to_data)
        data = [datapath] if datapath.is_file() else datapath.glob("*.png")

        _num_els = 1 if datapath.is_file() else sum(1 for _ in datapath.glob("*.png"))

        def _iter():
            for img_name in sorted(data):
                image = read_grayscale(img_name)
                preprocessed = transforms(image)
                preprocessed.unsqueeze_(dim=0)
                yield preprocessed.to(device)

        return cls(_iter, _num_els)

    @classmethod
    def load_zarr_data(cls, path_to_zarr: str):
        device = choose_device()
        data = zarr.open(path_to_zarr, mode="r")
        transform = Compose([ToTensor(), Resize([300, 400])])

        _num_els = data.initialized

        def _iter():
            for i in range(_num_els):
                # cheap trick
                img = transform(data[:, :, i]) * 255
                img.unsqueeze_(dim=0)
                yield img.to(device)

        return cls(_iter, _num_els)


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")

    no_imgs = args.images is None
    no_zarr = args.zarr is None
    no_output = args.output is None
    if (no_imgs and no_zarr) or (not no_imgs and not no_zarr):
        print("you must supply a value for only one of --images or --zarr")
        sys.exit(1)

    model = load_model_for_inference(args.pth_path, device)
    image_loader = (
        ImageLoader.load_image_data(args.images)
        if no_zarr
        else ImageLoader.load_zarr_data(args.zarr)
    )

    if args.allan_dev:
        calculate_allan_dev(model, image_loader)
    elif no_output:
        for res in infer(model, image_loader):
            print(res)
    else:
        with open(args.output, "w") as file:
            for res in infer(model, image_loader):
                file.write(f"{res}\n")
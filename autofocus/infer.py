#! /usr/bin/env python3

import sys
import zarr

import torch
import allantools as at

from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional

from torchvision.transforms import Resize, Compose, ToTensor

from autofocus.model import AutoFocus
from autofocus.argparsers import infer_parser
from autofocus.dataloader import read_grayscale


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_inference(
    path_to_pth: Union[str, Path], device: Union[str, torch.device]
):
    net = AutoFocus.from_pth(path_to_pth)
    net.eval()
    net.to(device)
    return net


def infer(model, image_loader):
    with torch.no_grad():
        for image in image_loader:
            res = model(image)
            yield res.item()


def calculate_allan_dev(model, image_loader):
    ds = at.Dataset(data=[v for v in infer(model, tqdm(image_loader))])
    res = ds.compute("tdev")

    pl = at.Plot()
    pl.plot(ds, errorbars=True, grid=True)
    pl.ax.set_xlabel("frames")
    pl.ax.set_ylabel("Allan Deviation")
    pl.show()


class ImageLoader:
    # TODO add batch size
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
    def load_image_data(
        cls, path_to_data: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ):
        "takes a path to either a single png image or a folder of pngs"
        transforms = Resize([300, 400], antialias=True)

        datapath = Path(path_to_data)
        data = [datapath] if datapath.is_file() else datapath.glob("*.png")

        _num_els = 1 if datapath.is_file() else sum(1 for _ in datapath.glob("*.png"))

        def _iter():
            for img_name in sorted(data):
                image = read_grayscale(img_name)
                preprocessed = transforms(image)
                preprocessed.unsqueeze_(dim=0)
                yield preprocessed.to(device, dtype=torch.float)

        return cls(_iter, _num_els)

    @classmethod
    def load_zarr_data(
        cls, path_to_zarr: Union[str, Path], device: Union[str, torch.device] = "cpu"
    ):
        data = zarr.open(path_to_zarr, mode="r")
        transform = Compose([ToTensor(), Resize([300, 400], antialias=True)])

        _num_els = data.initialized if isinstance(data, zarr.Array) else len(data)

        def _iter():
            for i in range(_num_els):
                # cheap trick
                img = data[:, :, i] if isinstance(data, zarr.Array) else data[i][:]
                img = transform(img) * 255
                img.unsqueeze_(dim=0)
                yield img.to(device)

        return cls(_iter, _num_els)


@torch.no_grad()
def predict(
    path_to_pth: Path,
    path_to_images: Optional[Path] = None,
    path_to_zarr: Optional[Path] = None,
    calc_allan_dev: bool = False,
    output: Optional[Path] = None,
    print_results: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> Optional[torch.Tensor]:
    model = load_model_for_inference(path_to_pth, device)
    model = torch.jit.script(model)

    image_loader = (
        ImageLoader.load_image_data(path_to_images, device=device)
        if path_to_images is not None
        else ImageLoader.load_zarr_data(
            path_to_zarr, device=device
        )
    )

    if calc_allan_dev:
        calculate_allan_dev(model, image_loader)
    elif output is None and print_results:
        for res in infer(model, image_loader):
            print(res)
    elif output is None:
        arr = torch.zeros(len(image_loader))
        for i, res in enumerate(infer(model, image_loader)):
            arr[i] = res
        return arr
    else:
        with open(args.output, "w") as file:
            for res in infer(model, image_loader):
                file.write(f"{res}\n")


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no_imgs = args.images is None
    no_zarr = args.zarr is None
    no_output = args.output is None
    if (no_imgs and no_zarr) or (not no_imgs and not no_zarr):
        print("you must supply a value for only one of --images or --zarr")
        sys.exit(1)

    predict(
        path_to_pth=args.pth_path,
        path_to_images=args.images,
        path_to_zarr=args.zarr,
        calc_allan_dev=args.allan_dev,
        output=args.output,
        print_results=args.print_results,
        device=device,
    )

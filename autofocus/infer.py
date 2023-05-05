#! /usr/bin/env python3

import sys
import zarr

import torch
import allantools as at
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from typing import Union, Optional

from torchvision.transforms import Resize, Compose, ToTensor

from autofocus.model import AutoFocus, AutoFocusOlder
from autofocus.argparsers import infer_parser
from autofocus.dataloader import read_grayscale


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_inference(
    path_to_pth: Union[str, Path], device: Union[str, torch.device]
):
    net: Union[AutoFocus, AutoFocusOlder]
    try:
        net = AutoFocus.from_pth(path_to_pth)
    except:
        net = AutoFocusOlder.from_pth(path_to_pth)
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
    ds.compute("tdev")

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
        cls,
        path_to_data: Union[str, Path],
        img_size=(300, 400),
        device: Union[str, torch.device] = "cpu",
    ):
        "takes a path to either a single png image or a folder of pngs"
        transforms = Resize(img_size, antialias=True)

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
        cls,
        path_to_zarr: Union[str, Path],
        img_size=(300, 400),
        device: Union[str, torch.device] = "cpu",
    ):
        data = zarr.open(path_to_zarr, mode="r")
        transform = Compose([ToTensor(), Resize(img_size, antialias=True)])

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
    plot: bool = False,
    output: Optional[Path] = None,
    device: Union[str, torch.device] = "cpu",
) -> Optional[torch.Tensor]:
    model = load_model_for_inference(path_to_pth, device)
    model = torch.jit.script(model)

    if path_to_images:
        image_loader = ImageLoader.load_image_data(
            path_to_images, img_size=model.img_size, device=device
        )
        data_path = path_to_images
    elif path_to_zarr:
        image_loader = ImageLoader.load_zarr_data(
            path_to_zarr, img_size=model.img_size, device=device
        )
        data_path = path_to_zarr
    else:
        raise ValueError("need path_to_images or path_to_zarr")

    if calc_allan_dev:
        calculate_allan_dev(model, image_loader)
        return None

    arr = torch.zeros(len(image_loader))
    for i, res in enumerate(tqdm(infer(model, image_loader))):
        arr[i] = res

    if plot:
        fix, ax = plt.subplots(figsize=(10, 10))
        ax.plot(arr)
        ax.set_ylim([-20, 20])
        ax.set_title(f"{data_path.name}\n{path_to_pth.parent.name}")

        if output is None:
            plt.show()
        else:
            plt.savefig(output.with_suffix(".png"), dpi=500)

    elif output is not None:
        with open(output.with_suffix(".txt"), "w") as f:
            f.write("\n".join(map(str, arr.tolist())))
    else:
        for r in arr:
            print(r.item())


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no_imgs = args.images is None
    no_zarr = args.zarr is None
    if (no_imgs and no_zarr) or (not no_imgs and not no_zarr):
        print("you must supply a value for only one of --images or --zarr")
        sys.exit(1)

    predict(
        path_to_pth=args.pth_path,
        path_to_images=args.images,
        path_to_zarr=args.zarr,
        calc_allan_dev=args.allan_dev,
        plot=args.plot,
        output=args.output_path,
        device=device,
    )

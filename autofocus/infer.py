#! /usr/bin/env python3

import sys
from pathlib import Path
from typing import Union, Optional

import allantools as at
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import Compose, ToTensor, CenterCrop
from tqdm import tqdm
import zarr

from autofocus.model import AutoFocus, AutoFocusOlder
from autofocus.argparsers import infer_parser
from autofocus.dataloader import read_grayscale, IMG_H, IMG_W
from autofocus.constants import CENTER_CROP_PERC


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
        img_center_crop_perc: float = CENTER_CROP_PERC,
        device: Union[str, torch.device] = "cpu",
    ):
        "takes a path to either a single png image or a folder of pngs"
        center_crop_h = int(IMG_H * img_center_crop_perc)
        center_crop_w = int(IMG_W * img_center_crop_perc)
        transforms = CenterCrop((center_crop_h, center_crop_w))

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
        center_crop_perc: float = CENTER_CROP_PERC,
        device: Union[str, torch.device] = "cpu",
    ):
        data = zarr.open(path_to_zarr, mode="r")

        center_crop_h = int(IMG_H * center_crop_perc)
        center_crop_w = int(IMG_W * center_crop_perc)
        transform = Compose([ToTensor(), CenterCrop((center_crop_h, center_crop_w))])

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
    overlay: bool = False,
    output: Optional[Path] = None,
    print_output: bool = False,
    device: Union[str, torch.device] = "cpu",
) -> Optional[torch.Tensor]:
    model = load_model_for_inference(path_to_pth, device)
    model = torch.jit.script(model)

    if path_to_images:
        image_loader = ImageLoader.load_image_data(
            path_to_images, img_center_crop_perc=model.center_crop_perc, device=device
        )
        data_path = path_to_images
    elif path_to_zarr:
        image_loader = ImageLoader.load_zarr_data(
            path_to_zarr, img_center_crop_perc=model.center_crop_perc, device=device
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
        _, ax = plt.subplots(figsize=(10, 10))
        ax.plot(arr)
        ax.set_ylim([-20, 20])
        ax.set_title(f"{data_path.name}\n{path_to_pth.parent.name}")
        ax.set_xlabel("frames")
        ax.set_ylabel("focus value")

        if output is None:
            plt.show()
        else:
            plt.savefig(output.with_suffix(".png"), dpi=500)

    elif path_to_images and overlay:
        output.mkdir(parents=True, exist_ok=True)
        image_loader = ImageLoader.load_image_data(
            path_to_images, img_center_crop_perc=model.center_crop_perc, device=device
        )
        for i, (img_path, img) in enumerate(
            zip(sorted(Path(path_to_images).glob("*.png")), image_loader)
        ):
            img = img.squeeze().cpu().numpy()
            # img = (img * 255).astype('uint8')  # Convert to uint8 for OpenCV
            cv2.putText(
                img,
                f"SSAF: {arr[i]:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            output_img_path = output / f"overlay_{img_path.name}"
            cv2.imwrite(str(output_img_path), img)

    elif output is not None:
        with open(output.with_suffix(".txt"), "w") as f:
            f.write("\n".join(map(str, arr.tolist())))
    elif print_output:
        for r in arr:
            print(r.item())
    else:
        return arr


@torch.no_grad()
def predict_training_data(
    path_to_pth: Path,
    path_to_dataset_defn_file_used_for_training: Path,
    device: Union[str, torch.device] = "cpu",
) -> Optional[torch.Tensor]:
    model = load_model_for_inference(path_to_pth, device)
    model = torch.jit.script(model)

    # Parse dataset definition yml file and pull out dataset_paths
    import yaml

    with open(path_to_dataset_defn_file_used_for_training) as f:
        dataset_defn = yaml.safe_load(f)
    dataset_paths = dataset_defn["dataset_paths"]

    # Loop through each subfolder, making an image loader and keeping track of the results and that folder's name
    csv_file = "training_data_introspection_results.csv"
    with open(csv_file, "w") as f:
        f.write("image_path,folder_name,ssaf_result\n")

        for key, path_to_training_data in tqdm(
            dataset_paths.items(), "all dataset paths"
        ):
            # path_to_training_data is a folder containing subfolders with names like '-1', '25', etc.
            # Each subfolder contains images
            # We want to infer on all images in the subfolders, and save the result in a csv
            # where one column is the image path, the second is the name of that image's parent folder as a float, and the third is the SSAF result

            path_to_images = Path(path_to_training_data)
            subfolders = [f for f in path_to_images.iterdir() if f.is_dir()]

            for subfolder in tqdm(subfolders, "processing subfolders"):
                # Get the name of the subfolder as a float
                folder_name = float(subfolder.name)
                # Get the path to the images in the subfolder
                path_to_images = subfolder

                image_loader = ImageLoader.load_image_data(
                    path_to_images,
                    img_center_crop_perc=model.center_crop_perc,
                    device=device,
                )

                # Infer
                arr = torch.zeros(len(image_loader))
                for i, res in enumerate(tqdm(infer(model, image_loader))):
                    arr[i] = res

                # Write to a csv file
                for img_path, res in zip(sorted(path_to_images.glob("*.png")), arr):
                    f.write(f"{img_path},{folder_name},{res.item()}\n")
                f.flush()


if __name__ == "__main__":
    parser = infer_parser()
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no_imgs = args.images is None
    no_zarr = args.zarr is None

    if not args.test_training_data:
        if (no_imgs and no_zarr) or (not no_imgs and not no_zarr):
            print("you must supply a value for only one of --images or --zarr")
            sys.exit(1)

    else:
        predict_training_data(
            path_to_pth=args.pth_path,
            path_to_dataset_defn_file_used_for_training="/home/ilakkiyan.jeyakumar/Documents/lfm-autofocus-dataset-definitions/20240923_hochuen_chips_spirit_perseverance_zenith_combined_with_old_spirit_hochuen_data.yml",
            device=device,
        )

    predict(
        path_to_pth=args.pth_path,
        path_to_images=args.images,
        path_to_zarr=args.zarr,
        calc_allan_dev=args.allan_dev,
        plot=args.plot,
        overlay=args.overlay,
        output=args.output_path,
        print_output=args.print_output,
        device=device,
    )

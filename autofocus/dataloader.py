import os
import yaml
import torch
import tarfile

import numpy as np
import multiprocessing as mp

from torch import nn
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
)

from pathlib import Path
from functools import partial
from typing import List, Dict, Union, Tuple, Optional


def _dict_get_and_cast_to_int(dct: Dict[int, int], idx: int) -> int:
    "some weird picklability / multiprocessing thing with num_workers > 0 did this"
    return int(dct[idx])


class ImageFolderWithLabels(datasets.ImageFolder):
    """ImageFolder with minor modifications to make training/use easier

    Changes:
        - save the idx_to_class in the instance so the target_transform to folder name is quick
        - `sample_from_class` method that quickly gives a sample of a given class of a given size
        - modification of how mem is stored for dataloader num_workers > 0
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.target_transform = partial(_dict_get_and_cast_to_int, self.idx_to_class)

        # self.samples is a list of tuples, so it won't play well w/ dataloader.num_workers > 0
        paths, targets = map(list, zip(*self.samples))
        self.paths = np.array(paths).astype(np.string_)
        self.targets = torch.tensor(targets)

    def find_classes(self, directory: Path) -> Tuple[List[str], Dict[str, int]]:
        "Adapted from torchvision.datasets.folder.py"
        classes = sorted(
            entry.name for entry in os.scandir(directory) if entry.is_dir()
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.paths[index], encoding="utf-8")
        sample = self.loader(img_path)
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target.item())
        return sample, target


def load_dataset_description(
    dataset_description,
) -> Tuple[List[str], Dict[str, float]]:
    with open(dataset_description, "r") as desc:
        yaml_data = yaml.safe_load(desc)

        # either we have image_path and label_path directly defined
        # in our yaml file (describing 1 dataset exactly), or we have
        # a nested dict structure describing each dataset description.
        # see README.md for more detail

        if "dataset_paths" in yaml_data and "dataset_zip_paths" in yaml_data:
            dataset_paths_dict = yaml_data["dataset_paths"]
            dataset_zip_paths_dict = yaml_data["dataset_zip_paths"]

            if dataset_paths_dict.keys() != dataset_zip_paths_dict.keys():
                raise ValueError(
                    "keys of dataset descriptor file 'dataset_paths' and 'dataset_zip_paths' must be the same"
                )

            print("transfering data")
            for k in dataset_zip_paths_dict:
                with tarfile.open(dataset_zip_paths_dict[k]) as tar:
                    tar.extractall(path=dataset_paths_dict[k])

            # hack! i should extract out the "training_data" directory if that exists.
            dataset_paths = [
                Path(d) / "training_data" for d in yaml_data["dataset_paths"].values()
            ]
            print("transferred")
        elif "dataset_paths" in yaml_data:
            dataset_paths = [Path(d) for d in yaml_data["dataset_paths"].values()]
        else:
            raise ValueError("dataset description file missing dataset_paths")

        split_fractions = {
            k: float(v) for k, v in yaml_data["dataset_split_fractions"].items()
        }

        if abs(sum(split_fractions.values()) - 1) > 0.001:
            raise ValueError(
                f"invalid split fractions for dataset: split fractions must add to 1, got {split_fractions} -> {sum(split_fractions.values())}"
            )

        check_dataset_paths(dataset_paths)
        return dataset_paths, split_fractions


def check_dataset_paths(dataset_paths: List[Path]):
    for dataset_desc in dataset_paths:
        if not (dataset_desc.is_dir()):
            raise FileNotFoundError("dataset not found")


def read_grayscale(img_path):
    try:
        return read_image(str(img_path), ImageReadMode.GRAY)
    except RuntimeError as e:
        raise RuntimeError(f"file {img_path} threw: {e}")


def is_valid_file(path: str) -> bool:
    return not Path(path).name.startswith(".")


def get_datasets(
    dataset_description_file: str,
    batch_size: int,
    img_size: Tuple[int, int] = (300, 400),
    split_fractions_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Subset[ConcatDataset[ImageFolderWithLabels]]]:
    (
        dataset_paths,
        split_fractions,
    ) = load_dataset_description(dataset_description_file)

    transforms = Resize([300, 400], antialias=True)

    full_dataset: ConcatDataset[ImageFolderWithLabels] = ConcatDataset(
        ImageFolderWithLabels(
            root=dataset_desc,
            transform=transforms,
            loader=read_grayscale,
            is_valid_file=is_valid_file,
        )
        for dataset_desc in dataset_paths
    )

    if split_fractions_override is not None:
        split_fractions = split_fractions_override

    split_keys = list(split_fractions.keys())
    if len(split_keys) > 1:
        dataset_sizes = {
            designation: int(split_fractions[designation] * len(full_dataset))
            for designation in split_keys[:-1]
        }
        test_dataset_size = {
            split_keys[-1]: len(full_dataset) - sum(dataset_sizes.values())
        }
        split_sizes = {**dataset_sizes, **test_dataset_size}
    else:
        split_sizes = {split_keys[0]: len(full_dataset)}

    assert all([sz > 0 for sz in split_sizes.values()]) and sum(
        split_sizes.values()
    ) == len(
        full_dataset
    ), f"could not create valid dataset split sizes: {split_sizes}, full dataset size is {len(full_dataset)}"

    # YUCK! Want a map from the dataset designation to the set itself, but "random_split" takes a list
    # of lengths of dataset. So we do this verbose rigamarol.
    return dict(
        zip(
            split_keys,
            random_split(
                full_dataset,
                [split_sizes[split_key] for split_key in split_keys],
                generator=torch.Generator().manual_seed(101010),
            ),
        )
    )


def collate_batch(batch, transforms: Optional[nn.Module] = None):
    inputs, labels = zip(*batch)
    batched_inputs = torch.stack(inputs)
    batched_labels = torch.tensor(labels)
    if transforms is not None:
        return transforms(batched_inputs), batched_labels
    return batched_inputs, batched_labels


def get_dataloader(
    dataset_description_file: str,
    batch_size: int,
    img_size: Tuple[int, int] = (300, 400),
    device: Union[str, torch.device] = "cpu",
    split_fractions_override: Optional[Dict[str, float]] = None,
    num_workers: Optional[int] = None,
    no_augmentation_split_fraction_name: str = "train",
):
    split_datasets = get_datasets(
        dataset_description_file,
        batch_size,
        img_size=img_size,
        split_fractions_override=split_fractions_override,
    )

    d = dict()
    num_workers = (
        num_workers if num_workers is not None else min(max(mp.cpu_count() - 1, 4), 16)
    )
    for designation, dataset in split_datasets.items():
        augmentations = (
            Compose(
                [
                    RandomHorizontalFlip(0.5),
                    RandomVerticalFlip(0.5),
                    ColorJitter(brightness=(0.95, 1.05)),
                ]
            )
            if designation == no_augmentation_split_fraction_name
            else None
        )
        d[designation] = DataLoader(
            dataset,
            shuffle=True,
            drop_last=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0,
            generator=torch.Generator().manual_seed(101010),
            collate_fn=partial(collate_batch, transforms=augmentations),
        )
    return d

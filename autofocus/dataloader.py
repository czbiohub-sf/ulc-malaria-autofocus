import yaml
import torch

import numpy as np
import multiprocessing as mp

from yogo.data.utils import read_image_robust

from torch import nn
from torchvision import datasets
from torch.utils.data import ConcatDataset, DataLoader, random_split, Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    ColorJitter,
    RandomErasing,
)

from pathlib import Path
from functools import partial
from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional


def _dict_get_and_cast_to_int(dct: Dict[int, int], idx: int) -> torch.Tensor:
    "some weird picklability / multiprocessing thing with num_workers > 0 did this"
    return torch.tensor(int(dct[idx]))


class ImageFolderWithLabels(datasets.ImageFolder):
    """ImageFolder with minor modifications to make training/use easier

    Changes:
        - save the idx_to_class in the instance so the target_transform to folder name is quick
        - `sample_from_class` method that quickly gives a sample of a given class of a given size
        - modification of how mem is stored for dataloader num_workers > 0
    """

    def __init__(self, *args, **kwargs):
        self.valid_classes = list(range(-20, 21))

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
            entry.name
            for entry in directory.iterdir()
            if entry.is_dir() and self.in_range(entry.name)
        )
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def in_range(self, class_name: str) -> bool:
        if self.valid_classes is None:
            return True

        try:
            return int(class_name) in self.valid_classes
        except ValueError:
            return False

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = str(self.paths[index], encoding="utf-8")
        sample = self.loader(img_path)
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target.item())
        return sample, target


class InvalidDatasetDescriptionFile(Exception):
    ...


@dataclass
class DatasetDescription:
    split_fractions: Dict[str, float]
    dataset_paths: List[Path]
    test_dataset_paths: Optional[List[Path]]

    def __iter__(self):
        return iter(
            (self.split_fractions, self.dataset_paths, self.test_dataset_paths,)
        )


def load_dataset_description(dataset_description: str) -> DatasetDescription:
    """Loads and validates dataset description file"""
    required_keys = [
        "dataset_split_fractions",
        "dataset_paths",
    ]
    with open(dataset_description, "r") as desc:
        yaml_data = yaml.safe_load(desc)

        for k in required_keys:
            if k not in yaml_data:
                raise InvalidDatasetDescriptionFile(
                    f"{k} is required in dataset description files, but was "
                    f"found missing for {dataset_description}"
                )

        split_fractions = {
            k: float(v) for k, v in yaml_data["dataset_split_fractions"].items()
        }
        dataset_paths = [Path(d) for d in yaml_data["dataset_paths"].values()]
        check_dataset_paths(dataset_paths)

        if "test_paths" in yaml_data:
            test_dataset_paths = [Path(d) for d in yaml_data["test_paths"].values()]
            check_dataset_paths(test_dataset_paths)

            # when we have 'test_paths', all the data from dataset_paths
            # will be used for training, so we should only have 'test' and
            # 'val' in dataset_split_fractions.
            if "val" not in split_fractions or "train" not in split_fractions:
                raise InvalidDatasetDescriptionFile(
                    "'val' and 'train' are required keys for dataset_split_fractions"
                )
        else:
            test_dataset_paths = None
            if any(k not in split_fractions for k in ("test", "train", "val")):
                raise InvalidDatasetDescriptionFile(
                    "'train', 'val', and 'test' are required keys for dataset_split_fractions - missing at least one. "
                    f"split fractions was {split_fractions}"
                )

        if not sum(split_fractions.values()) == 1:
            raise InvalidDatasetDescriptionFile(
                "invalid split fractions for dataset: split fractions must add to 1, "
                f"got {split_fractions}"
            )

    return DatasetDescription(split_fractions, dataset_paths, test_dataset_paths)


def check_dataset_paths(dataset_paths: List[Path]):
    for dataset_desc in dataset_paths:
        if not (dataset_desc.is_dir()):
            raise FileNotFoundError(f"dataset not found {dataset_desc}")


def read_grayscale(img_path):
    return read_image_robust(img_path, rgb=False)


def get_datasets(
    dataset_description_file: str,
    batch_size: int,
    img_size: Tuple[int, int] = (300, 400),
    split_fractions_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Dataset]:
    dd = load_dataset_description(dataset_description_file)

    split_fractions = dd.split_fractions
    dataset_paths = dd.dataset_paths
    test_dataset_paths = dd.test_dataset_paths

    transforms = Resize(img_size, antialias=True)

    if split_fractions_override is not None:
        split_fractions = split_fractions_override

    full_dataset: ConcatDataset[ImageFolderWithLabels] = ConcatDataset(
        ImageFolderWithLabels(
            root=dataset_desc, transform=transforms, loader=read_grayscale,
        )
        for dataset_desc in dataset_paths
    )

    if test_dataset_paths is not None:
        test_dataset: ConcatDataset[ImageFolderWithLabels] = ConcatDataset(
            ImageFolderWithLabels(
                root=dataset_desc, transform=transforms, loader=read_grayscale,
            )
            for dataset_desc in test_dataset_paths
        )
        assert "train" in split_fractions and "val" in split_fractions
        return {**split_dataset(full_dataset, split_fractions), "test": test_dataset}

    return split_dataset(full_dataset, split_fractions)


def split_dataset(
    dataset: Dataset, split_fractions: Dict[str, float]
) -> Dict[str, Dataset]:
    if not hasattr(dataset, "__len__"):
        raise ValueError(
            f"dataset {dataset} must have a length (specifically, `__len__` must be defined)"
        )

    if len(split_fractions) == 0:
        raise ValueError("must have at least one value for the split!")
    elif len(split_fractions) == 1:
        if not next(iter(split_fractions.values())) == 1:
            raise ValueError(
                "when split_fractions has length 1, it must have a value of 1"
            )
        keys = list(split_fractions)
        return {keys.pop(): dataset}

    keys = list(split_fractions)

    # very annoying type hint here - `Dataset` doesn't necessarily have `__len__`,
    # so we manually check it. But I am not sure that you can cast to Sizedj so mypy complains
    dataset_sizes = {
        k: round(split_fractions[k] * len(dataset)) for k in keys[:-1]  # type: ignore
    }
    final_dataset_size = {keys[-1]: len(dataset) - sum(dataset_sizes.values())}  # type: ignore
    split_sizes = {**dataset_sizes, **final_dataset_size}

    all_sizes_are_gt_0 = all([sz >= 0 for sz in split_sizes.values()])
    split_sizes_eq_dataset_size = sum(split_sizes.values()) == len(dataset)  # type: ignore
    if not (all_sizes_are_gt_0 and split_sizes_eq_dataset_size):
        raise ValueError(
            f"could not create valid dataset split sizes: {split_sizes}, "
            f"full dataset size is {len(dataset)}"  # type: ignore
        )

    # YUCK! Want a map from the dataset designation to teh set itself, but "random_split" takes a list
    # of lengths of dataset. So we do this verbose rigamarol.
    return dict(
        zip(
            keys,
            random_split(
                dataset,
                [split_sizes[k] for k in keys],
                generator=torch.Generator().manual_seed(111111),
            ),
        )
    )


def collate_batch(batch, transforms: Optional[nn.Module] = None):
    inputs, labels = zip(*[pair for pair in batch if pair is not None])
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
    augmentation_split_fraction_name: str = "train",
    color_jitter: bool = False,
    random_erasing: bool = False,
):
    split_datasets = get_datasets(
        dataset_description_file,
        batch_size,
        img_size=img_size,
        split_fractions_override=split_fractions_override,
    )

    d = dict()
    num_workers = (
        num_workers if num_workers is not None else min(max(mp.cpu_count() - 1, 4), 32)
    )
    for designation, dataset in split_datasets.items():
        ggggs = [
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
        ]
        if color_jitter:
            ggggs.append(ColorJitter(brightness=(0.90, 1.10)),)
        if random_erasing:
            ggggs.append(
                RandomErasing(
                    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False,
                )
            )
        augmentations = (
            Compose(ggggs) if designation == augmentation_split_fraction_name else None
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

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

from typing import List


class ImageFolderWithLabels(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        def idx_to_class(idx):
            return int(self.idx_to_class[idx])

        self.target_transform = idx_to_class

        # for `sample_from_class`
        self.class_to_samples = dict()
        for el, label in self.imgs:
            el_class = idx_to_class(label)
            if el_class in self.class_to_samples:
                self.class_to_samples[el_class].append(el)
            else:
                self.class_to_samples[el_class] = [el]

    def sample_from_class(self, clss, count):
        # TODO: probably a better way to do this
        sample_set = self.class_to_samples[clss]
        idxs = torch.randint(len(sample_set), [count, 1])
        samples = []
        for idx in idxs:
            T = self.transform(self.loader(sample_set[idx]))
            T = torch.unsqueeze(T, 0)
            samples.append(T)
        return torch.cat(samples)


def get_dataset(root_dir: str, batch_size: int, split_percentages: List[float] = [1]):
    assert (
        sum(split_percentages) == 1
    ), f"split_percentages must add to 1 - got {split_percentages}"

    transforms = Compose(
        [Resize([150, 200]), RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5)]
    )
    full_dataset = ImageFolderWithLabels(
        root=root_dir, transform=transforms, loader=read_image
    )

    first_split_sizes = [
        int(rel_size * len(full_dataset)) for rel_size in split_percentages[:-1]
    ]
    final_split_size = [len(full_dataset) - sum(first_split_sizes)]
    split_sizes = first_split_sizes + final_split_size

    assert all([sz > 0 for sz in split_sizes]) and sum(split_sizes) == len(
        full_dataset
    ), f"could not create valid dataset split sizes: {split_sizes}, full dataset size is {len(full_dataset)}"

    return random_split(full_dataset, split_sizes)


def get_dataloader(
    root_dir: str, batch_size: int, split_percentages: List[float] = [1]
):
    split_datasets = get_dataset(root_dir, batch_size, split_percentages)
    return [
        DataLoader(split_dataset, batch_size=batch_size, shuffle=True)
        for split_dataset in split_datasets
    ]

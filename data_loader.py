#! /usr/bin/env python3

import torch
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict

from torchvision.io import read_image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, random_split


DATA_DIRS = "/Volumes/flexo/MicroscopyData/Bioengineering/LFM Scope/ssaf_trainingdata/2022-06-10-1056/training_data"
ROOT_DATA_DIR = Path(DATA_DIRS)


class FocusDataSet(Dataset):
    """
    PNG image data, 800 x 600, 8-bit grayscale, non-interlaced
    """
    def __init__(self, folder_paths: str, device: str = "cpu"):
        self.data_locs = sorted(
            [
                (int(image.parent.name), image)
                for data_dir in Path(folder_paths).iterdir()
                for image in data_dir.iterdir()
            ],
            key=lambda e: e[0],
        )

    def __len__(self):
        return len(self.data_locs)

    def __getitem__(self, idx):
        label, image_path = self.data_locs[idx]
        return read_image(str(image_path)), label

full_dataset = FocusDataSet(DATA_DIRS)
test_size = int(0.3 * len(full_dataset))
train_size = len(full_dataset) - test_size
testing_dataset, training_dataset = random_split(full_dataset, [test_size, train_size])

test_dataloader = DataLoader(training_dataset, batch_size=128, shuffle=True)
train_dataloader = DataLoader(training_dataset, batch_size=128, shuffle=True)


if __name__ == "__main__":
    f = FocusDataSet(DATA_DIRS)
    print(f.data_locs)

import pickle as pkl

import torch
import torch.nn as nn
import numpy as np

from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision.transforms import Resize

from nn_analysis import load_model_for_inference
from dataloader import ImageFolderWithLabels, get_dataset, get_dataloader

from typing import List
from collections import namedtuple


# custom dataset

class ImagesAndPaths(ImageFolderWithLabels):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target), path

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = load_model_for_inference("trained_models/efficient-donkey-final.pth", dev=dev)

L2 = nn.MSELoss().to(dev)

full_dataset = ImagesAndPaths(
    root = "training_data",
    transform = Resize([150, 200]),
    loader = read_image
)

dataloader = DataLoader(full_dataset, batch_size=1)

import time
t0 = time.perf_counter()

Res = namedtuple("Res", ["path", "pred", "clss", "loss"])
reses = []
for data in dataloader:
    (img, clss), path = data
    img.to(dev)
    clss.to(dev)

    with torch.no_grad():
        outputs = net(img).reshape(-1)
        loss = L2(outputs, clss.float())
        reses.append(Res(path[0], outputs.item(), clss.item(), loss.item()))

time.perf_counter() - t0

with open("res.pkl", "wb") as f:
    pkl.dump(reses, f)

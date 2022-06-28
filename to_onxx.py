#! /usr/bin/env python3

import sys

import torch
import torchvision

from model import AutoFocus


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: ./to_onxx.py <your_file.pth>")
        sys.exit(1)

    pth_filename = sys.argv[1]

    net = AutoFocus()
    net.eval()

    model_save = torch.load(pth_filename, map_location=torch.device("cpu"))
    net.load_state_dict(model_save["model_state_dict"])

    dummy_input = torch.randn(1, 1, 150, 200)

    torch.onnx.export(net, dummy_input, "autofocus.onnx", verbose=True)

#! /usr/bin/env python3

import sys

import onnx
import onnxruntime

import torch
import torchvision

import numpy as np

from model import AutoFocus


"""
Learning from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: ./to_onxx.py <your_file.pth>")
        sys.exit(1)

    pth_filename = sys.argv[1]
    dev = torch.device("cpu")

    net = AutoFocus()
    net.eval()
    net.half()

    # TODO CPU vs GPU vs whatever else?
    model_save = torch.load(pth_filename, map_location=dev)
    net.load_state_dict(model_save["model_state_dict"])
    net.to(dev)

    dummy_input = torch.randn(
        1, 1, 150, 200, requires_grad=False, device=dev, dtype=torch.float16
    )
    torch_out = net(dummy_input)

    torch.onnx.export(
        net,
        dummy_input,
        "autofocus.onnx",
    )

    # Load the ONNX model
    model = onnx.load("autofocus.onnx")

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Compare model output from pure torch and onnx
    ort_session = onnxruntime.InferenceSession(model)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(
        to_numpy(torch_out),
        ort_outs[0],
        rtol=1e-3,
        atol=1e-5,
        err_msg="onnx and pytorch outputs are far apart",
    )

    # Print a human readable representation of the graph
    print("Export successful")
    print(onnx.helper.printable_graph(model.graph))

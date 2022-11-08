import argparse

from typing import Union


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def infer_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="infer results over some dataset")

    parser.add_argument(
        "pth_path", type=str, help="path to .pth file defining the model"
    )
    parser.add_argument("--images", type=str, help="path to image or images")
    parser.add_argument("--zarr", type=str, help="path to zarr store")

import sys
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
    parser.add_argument(
        "--output",
        type=str,
        help="place to write data to",
        default=None,
    )
    parser.add_argument(
        "--allan-dev",
        help="calculate allan deviation",
        action=boolean_action,
        default=False,
    )

    return parser


def train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="commence a training run")

    parser.add_argument(
        "dataset_descriptor_file",
        type=str,
        help="path to yml dataset descriptor file",
    )
    parser.add_argument(
        "--note",
        type=str,
        nargs="?",
        help="note for the run (e.g. 'run on a TI-82')",
        default="",
    )
    parser.add_argument(
        "--group",
        type=str,
        nargs="?",
        help="group that the run belongs to (e.g. 'mAP test')",
    )
    return parser

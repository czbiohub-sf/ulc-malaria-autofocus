import argparse

from pathlib import Path


try:
    boolean_action = argparse.BooleanOptionalAction  # type: ignore
except AttributeError:
    boolean_action = "store_true"  # type: ignore


def infer_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="infer results over some dataset")

    parser.add_argument(
        "pth_path", type=Path, help="path to .pth file defining the model"
    )

    data_option = parser.add_mutually_exclusive_group(required=True)
    data_option.add_argument("--images", type=Path, help="path to image or images")
    data_option.add_argument("--zarr", type=Path, help="path to zarr store")

    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "output file name - if --plot is present, this will be the "
            "image file name. Else, this will be the text file name"
        ),
        default=None,
    )
    parser.add_argument("--print-output", action=boolean_action, default=False)

    result_options = parser.add_mutually_exclusive_group(required=False)
    result_options.add_argument(
        "--allan-dev",
        help="calculate allan deviation",
        action=boolean_action,
        default=False,
    )
    result_options.add_argument(
        "--plot",
        help="plot results",
        action=boolean_action,
        default=False,
    )

    return parser


def train_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(description="commence a training run")

    parser.add_argument(
        "dataset_descriptor_file",
        type=Path,
        help="path to yml dataset descriptor file",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
        default=3e-4,
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(300, 400),
        help="resize image to these dimensions. e.g. '-r 300 400' to resize to width=300, height=400 (default 300 400)",
    )
    parser.add_argument(
        "--note",
        type=str,
        help="note for the run (e.g. 'was run on a TI-82')",
        default="",
    )
    parser.add_argument(
        "--group",
        type=str,
        help="group that the run belongs to (e.g. 'mAP test')",
    )
    parser.add_argument(
        "--allow-tf32",
        action=boolean_action,
        default=False,
    )
    parser.add_argument(
        "--color-jitter",
        action=boolean_action,
        default=False,
    )
    parser.add_argument(
        "--random-erasing",
        action=boolean_action,
        default=False,
    )

    return parser

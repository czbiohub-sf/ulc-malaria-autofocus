import glob
import os
import torch
import sys

from infer import load_model_for_inference, infer, ImageLoader
from time import perf_counter
from tqdm import tqdm

def ssaf_rerun(data_dir, model_dir, output_dir):

    file_format = f'{data_dir}/*-*-*-*/*-*-*-*_/*.zip'

    a = perf_counter()
    files = glob.glob(file_format)
    print(files)
    print(f"Number of files: {len(files)}")
    b = perf_counter()
    print(f"Get files: {b-a} s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = load_model_for_inference(model_dir, device)

    for file in files:
        image_loader = (ImageLoader.load_zarr_data(file))
        basename = os.path.basename(file)

        print(f"Started processing {basename}")

        c = perf_counter()
        with open(f"{output_dir}/{basename.removesuffix('.zip')}__ssaf.txt", 'w') as file:
            for res in tqdm(infer(model, image_loader)):
                file.write(f"{res}\n")
        d = perf_counter()
        print(f"Finished writing {basename} SSAF data in {d-c} s")


if __name__ == "__main__":
    try:
        scope_folder = sys.argv[1]
        model_file = sys.argv[2]
        output_folder = sys.argv[3]
    except IndexError:
        raise Exception(
            "Expected format 'python3 ssaf_rerun_all.py <path to scope folder> <path to model .pth file> <path to output folder>'"
        )

    ssaf_rerun(scope_folder, model_file, output_folder)

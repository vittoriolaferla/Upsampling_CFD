# downsample_with_vars.py
# A script to downsample images by factors of 2 and 4 using nearest-neighbor decimation.
# Input and output paths are defined as variables, and outputs are organized into subfolders.

import os
from PIL import Image
import numpy as np

# === User-defined paths ===
# Path to an image file or a directory of images
INPUT_PATH = "/home/vittorio/Scrivania/ETH/Upsampling/UpsamplingCFD/datasets/GoundTruthUnifom"
# Directory where subfolders (original, ds2x, ds4x) will be created
OUTPUT_DIR = "/home/vittorio/Scrivania/ETH/Upsampling/UpsamplingCFD/datasets/Dataset_Outdoor_Flow"
# Reduction factors and corresponding subfolder names
FACTORS = {2: "2_Downsample", 4: "4_Downsample"}
# Subfolder for original copies
ORIGINAL_SUBFOLDER = "original"

# Supported image extensions
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"}

def is_image_file(fname):
    return os.path.splitext(fname.lower())[1] in SUPPORTED_EXTS


def downsample_image_nn(img: Image.Image, factor: int) -> Image.Image:
    """
    Nearest-neighbor decimation downsample:
    take every factor-th pixel in both dimensions.
    This preserves the exact original pixel values.
    """
    arr = np.array(img)
    down = arr[::factor, ::factor]
    return Image.fromarray(down)


def process_path(input_path: str, output_dir: str):
    # Build list of image files
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, fn) for fn in os.listdir(input_path) if is_image_file(fn)]
    else:
        if not is_image_file(input_path):
            raise ValueError(f"Input file {input_path} is not a supported image type.")
        files = [input_path]

    # Create subfolders
    orig_dir = os.path.join(output_dir, ORIGINAL_SUBFOLDER)
    os.makedirs(orig_dir, exist_ok=True)
    factor_dirs = {}
    for factor, sub in FACTORS.items():
        path = os.path.join(output_dir, sub)
        os.makedirs(path, exist_ok=True)
        factor_dirs[factor] = path

    for img_path in files:
        img = Image.open(img_path)
        base, ext = os.path.splitext(os.path.basename(img_path))

        # Save original copy
        orig_out = os.path.join(orig_dir, f"{base}{ext}")
        img.save(orig_out)
        print(f"Saved original: {orig_out}")

        # Save downsampled versions
        for factor, sub_dir in factor_dirs.items():
            out_img = downsample_image_nn(img, factor)
            out_name = f"{base}{ext}"
            out_path = os.path.join(sub_dir, out_name)
            out_img.save(out_path)
            print(f"Saved downsample×{factor}: {out_path}")


if __name__ == "__main__":
    # Execute processing with user-defined paths
    process_path(INPUT_PATH, OUTPUT_DIR)

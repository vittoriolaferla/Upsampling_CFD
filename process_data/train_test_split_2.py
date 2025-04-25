# train_test_split.py
# A script to split a dataset of high- and low-resolution image pairs into train/test folders.

import os
import shutil
import random

# === User-defined variables ===
# Paths to separate folders for high- and low-resolution images
HIGH_RES_DIR = "/home/vittorio/Scrivania/ETH/Upsampling/UpsamplingCFD/datasets/Dataset_Outdoor_Flow/original"
LOW_RES_DIR = "/home/vittorio/Scrivania/ETH/Upsampling/UpsamplingCFD/datasets/Dataset_Outdoor_Flow/2_Downsample"
# Path to output split folder
OUTPUT_SPLIT_DIR = "/home/vittorio/Scrivania/ETH/Upsampling/UpsamplingCFD/datasets/Split_Outdoor_Flow"
# Train/test split ratio (e.g., 0.8 means 80% train, 20% test)
TRAIN_RATIO = 0.8
# Supported image extensions
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"}


def is_image_file(fname):
    ext = os.path.splitext(fname.lower())[1]
    return ext in SUPPORTED_EXTS


def gather_image_pairs(high_dir, low_dir):
    """
    Collect matching high- and low-resolution image file paths from two separate directories.
    Returns a list of (high_path, low_path) tuples.
    """
    high_files = [f for f in os.listdir(high_dir) if is_image_file(f)]
    pairs = []
    for fname in high_files:
        high_path = os.path.join(high_dir, fname)
        low_path = os.path.join(low_dir, fname)
        if os.path.exists(low_path):
            pairs.append((high_path, low_path))
        else:
            print(f"Warning: low-res file missing for {fname}")
    return pairs


def split_pairs(pairs, train_ratio):
    random.shuffle(pairs)
    split_idx = int(len(pairs) * train_ratio)
    return pairs[:split_idx], pairs[split_idx:]


def make_split_dirs(output_dir):
    """
    Create train/high_res, train/low_res, test/high_res, test/low_res folders.
    """
    for split in ("train", "test"):
        for res in ("high_res", "low_res"):
            dir_path = os.path.join(output_dir, split, res)
            os.makedirs(dir_path, exist_ok=True)


def copy_pairs(pairs, split, output_dir):
    """
    Copy each (high, low) pair into the appropriate split folder.
    """
    for high_path, low_path in pairs:
        fname = os.path.basename(high_path)
        dest_high = os.path.join(output_dir, split, "high_res", fname)
        dest_low = os.path.join(output_dir, split, "low_res", fname)
        shutil.copy2(high_path, dest_high)
        shutil.copy2(low_path, dest_low)


def main():
    # Gather all pairs from separate high/low directories
    pairs = gather_image_pairs(HIGH_RES_DIR, LOW_RES_DIR)
    print(f"Found {len(pairs)} image pairs.")

    # Split into train/test
    train_pairs, test_pairs = split_pairs(pairs, TRAIN_RATIO)
    print(f"Assigning {len(train_pairs)} pairs to train, {len(test_pairs)} to test.")

    # Create directory structure
    make_split_dirs(OUTPUT_SPLIT_DIR)

    # Copy files
    copy_pairs(train_pairs, "train", OUTPUT_SPLIT_DIR)
    copy_pairs(test_pairs, "test", OUTPUT_SPLIT_DIR)

    print("Dataset split complete.")

if __name__ == "__main__":
    main()

import random
import numpy as np
import torch.utils.data as data
import utils.utils_SwinIR.utils_image as util
import pandas as pd
import torch

class DatasetSR(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # Also retrieves corresponding CSV files if 'dataroot_CSV' is provided.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSR, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['lq_patchsize'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf
        self.csv_delimiter = opt.get('csv_delimiter', ',')
        self.csv_coord_cols = opt.get('csv_coord_cols', None)
        self.phase = opt['phase']  # Add phase attribute

        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])
        if opt['dataroot_CSV']:
            self.paths_CSV = util._get_paths_from_csvs(opt['dataroot_CSV']) if opt.get('dataroot_CSV') else None
        else:
            self.paths_CSV = None

        assert self.paths_H, 'Error: H path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):
        L_path = None
        csv_path = None
        csv_data_cropped = None

        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)
        img_H = util.modcrop(img_H, self.sf)

        if self.paths_L:
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)
        else:
            img_L = util.imresize_np(img_H, 1 / self.sf, True)

        csv_data_full = None
        if self.paths_CSV:
            csv_path = self.paths_CSV[index]
            if csv_path:
                try:
                    csv_data_full = pd.read_csv(csv_path, header=None, delimiter=self.csv_delimiter)
                except Exception as e:
                    print(f"Error reading CSV file: {csv_path} - {e}")

        # ------------------------------------------------------------------
# 2.  Paired random crop (TRAIN ONLY) ------------------------------
# ------------------------------------------------------------------
        if self.phase == 'train':
            # --- compute equivalent L-patch size and pick a random position
            lq_patch  = self.patch_size // self.sf       # e.g. 256 → 64 for ×4
            h_lq, w_lq = img_L.shape[:2]

            if h_lq < lq_patch or w_lq < lq_patch:
                raise ValueError(
                    f'LQ patch ({h_lq}×{w_lq}) is smaller than requested size '
                    f'({lq_patch}×{lq_patch}). File: {H_path}'
                )

            top  = random.randint(0, h_lq - lq_patch)
            left = random.randint(0, w_lq - lq_patch)

            # --- crop LQ ---------------------------------------------------
            img_L = img_L[top : top + lq_patch,
                        left: left + lq_patch, :]

            # --- crop the matching HQ region ------------------------------
            top_gt, left_gt = top * self.sf, left * self.sf
            img_H = img_H[top_gt : top_gt + self.patch_size,
                        left_gt: left_gt + self.patch_size, :]

            # --- crop CSV (if any) with the *HQ* coordinates --------------
            if csv_data_full is not None:
                try:
                    csv_slice = csv_data_full.iloc[
                        top_gt : top_gt + self.patch_size,
                        left_gt: left_gt + self.patch_size
                    ]
                    csv_data_cropped = torch.tensor(csv_slice.values,
                                                    dtype=torch.float32)
                except Exception as e:
                    print(f'[CSV-crop] {csv_path} – {e}')
                    csv_data_cropped = None

        elif self.phase == 'test': # return the full csv data
            if csv_data_full is not None:
                try:
                    csv_data_cropped = torch.tensor(csv_data_full.values, dtype=torch.float32)
                except Exception as e:
                    print(f"Error converting full CSV data to tensor: {e}")
                    csv_data_cropped = None
            else:
                csv_data_cropped = None

        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path
        if csv_data_cropped is None:
            csv_data_cropped = torch.empty(0, dtype=torch.float32)
            csv_path = torch.empty(0, dtype=torch.float32)

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path, 'CSV': csv_data_cropped, 'CSV_path': csv_path}

    def __len__(self):
        return len(self.paths_H)
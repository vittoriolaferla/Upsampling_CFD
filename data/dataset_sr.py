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
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
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

        if self.phase == 'train':  # Only patch during training
            H_cropped_h, H_cropped_w, _ = img_L.shape
            rnd_h = random.randint(0, max(0, H_cropped_h - self.L_size))
            rnd_w = random.randint(0, max(0, H_cropped_w - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]
            img_H = img_H[int(rnd_h * self.sf):int(rnd_h * self.sf) + self.patch_size,
                          int(rnd_w * self.sf):int(rnd_w * self.sf) + self.patch_size, :]

            if csv_data_full is not None:
                csv_start_row = int(rnd_h * self.sf)
                csv_end_row = int(rnd_h * self.sf) + self.patch_size
                csv_start_col = int(rnd_w * self.sf)
                csv_end_col = int(rnd_w * self.sf) + self.patch_size
                try:
                    csv_data_cropped_pd = csv_data_full.iloc[csv_start_row:csv_end_row, csv_start_col:csv_end_col].copy()
                    csv_data_cropped = torch.tensor(csv_data_cropped_pd.values, dtype=torch.float32)
                except IndexError:
                    print(f"Warning: Cropped CSV indices out of bounds for {csv_path}")
                    csv_data_cropped = None
                except Exception as e:
                    print(f"Error converting CSV data to tensor: {e}")
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
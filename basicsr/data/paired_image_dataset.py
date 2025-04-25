from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import (
    paired_paths_from_folder,
    paired_paths_from_lmdb,
    paired_paths_from_meta_info_file,
)
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import (
    FileClient,
    imfrombytes,
    img2tensor,
    scandir,
)
from basicsr.utils.matlab_functions import bgr2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import os.path as osp
import numpy as np
import pandas as pd
import torch

def paths_from_folder(folder):
    """Generate paths from folder."""
    paths = list(scandir(folder, full_path=True))
    return paths

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from multiple folders."""
    assert len(folders) == len(keys), f"The number of folders ({len(folders)}) and keys ({len(keys)}) must match."
    if 'gt' not in keys:
        raise ValueError("Keys must include 'gt' for the ground truth folder.")

    gt_idx = keys.index('gt')
    gt_folder = folders[gt_idx]
    gt_files = list(scandir(gt_folder, full_path=False))

    paths = []
    for gt_file in gt_files:
        basename, gt_ext = osp.splitext(osp.basename(gt_file))
        path_dict = {}
        for i, key in enumerate(keys):
            folder = folders[i]
            if key == 'gt':
                file_name = gt_file
            else:
                file_ext = '.csv' if key == 'csv' else gt_ext
                file_name = filename_tmpl.format(basename) + file_ext
            file_path = osp.join(folder, file_name)
            path_dict[f'{key}_path'] = file_path
        paths.append(path_dict)
    return paths

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration with corresponding CSV and Geometry data."""

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.csv_folder = opt.get('dataroot_csv', None)
        self.geometry_folder = opt.get('dataroot_geometry', None) #Added geometry_folder
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        self.csv_coord_cols = opt.get('csv_coord_cols', None)
        self.csv_delimiter = opt.get('csv_delimiter', ',')

        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError("LMDB backend with CSV data is not yet implemented.")
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self._load_paths_from_meta_file()
        else:
            self._load_paths_from_folders()

    def _load_paths_from_meta_file(self):
        self.paths = paired_paths_from_meta_info_file(
            [self.lq_folder, self.gt_folder],
            ['lq', 'gt'],
            self.opt['meta_info_file'],
            self.filename_tmpl,
        )
        if self.csv_folder:
            csv_paths = paths_from_folder(self.csv_folder)
            csv_paths = sorted(csv_paths)
            assert len(csv_paths) == len(self.paths), 'Mismatch between number of CSV files and images'
            for i, path_dict in enumerate(self.paths):
                path_dict['csv_path'] = csv_paths[i]
        if self.geometry_folder: #Added geometry folder
            geometry_paths = paths_from_folder(self.geometry_folder)
            geometry_paths = sorted(geometry_paths)
            assert len(geometry_paths) == len(self.paths), 'Mismatch between number of geometry files and images'
            for i, path_dict in enumerate(self.paths):
                path_dict['geometry_path'] = geometry_paths[i]

    def _load_paths_from_folders(self):
        if self.csv_folder and self.geometry_folder:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.csv_folder, self.geometry_folder],
                ['lq', 'gt', 'csv', 'geometry'],
                self.filename_tmpl,
            )
        elif self.csv_folder:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.csv_folder], ['lq', 'gt', 'csv'], self.filename_tmpl
            )
        elif self.geometry_folder:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.geometry_folder], ['lq', 'gt', 'geometry'], self.filename_tmpl
            )
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl
            )

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt
            )

        scale = self.opt['scale']

        # Load gt and lq images
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Load CSV data
        csv_data = None
        csv_path = self.paths[index].get('csv_path')
        if self.csv_folder is not None and csv_path:
            try:
                csv_data = pd.read_csv(csv_path, header=None, delimiter=self.csv_delimiter) # Read without header
                if self.csv_coord_cols is not None and len(self.csv_coord_cols) == 2:
                    try:
                        x_col_index = int(self.csv_coord_cols[0])
                        y_col_index = int(self.csv_coord_cols[1])
                    except ValueError:
                        raise ValueError(f"Invalid CSV coordinate column indices: '{self.csv_coord_cols}'. Must be integers.")
                    except IndexError:
                        raise ValueError(f"CSV file {csv_path} does not have enough columns based on provided indices: '{self.csv_coord_cols}'.")
            except Exception as e:
                print(f"Error reading CSV file: {csv_path} - {e}")

        #Load geometry data
        geometry_data = None
        geometry_path = self.paths[index].get('geometry_path')
        if self.geometry_folder is not None and geometry_path:
            try:
                geometry_data = imfrombytes(self.file_client.get(geometry_path, 'geometry'), float32=True)
            except Exception as e:
                print(f"Error reading geometry file: {geometry_path} - {e}")

        # Color space transform
        if self.opt.get('color', None) == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]

        # Crop unmatched GT images during validation or testing
        if self.opt['phase'] != 'train':
            img_gt = img_gt[
                0 : img_lq.shape[0] * scale, 0 : img_lq.shape[1] * scale, :
            ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # Convert CSV data to tensor if it exists
        if csv_data is not None:
            csv_tensor = torch.tensor(csv_data.values, dtype=torch.float32)
        else:
            csv_tensor = None

        # Convert geometry data to tensor if it exists
        if geometry_data is not None:
            geometry_tensor = torch.tensor(geometry_data, dtype=torch.float32)
            geometry_tensor = geometry_tensor.permute(2,0,1) #h,w,c -> c,h,w
        else:
            geometry_tensor = None

        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'csv': csv_tensor,
            'geometry': geometry_tensor,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'csv_path': csv_path,
            'geometry_path': geometry_path,
        }

    def __len__(self):
        return len(self.paths)
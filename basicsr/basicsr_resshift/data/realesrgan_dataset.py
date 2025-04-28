import cv2
import math
import numpy as np
import random
import time
import torch
from utils.utils_ResShift import util_common

import albumentations

import torch.nn.functional as F
from torch.utils import data as data

from basicsr.basicsr_resshift.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.basicsr_resshift.data.transforms import augment
from basicsr.basicsr_resshift.utils import FileClient, imfrombytes, img2tensor
from basicsr.basicsr_resshift.utils.registry import DATASET_REGISTRY
from basicsr.basicsr_resshift.utils.img_process_util import filter2D
from basicsr.basicsr_resshift.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

import pandas as pd

def readline_txt(txt_file):
    txt_file = [txt_file, ] if isinstance(txt_file, str) else txt_file
    out = []
    for txt_file_current in txt_file:
        with open(txt_file_current, 'r') as ff:
            out.extend([x.strip() for x in ff.readlines()])
    return out

@DATASET_REGISTRY.register(suffix='basicsr')
class RealESRGANDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model with Velocity Data Integration.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_vel (str, optional): Data root path for velocity CSV files.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwargs.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            gt_size (int): Ground truth size.
            crop_pad_size (int): Crop or pad size.
            rescale_gt (bool): Whether to rescale ground truth.
            ... (other degradation related settings)
    """

    def __init__(self, opt, mode='training'):
        super(RealESRGANDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.file_client = None
        self.io_backend_opt = opt['io_backend']


        # ----------------------------- Load Image Paths ----------------------------- #
        self.paths = []
        if 'dir_paths' in opt:
            current_dir = opt['dir_paths']
            current_ext = opt['im_exts']
            self.paths.extend(util_common.scan_files_from_folder(current_dir, current_ext, False))
        
        if 'txt_file_path' in opt:
            for current_txt in opt['txt_file_path']:
                self.paths.extend(readline_txt(current_txt))
        
        if 'length' in opt:
            self.paths = random.sample(self.paths, opt['length'])

        # --------------------------- Load Velocity Paths ---------------------------- #
        self.paths_Vel = []
        if 'dataroot_vel' in opt:
            vel_dir = opt['dataroot_vel']
            vel_ext = opt.get('vel_ext', '.csv')  # Default extension is .csv
            self.paths_Vel = util_common.scan_csv_files_from_folder(vel_dir, False)
        
        # Ensure that velocities are provided if required
        if self.paths_Vel:
            assert len(self.paths_Vel) == len(self.paths), \
                f"Number of velocity files ({len(self.paths_Vel)}) does not match number of images ({len(self.paths)})."
        else:
            print("Warning: No velocity files found. Proceeding without velocity data.")

        # ----------------------------- Blur Settings ------------------------------ #
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range1 = [x for x in range(3, self.blur_kernel_size + 1, 2)] # Adjusted to include blur_kernel_size
        self.kernel_range2 = [x for x in range(3, self.blur_kernel_size2 + 1, 2)] # Adjusted to include blur_kernel_size2
        self.pulse_tensor = torch.zeros(self.blur_kernel_size2, self.blur_kernel_size2).float()
        self.pulse_tensor[self.blur_kernel_size2 // 2, self.blur_kernel_size2 // 2] = 1

        # ------------------------------ Rescaling ------------------------------ #
        self.rescale_gt = opt['rescale_gt']

        # --------------------------- Patch Size Settings ------------------------- #
        self.gt_size = opt.get('gt_size', 256)  # Default ground truth size
        self.crop_pad_size = opt.get('crop_pad_size', 400)  # Default crop/pad size

        # ----------------------------- Additional Attributes ----------------------- #
        # Initialize any additional attributes if necessary

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        gt_path = self.paths[index]
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
                img_gt = imfrombytes(img_bytes, float32=True)
                break
            except Exception as e:
                print(f"Error loading {gt_path}: {e}. Retrying...")
                index = random.randint(0, self.__len__() - 1)
                gt_path = self.paths[index]
                time.sleep(1)
            finally:
                retry -= 1
        else:
            raise IOError(f"Failed to load image after retries: {gt_path}")

        # --------------------------------- Load Velocity Data (CSV) --------------------------------- #
        vel_df = None
        vel_path = None
        if self.paths_Vel:
            vel_path = self.paths_Vel[index]
            vel_df = self.load_velocity_csv(vel_path)

        # --------------------------------- Preprocessing --------------------------------- #
        if self.mode == 'testing':
            if not hasattr(self, 'test_aug'):
                self.test_aug = albumentations.Compose([
                    albumentations.SmallestMaxSize(max_size=self.gt_size),
                    albumentations.CenterCrop(self.gt_size, self.gt_size),
                ])
            augmented = self.test_aug(image=img_gt)
            img_gt = augmented['image']
        elif self.mode == 'training':
            img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])
        else:
            raise ValueError(f"Unexpected value {self.mode} for mode parameter")

        # -------------------- Crop or Pad to Desired Size -------------------- #
        h, w = img_gt.shape[0:2]
        if self.rescale_gt:
            crop_pad_size = max(min(h, w), self.gt_size)
        else:
            crop_pad_size = self.crop_pad_size

        while h < crop_pad_size or w < crop_pad_size:
            pad_h = min(max(0, crop_pad_size - h), h)
            pad_w = min(max(0, crop_pad_size - w), w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            h, w = img_gt.shape[0:2]

        top = random.randint(0, img_gt.shape[0] - crop_pad_size) if img_gt.shape[0] > crop_pad_size else 0
        left = random.randint(0, img_gt.shape[1] - crop_pad_size) if img_gt.shape[1] > crop_pad_size else 0
        img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        # Crop CSV data if available and in training mode
        if vel_df is not None and 'csv_coord_cols' in self.opt:
            csv_coord_cols = self.opt['csv_coord_cols']
            if isinstance(csv_coord_cols, list) and len(csv_coord_cols) == 2:
                try:
                    x_col = csv_coord_cols[0]
                    y_col = csv_coord_cols[1]
                    H_crop, W_crop = img_gt.shape[:2]
                    vel_df_cropped = vel_df[
                        (vel_df[x_col] >= left) & (vel_df[x_col] < left + W_crop) &
                        (vel_df[y_col] >= top) & (vel_df[y_col] < top + H_crop)
                    ].copy()
                    # Adjust coordinates to be relative to the cropped image
                    vel_df_cropped.loc[:, x_col] -= left
                    vel_df_cropped.loc[:, y_col] -= top
                    vel_df = vel_df_cropped
                except KeyError:
                    print(f"Warning: Coordinate columns {csv_coord_cols} not found in {vel_path}")
                except Exception as e:
                    print(f"Error during CSV cropping: {e}")

        if self.rescale_gt and crop_pad_size != self.gt_size:
            img_gt = cv2.resize(img_gt, dsize=(self.gt_size, self.gt_size), interpolation=cv2.INTER_AREA)
            # Rescale velocity coordinates if needed - this depends on what the coordinates represent
            # If they are pixel locations, you would need to scale them.
            if vel_df is not None and 'csv_coord_cols' in self.opt:
                csv_coord_cols = self.opt['csv_coord_cols']
                if isinstance(csv_coord_cols, list) and len(csv_coord_cols) == 2:
                    try:
                        x_col = csv_coord_cols[0]
                        y_col = csv_coord_cols[1]
                        original_size = crop_pad_size
                        target_size = self.gt_size
                        vel_df.loc[:, x_col] = (vel_df[x_col] * (target_size / original_size)).round().astype(int)
                        vel_df.loc[:, y_col] = (vel_df[y_col] * (target_size / original_size)).round().astype(int)
                    except KeyError:
                        pass # Handle if coordinate columns are not present

        # ------------------------ Generate Kernels ------------------------ #
        kernel1 = self.generate_kernel(kernel_list=self.kernel_list,
                                     kernel_prob=self.kernel_prob,
                                       blur_sigma=self.blur_sigma,
                                       betag_range=self.betag_range,
                                       betap_range=self.betap_range,
                                       kernel_range=self.kernel_range1,
                                       sinc_prob=self.sinc_prob)
        kernel2 = self.generate_kernel(kernel_list=self.kernel_list2,
            kernel_prob=self.kernel_prob2,
                                       blur_sigma=self.blur_sigma2,
                                       betag_range=self.betag_range2,
                                       betap_range=self.betap_range2,
                                       sinc_prob=self.sinc_prob2,
                                       kernel_range=self.kernel_range2,)
        sinc_kernel = self.generate_final_sinc_kernel()

        # ------------------------ Convert to Tensors ------------------------ #
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel1 = torch.FloatTensor(kernel1)
        kernel2 = torch.FloatTensor(kernel2)
        sinc_kernel = torch.FloatTensor(sinc_kernel) if sinc_kernel is not None else self.pulse_tensor

        # ------------------------ Process Velocity Data for Tensor ------------------------ #
        if vel_df is not None and 'csv_coord_cols' in self.opt and len(self.opt['csv_coord_cols']) == 2:
            csv_coord_cols = self.opt['csv_coord_cols']
            try:
                x_col = csv_coord_cols[0]
                y_col = csv_coord_cols[1]
                vel_col = self.opt.get('csv_vel_col', 'Velocity')  # Assuming 'Velocity' is default

                if x_col in vel_df.columns and y_col in vel_df.columns and vel_col in vel_df.columns:
                    X = vel_df[x_col].values.astype(int)
                    Y = vel_df[y_col].values.astype(int)
                    Velocity = vel_df[vel_col].values.astype(np.float32)

                    H_out, W_out = self.gt_size, self.gt_size
                    vel_array_np = np.zeros((H_out, W_out), dtype=np.float32)

                    valid_indices = (Y >= 0) & (Y < H_out) & (X >= 0) & (X < W_out)
                    vel_array_np[Y[valid_indices], X[valid_indices]] = Velocity[valid_indices]

                    # This produces [H_out, W_out], i.e. a 2D array
                    vel_tensor = torch.from_numpy(vel_array_np).float()
                else:
                    print(f"Warning: Missing required columns in {vel_path}")
            except KeyError as e:
                print(f"KeyError: {e} in {vel_path}")
            except Exception as e:
                print(f"Error processing velocity data: {e} in {vel_path}")



        # ------------------------ Return Dictionary ------------------------ #
        return_d = {
            'gt': img_gt,
            'kernel1': kernel1,
            'kernel2': kernel2,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path,
            #'vel': vel_tensor,
           # 'vel_path': vel_path # Optionally return the velocity path
        }

        return return_d

    def load_velocity_csv(self, vel_path, img_shape=None):
        """
        Load the velocity CSV file.

        Args:
            vel_path (str): Path to the velocity CSV file.
            img_shape (tuple, optional): Shape of the ground truth image (H, W, C).
                                         This is now optional as the loading becomes more generic.

        Returns:
            pandas.DataFrame: DataFrame containing the data from the CSV file, or None if loading fails.
        """
        try:
            vel_data = pd.read_csv(vel_path, delimiter=',',header=None)
            return vel_data
        except Exception as e:
            print(f"Error loading velocity CSV at {vel_path}: {e}")
            return None

    def generate_kernel(self, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range, kernel_range, sinc_prob):
        """
        Generate a blur kernel based on the provided configurations.

        Args:
            kernel_list (list): List of kernel types.
            kernel_prob (list): List of probabilities for each kernel type.
            blur_sigma (float): Standard deviation for Gaussian blur.
            betag_range (tuple): Range for generalized Gaussian blur kernels.
            betap_range (tuple): Range for plateau blur kernels.
            kernel_range (list): List of possible kernel sizes.
            sinc_prob (float): Probability of generating a sinc filter.

        Returns:
            np.ndarray: Generated kernel.
        """
        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < sinc_prob:
            # Sinc Filter
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            # Mixed Kernels
            kernel = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma,
                blur_sigma,
                [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None
            )
        # Pad kernel to the required size
        pad_size = (self.blur_kernel_size - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return kernel

    def generate_final_sinc_kernel(self):
        """
        Generate the final sinc kernel based on the configuration.

        Returns:
            torch.FloatTensor: Final sinc kernel tensor.
        """
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range2)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=self.blur_kernel_size2)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor
        return sinc_kernel

    def degrade_fun(self, conf_degradation, im_gt, kernel1, kernel2, sinc_kernel):
        ori_h, ori_w = im_gt.size()[2:4]
        sf = conf_degradation.sf

        # ----------------------- The First Degradation Process ----------------------- #
        # Blur
        out = filter2D(im_gt, kernel1)

        # Random Resize
        updown_type = random.choices(['up', 'down', 'keep'], conf_degradation['resize_prob'])[0]
        if updown_type == 'up':
            scale = random.uniform(1, conf_degradation['resize_range'][1])
        elif updown_type == 'down':
            scale = random.uniform(conf_degradation['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Add Noise
        gray_noise_prob = conf_degradation['gray_noise_prob']
        if random.random() < conf_degradation['gaussian_noise_prob']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=conf_degradation['noise_range'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=conf_degradation['poisson_scale_range'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False
            )

        # ----------------------- The Second Degradation Process ----------------------- #
        if random.random() < conf_degradation['second_order_prob']:
            # Blur
            if random.random() < conf_degradation['second_blur_prob']:
                out = filter2D(out, kernel2)
            # Random Resize
            updown_type = random.choices(['up', 'down', 'keep'], conf_degradation['resize_prob2'])[0]
            if updown_type == 'up':
                scale = random.uniform(1, conf_degradation['resize_range2'][1])
            elif updown_type == 'down':
                scale = random.uniform(conf_degradation['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                mode=mode,
            )
            # Add Noise
            gray_noise_prob = conf_degradation['gray_noise_prob2']
            if random.random() < conf_degradation['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=conf_degradation['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=conf_degradation['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                )

        # ----------------------- Final Sinc Filter and Resizing ----------------------- #
        if random.random() < 0.5:
            # Resize Back + Sinc Filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode,
            )
            out = filter2D(out, sinc_kernel)
        else:
            # Sinc Filter + Resize Back
            out = filter2D(out, sinc_kernel)
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode,
            )

        # Resize back if necessary
        if conf_degradation.get('resize_back', False):
            out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')

        # Clamp and Round
        im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        return {'lq': im_lq.contiguous(), 'gt': im_gt}  
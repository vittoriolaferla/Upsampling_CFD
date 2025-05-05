import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# fixed constants for all samples
channel_target_mean = torch.tensor([0.537238078213845, 0.53890670991845, 0.5364345268330284], dtype=torch.float32)
channel_target_steep = torch.tensor([12., 12., 12.], dtype=torch.float32)

GLOBAL_UMIN_5TH, GLOBAL_UMAX_95TH = -0.97893, 0.83819
GLOBAL_VMIN_5TH, GLOBAL_VMAX_95TH = -0.97893, 0.83819
GLOBAL_WMIN_5TH, GLOBAL_WMAX_95TH = -0.97893, 0.83819


def decode_rgb_to_vector(
    rgb_image: np.ndarray,
    epsilon: float = 1e-16
) -> np.ndarray:
    """
    Decode an RGB image back to the 3-component velocity vector (u, v, w)
    using fixed constants.
    """
    if rgb_image.dtype not in (np.float32, np.float64):
        x = rgb_image.astype(np.float32) / 255.0
    else:
        x = rgb_image.copy()
    x = np.clip(x, epsilon, 1 - epsilon)

    def inv_sigmoid(xc, k, c):
        return (np.log(xc / (1 - xc)) + k * c) / k

    u_scaled = inv_sigmoid(x[:, :, 0], channel_target_steep[0], channel_target_mean[0])
    v_scaled = inv_sigmoid(x[:, :, 1], channel_target_steep[1], channel_target_mean[1])
    w_scaled = inv_sigmoid(x[:, :, 2], channel_target_steep[2], channel_target_mean[2])

    u = u_scaled * (GLOBAL_UMAX_95TH - GLOBAL_UMIN_5TH) + GLOBAL_UMIN_5TH
    v = v_scaled * (GLOBAL_VMAX_95TH - GLOBAL_VMIN_5TH) + GLOBAL_VMIN_5TH
    w = w_scaled * (GLOBAL_WMAX_95TH - GLOBAL_WMIN_5TH) + GLOBAL_WMIN_5TH

    return np.stack([u, v, w], axis=-1)



class DifferentiableRGBtoVel(nn.Module):
    """
    Numerically‐stable decoder from RGB→(u,v,w).
    Clamps inputs to [eps, 1-eps] so logit derivative is bounded.
    """
    def __init__(self, device=None, clamp_eps: float = 1e-2):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Sigmoid inversion params
        self.register_buffer('k', channel_target_steep.view(1, 3, 1, 1))
        self.register_buffer('c', channel_target_mean.view(1, 3, 1, 1))
        # Min/max scalers
        self.register_buffer('gmin', torch.tensor(
            [GLOBAL_UMIN_5TH, GLOBAL_VMIN_5TH, GLOBAL_WMIN_5TH],
            dtype=torch.float32
        ).view(1, 3, 1, 1))
        self.register_buffer('gmax', torch.tensor(
            [GLOBAL_UMAX_95TH, GLOBAL_VMAX_95TH, GLOBAL_WMAX_95TH],
            dtype=torch.float32
        ).view(1, 3, 1, 1))
        # How far away from 0/1 we clamp
        self.eps = clamp_eps
        self.to(device)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # Move to same device/dtype as our buffers
        rgb = rgb.to(self.k.device, dtype=self.k.dtype)
        # Clamp in [eps, 1-eps] rather than [0,1] to bound the logit derivative
        x = rgb.clamp(self.eps, 1 - self.eps)
        # Safe logit
        logit = torch.log(x / (1 - x))
        # Invert your sigmoid+minmax
        scaled = (logit + self.k * self.c) / self.k
        vel = scaled * (self.gmax - self.gmin) + self.gmin
        return vel

class Umass(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        # coefficient normalized by interior size
        self.register_buffer('coef', torch.tensor(1 / ((48 - 2) * (48 - 2)), dtype=torch.float32))

    def forward(self, fv: torch.Tensor, rv: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # fv, rv: (N,3,H,W) velocity components [u,v,w]
        device = fv.device
        rv = rv.to(device=device, dtype=fv.dtype)
        if mask is not None:
            mask = mask.to(device=device)
        f_u, f_w = fv[:, 0:1, :, :], fv[:, 2:3, :, :]
        r_u, r_w = rv[:, 0:1, :, :], rv[:, 2:3, :, :]
        m = mask.unsqueeze(1).float() if mask is not None else torch.ones_like(f_u)

        def deriv_loss(r_comp: torch.Tensor, f_comp: torch.Tensor, axis: int) -> torch.Tensor:
            dr = 0.5 * (torch.roll(r_comp, shifts=-1, dims=axis) - torch.roll(r_comp, shifts=1, dims=axis))
            df = 0.5 * (torch.roll(f_comp, shifts=-1, dims=axis) - torch.roll(f_comp, shifts=1, dims=axis))
            # zero borders for accurate central diff
            if axis == 2:
                dr[:, :, 0, :] = 0; dr[:, :, -1, :] = 0
                df[:, :, 0, :] = 0; df[:, :, -1, :] = 0
            elif axis == 3:
                dr[:, :, :, 0] = 0; dr[:, :, :, -1] = 0
                df[:, :, :, 0] = 0; df[:, :, :, -1] = 0
            diff = torch.abs(dr - df) * m
            return torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0).sum()

        lx = deriv_loss(r_u, f_u, axis=3)  # ∂u/∂x
        lz = deriv_loss(r_w, f_w, axis=2)  # ∂w/∂z
        total = lx + lz
        if self.debug:
            print(f"[Umass] loss_x={lx.item():.6f}, loss_z={lz.item():.6f}")
        return  self.coef*total



import torch 
import pandas as pd
from torchvision.transforms import Lambda
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def RGBtoVel(data): #based off Christoph's MATLAB code to generate the velocities from the images for validation steps (/FinalSubissions_Vecchiarelli/Code/MATLAB Code/RevertImageToNumerical.m)
    
    #takes the index from lookUpRGBValue and scales it within the bounds of the velocity for the specific vector
    def physicalFromNormalized(normalized, bounds):
        custom_min = bounds[0]
        custom_max = bounds[1]
        physical_val = custom_min + (custom_max - custom_min) * normalized
        return physical_val
    
    #This function gets takes the image data as a tensor and compares it to the colormap (currently '.csv' of RGB values from MATLAB's "parula" colormap).
    #calculates the minimum distance of the pixel to a cmap value and returns that index
    def lookUpRGBValue(cmap, image):
        cmap = cmap.unsqueeze(1).unsqueeze(1)
        min_value = -1
        max_value = 1
        normalizedRGB = (image - min_value) / (max_value - min_value)
        normalizedRGB = normalizedRGB.unsqueeze(0).repeat(256, 1, 1, 1)
        distances = torch.sqrt(torch.sum((torch.sub(normalizedRGB, cmap))**2, axis =3))
        [minVals,minIndex] = torch.min(distances, dim=(0))
        normalizedIndex = minIndex/ (255)
        return normalizedIndex
    
    
    #bounds for the velocity 
    boundsU = [-10.6864, 9.6914]
    #inputted data 
    img = torch.Tensor.permute(data, (1,2,0)).to(device)
    #import the parula colormap values from .csv
    cmap = torch.tensor(pd.read_csv('parula.csv', header= None).values).to(device)


    normalizedValueFromRGB = lookUpRGBValue(cmap, img)
    ValuesRemapped = physicalFromNormalized(normalizedValueFromRGB,boundsU)

    return ValuesRemapped


transform = Lambda(RGBtoVel)


#make the subset to be used in the "mass loss"
class GetVelfromRGB(Dataset):
    #initialize the class, get the data from the images and the transform (custom lambda transform)
    def __init__(self, rgb_data, transform = transform):
        self.rgb_data = rgb_data
        self.transform = transform

    #return the number of samples of the data
    def __len__ (self):
        return len(self.rgb_data)
    
    def __getitem__(self,index):
        #get the data for each index
        rgb_data = self.rgb_data[index]
        #apply the transform (RGBtoVel) to the data subset
        if self.transform:
            vel_data = self.transform(rgb_data)
        #return the ground truth and generated velocities as tensors
        return vel_data
    

class Umass_Vecchairelli(nn.Module):
    def __init__(self):
        super(Umass_Vecchairelli, self).__init__()
        self.coef = torch.tensor((1/(1*(256-2)*(256-2))))

    def forward(self,fake_data,real_data):
        fake_data = torch.Tensor.permute(fake_data, (1,2,0))
        real_data = torch.Tensor.permute(real_data, (1,2,0))
         
        diff = torch.abs((0.5 * (real_data[1:-1, 2:] - real_data[1:-1, :-2]))- (0.5 * (fake_data[1:-1, 2:] - fake_data[1:-1, :-2]))).sum()
        
        L_mass = self.coef * diff
        
        return L_mass
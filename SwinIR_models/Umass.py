import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Umass(nn.Module):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.coef = torch.tensor((1/(1*(256-2)*(256-2))))

    def forward(self, fake_velocities, real_velocities, mask=None):
        N, H, W = fake_velocities.shape

        fv = fake_velocities.unsqueeze(1)
        rv = real_velocities.unsqueeze(1)

        if mask is not None:
            m = mask.unsqueeze(1).float()
            if self.debug:
                total_pixels = mask.numel()
                kept = m.sum().item()
                dropped = total_pixels - kept
                print(f"[Umass] mask: keeping {kept:.0f}/{total_pixels} pixels, dropping {dropped:.0f}")
        else:
            m = torch.ones_like(fv)

        def deriv_loss(r, f, dim):
            dr = torch.zeros_like(r)
            df = torch.zeros_like(f)
            idx = [slice(None)]*4
            idx_f = idx.copy(); idx_b = idx.copy()
            idx[dim]    = slice(1, -1)
            idx_f[dim]  = slice(2, None)
            idx_b[dim]  = slice(None, -2)
            dr[tuple(idx)] = 0.5*(r[tuple(idx_f)] - r[tuple(idx_b)])
            df[tuple(idx)] = 0.5*(f[tuple(idx_f)] - f[tuple(idx_b)])
            diff = torch.abs(dr - df) * m

            if self.debug:
                # sum before/after mask
                raw = torch.abs(dr-df).sum().item()
                masked = diff.sum().item()
                print(f"  dim={dim} raw‐loss={raw:.4f}, masked‐loss={masked:.4f}")
            return diff.sum()

        lx = deriv_loss(rv, fv, dim=3)
        ly = deriv_loss(rv, fv, dim=2)
        total = lx + ly
        if self.debug:
            print(f"[Umass] total masked loss = {total.item():.4f}")
        return self.coef*total

class DifferentiableRGBtoVel(nn.Module):
    def __init__(self, vmin, vmax, num_values=256, colormap_name='gist_rainbow_r', device=None):
        super(DifferentiableRGBtoVel, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.vmin = vmin
        self.vmax = vmax
        self.num_values = num_values
        self.colormap_name = colormap_name
        self.cmap = self._get_colormap(colormap_name, num_values).to(self.device)
        # Precompute scalar velocities corresponding to colormap indices
        indices = torch.arange(num_values, device=self.device).float()
        self.v_i = vmin + (vmax - vmin) * indices / (num_values - 1)

    def _get_colormap(self, colormap_name, num_values):
        """Generates the specified colormap as a PyTorch tensor."""
        cmap = plt.cm.get_cmap(colormap_name, num_values)
        rgb_values = cmap(np.linspace(0, 1, num_values))[:, :3]
        return torch.tensor(rgb_values, dtype=torch.float32)

    def forward(self, image, temperature=1e-2):
        """
        Converts an RGB image tensor to a velocity field tensor.

        Args:
            image (torch.Tensor): Tensor of shape (N, 3, H, W) with RGB values
                                  normalized to [0, 1].
            temperature (float): Temperature parameter for the softmax function.

        Returns:
            torch.Tensor: Tensor of shape (N, 1, H, W) representing the velocity field.
        """
        N, C, H, W = image.shape
        num_values = self.cmap.shape[0]
        # Reshape image to (N, H, W, 3)
        image = image.permute(0, 2, 3, 1)
        # Reshape cmap to (1, 1, 1, num_values, 3)
        cmap = self.cmap.view(1, 1, 1, num_values, 3)
        # Compute squared distances
        distances = ((image.unsqueeze(-2) - cmap) ** 2).sum(dim=-1)
        # Apply softmax over num_values dimension
        weights = F.softmax(-distances / temperature, dim=-1)
        # Compute scalar velocities
        velocities = (weights * self.v_i.view(1, 1, 1, num_values)).sum(dim=-1)
        # Reshape to (N, 1, H, W)
        #velocities = velocities.unsqueeze(1)
        return velocities
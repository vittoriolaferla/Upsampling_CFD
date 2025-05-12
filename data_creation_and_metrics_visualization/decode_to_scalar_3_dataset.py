#!/usr/bin/env python3
"""
Decode RGB velocity images back to scalar-magnitude PNGs and rotate each
decoded field 90° clockwise.

Requires: numpy pillow matplotlib tqdm
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ───────── paths ──────────────────────────────────────────────────
OUT_RGB = Path('/home/vittorio/Documenti/Upsampling_CFD/results/test_DAT_x2_Outdoor_Vector/visualization/test_set_physic')
OUT_DEC = Path('/home/vittorio/Documenti/Upsampling_CFD/results/test_DAT_x2_Outdoor_Vector/visualization/decoded')
OUT_DEC.mkdir(parents=True, exist_ok=True)

# ───────── constants (must match the encoder) ─────────────────────
LIMITS = {'u': (-26.92474, 40.70022),
          'v': (-45.96069, 29.49678),
          'w': (-14.85385, 10.54332)}
MEANS  = {'u': 0.39610, 'v': 0.60889, 'w': 0.58505}
K      = (10, 10, 10)
CMAP   = 'gist_rainbow_r'
V_MIN, V_MAX = 0.0, 5.0
# ------------------------------------------------------------------


def decode_rgb(rgb: np.ndarray) -> np.ndarray:
    """Invert the RGB encoding → velocity magnitude."""
    if rgb.dtype.kind in 'iu':
        rgb = rgb.astype(np.float32) / 255.0
    else:
        rgb = rgb.astype(np.float32, copy=False)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    def inv_sig(x, k_, c_):
        x = np.clip(x, 1e-16, 1.0 - 1e-16)
        return (np.log(x / (1. - x)) + k_ * c_) / k_

    def rec(norm, bounds, k_, c_):
        return inv_sig(norm, k_, c_) * (bounds[1] - bounds[0]) + bounds[0]

    u = rec(r, LIMITS['u'], K[0], MEANS['u'])
    v = rec(g, LIMITS['v'], K[1], MEANS['v'])
    w = rec(b, LIMITS['w'], K[2], MEANS['w'])
    return np.sqrt(u*u + v*v + w*w)


# ───────── main loop ──────────────────────────────────────────────
pngs = list(OUT_RGB.rglob('*.png'))
print(f"Found {len(pngs)} RGB images …")

for png_path in tqdm(pngs, desc="Decoding + rotating"):
    rgb = np.array(Image.open(png_path))
    mag = decode_rgb(rgb)

    # 90° clockwise rotation (k = -1) — use k=1 for anti-clockwise
    mag_rot = np.rot90(mag, k=-1)

    rel   = png_path.relative_to(OUT_RGB)
    out_p = OUT_DEC / rel
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.imsave(out_p, mag_rot,
               cmap=CMAP, vmin=V_MIN, vmax=V_MAX,
               origin='lower')

print("Done.  Rotated decoded PNGs written to", OUT_DEC)

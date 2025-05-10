#!/usr/bin/env python3
"""
Build RGB / scalar / decoded PNGs from <stem>_{U,V,W}.csv triplets.

  • If USE_PRECOMPUTED = True, the script uses the constants you paste
    below (GLOBAL_* and CHANNEL_MEAN) and skips the file scan.

  • If USE_PRECOMPUTED = False, it rescans ROOT_DIR to compute those
    limits and means automatically (robust‐percentile optional).

Requires: numpy pandas matplotlib scipy tqdm pillow
"""

from __future__ import annotations
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from tqdm import tqdm
from PIL import Image

# ─────────────────────────── USER SETTINGS ────────────────────────────
ROOT_DIR   = Path('/home/vittorio/Documenti/Upsampling_CFD/datasets/HighRes_5x5x1.25')
OUT_RGB    = Path('/home/vittorio/Documenti/Upsampling_CFD/datasets/out/Vector')
OUT_SCALAR = Path('/home/vittorio/Documenti/Upsampling_CFD/datasets/out/Scalar')
OUT_DECODED= Path('/home/vittorio/Documenti/Upsampling_CFD/datasets/out/Decoded')

TARGET_SZ   = 256
CROP_ROWS   = slice(40, 91)                       # MATLAB 41:91
CROP_COLS   = slice(20, 71)                       # MATLAB 21:71
CHANNEL_STEEP = [10, 10, 10]                      # sigmoid k
CMAP          = 'gist_rainbow_r'


# U: -26.92474  →  40.70022
#  V: -45.96069  →  29.49678
#  W: -14.85385  →  10.54332


#mean
#  U: 0.39610
#  V: 0.60889
#  W: 0.58505

# --- choose the mode --------------------------------------------------
USE_PRECOMPUTED = True        # ← set to False to force a fresh scan
USE_ROBUST      = False
SAME_LIMITS     = False        # ← True = one (min,max) & mean for u,v,w
       # when scanning, use robust percentiles?
LOW_PERC, HIGH_PERC = 0.05, 99.95
# ---------------------------------------------------------------------

# ───── paste your constants here (only used if USE_PRECOMPUTED=True) ──
### ⇣ PASTE YOUR CONSTANTS HERE ⇣ ######################################
PRECOMP_LIMITS = {
    'u': (-26.92474 ,  40.70022),   # GLOBAL_UMIN, GLOBAL_UMAX
    'v': (-45.96069,  29.49678),   # GLOBAL_VMIN, GLOBAL_VMAX
    'w': (-14.85385 ,  10.54332)    # GLOBAL_WMIN, GLOBAL_WMAX
}
PRECOMP_MEANS = {               # CHANNEL_MEAN for sigmoid mid-points
    'u':  0.39610,
    'v': 0.60889,
    'w': 0.58505
}
#######################################################################
# ──────────────────────────────────────────────────────────────────────


# ╭────────── helpers to load, normalise, encode, decode ──────────╮
def load_triplet(base: Path):
    """Return u, v, w  2-D arrays."""
    u = pd.read_csv(base.with_name(base.name + '_U.csv'),
                    header=None, dtype=float).to_numpy()
    v = pd.read_csv(base.with_name(base.name + '_V.csv'),
                    header=None, dtype=float).to_numpy()
    w = pd.read_csv(base.with_name(base.name + '_W.csv'),
                    header=None, dtype=float).to_numpy()
    if not (u.shape == v.shape == w.shape):
        raise ValueError(f'Shape mismatch in {base.name}')
    return u, v, w

def soft_clip(a, vmin, vmax, k, c):
    scaled = (a - vmin) / (vmax - vmin)
    return 1 / (1 + np.exp(-k * (scaled - c)))

def encode_rgb(u_n, v_n, w_n):
    return np.stack((u_n, v_n, w_n), axis=2)

# ─── put this right after the encode_rgb() helper ─────────────────────────
def decode_rgb(rgb: np.ndarray,
               lim: dict[str, tuple[float, float]],
               mean: dict[str, float],
               k: tuple[int, int, int] = CHANNEL_STEEP) -> np.ndarray:
    """
    Invert the soft-clipping+sigmoid encoding and return the velocity
    magnitude.  Accepts either float [0-1] arrays or uint8 RGB images.
    """
    if rgb.dtype.kind in 'iu':          # uint8 etc. → scale to 0-1
        rgb = rgb.astype(np.float32) / 255.0
    else:                               # already float, just cast
        rgb = rgb.astype(np.float32, copy=False)

    u_n, v_n, w_n = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    def inv_sig(x, k_, c_):
        x = np.clip(x, 1e-16, 1.0 - 1e-16)
        return (np.log(x / (1. - x)) + k_ * c_) / k_

    def recover(norm, bounds, k_, c_):
        return inv_sig(norm, k_, c_) * (bounds[1] - bounds[0]) + bounds[0]

    u = recover(u_n, lim['u'], k[0], mean['u'])
    v = recover(v_n, lim['v'], k[1], mean['v'])
    w = recover(w_n, lim['w'], k[2], mean['w'])
    return np.sqrt(u * u + v * v + w * w)
# ──────────────────────────────────────────────────────────────────────────

# ╰──────────────────────────────────────────────────────────────────╯


# ╭──── optional scan to compute limits & means (same as before) ───╮
def compute_limits_and_means() -> tuple[dict, dict]:
    mins   = dict(u=np.inf,  v=np.inf,  w=np.inf)
    maxs   = dict(u=-np.inf, v=-np.inf, w=-np.inf)
    sums   = dict(u=0.0, v=0.0, w=0.0)
    counts = dict(u=0,   v=0,   w=0)

    robust = {'u': [], 'v': [], 'w': []} if USE_ROBUST else None
    found_any = False

    for u_csv in tqdm(list(ROOT_DIR.rglob('*_U.csv')), desc='Scanning'):
        base = u_csv.with_suffix('').with_name(u_csv.stem[:-2])
        if not (base.with_name(base.name + '_V.csv').exists() and
                base.with_name(base.name + '_W.csv').exists()):
            continue
        found_any = True
        try:
            u, v, w = load_triplet(base)
        except Exception:
            continue
        u, v, w = u[CROP_ROWS, CROP_COLS], v[CROP_ROWS, CROP_COLS], w[CROP_ROWS, CROP_COLS]

        if USE_ROBUST:
            robust['u'].append(u.flatten())
            robust['v'].append(v.flatten())
            robust['w'].append(w.flatten())
        else:
            mins['u'], mins['v'], mins['w'] = min(mins['u'], u.min()), min(mins['v'], v.min()), min(mins['w'], w.min())
            maxs['u'], maxs['v'], maxs['w'] = max(maxs['u'], u.max()), max(maxs['v'], v.max()), max(maxs['w'], w.max())

        for comp, arr in zip('uvw', (u, v, w)):
            sums[comp]   += arr.sum()
            counts[comp] += arr.size

    if not found_any:
        raise RuntimeError(f"No complete triplets found in '{ROOT_DIR}'")

    if USE_ROBUST:
        limits = {c: (np.percentile(np.concatenate(robust[c]), LOW_PERC),
                      np.percentile(np.concatenate(robust[c]), HIGH_PERC))
                  for c in 'uvw'}
    else:
        limits = {c: (mins[c], maxs[c]) for c in 'uvw'}

    means = {c: (sums[c]/counts[c] - limits[c][0]) /
                  (limits[c][1] - limits[c][0]) for c in 'uvw'}
    return limits, means
# ╰──────────────────────────────────────────────────────────────────╯


# ╭────────────────────── dataset builder loop ─────────────────────╮
def build_dataset(lim, mean):
    made, skipped = 0, Counter()
    for u_csv in tqdm(list(ROOT_DIR.rglob('*_U.csv')), desc='Encoding'):
        base = u_csv.with_suffix('').with_name(u_csv.stem[:-2])
        if not (base.with_name(base.name + '_V.csv').exists() and
                base.with_name(base.name + '_W.csv').exists()):
            continue
        try:
            u, v, w = load_triplet(base)
        except Exception as err:
            skipped.update({'load_error': 1})
            tqdm.write(f'⨯ {base.name}: {err}')
            continue

        u, v, w = u[CROP_ROWS, CROP_COLS], v[CROP_ROWS, CROP_COLS], w[CROP_ROWS, CROP_COLS]
        if u.shape != (TARGET_SZ, TARGET_SZ):
            zf = (TARGET_SZ/u.shape[0], TARGET_SZ/u.shape[1])
            u = zoom(u, zf, order=1)
            v = zoom(v, zf, order=1)
            w = zoom(w, zf, order=1)


        mag = np.sqrt(u*u + v*v + w*w)
        u_n = soft_clip(u, *lim['u'], CHANNEL_STEEP[0], mean['u'])
        v_n = soft_clip(v, *lim['v'], CHANNEL_STEEP[1], mean['v'])
        w_n = soft_clip(w, *lim['w'], CHANNEL_STEEP[2], mean['w'])
        rgb = encode_rgb(u_n, v_n, w_n)

        if SAME_LIMITS:
            # one set of limits/mean for all → no component mismatch possible
            err = np.abs(decode_rgb(rgb, lim, mean) - mag).max()
            if err > 1e-2:
                tqdm.write(f'⚠ large decode error {err:.3g} for {base.name}')
        else:
            # keep the component-wise test
            err = np.abs(decode_rgb(rgb, lim, mean) - mag).max()
            if err > 1e-2:
                tqdm.write(f'⚠ large decode error {err:.3g} for {base.name}')

        rel = base.relative_to(ROOT_DIR).with_suffix('.png')
        p_rgb, p_sca, p_dec = OUT_RGB/rel, OUT_SCALAR/rel, OUT_DECODED/rel
        for p in (p_rgb, p_sca, p_dec):
            p.parent.mkdir(parents=True, exist_ok=True)

# ─── replace the three plt.imsave() calls at the end of build_dataset() ──
        # choose the same colour range as the “3rd-dataset” script
        C_AXIS = (0.0, 5.0)

        # 1) RGB vector image (no cmap)
        plt.imsave(p_rgb, rgb, origin='lower')

        # 2) scalar (= true) magnitude
        plt.imsave(p_sca, mag,
                   cmap=CMAP, vmin=C_AXIS[0], vmax=C_AXIS[1],
                   origin='lower')

        # 3) decoded magnitude
        mag_dec = decode_rgb(rgb, lim, mean)
        plt.imsave(p_dec, mag_dec,
                   cmap=CMAP, vmin=C_AXIS[0], vmax=C_AXIS[1],
                   origin='lower')
# ──────────────────────────────────────────────────────────────────────────

        made += 1
    print(f"\nImages created: {made}")
    if skipped:
        print(f"Skipped triplets: {dict(skipped)}")
# ╰──────────────────────────────────────────────────────────────────╯


# ╭───────────────────────────── main ═─────────────────────────────╮
if __name__ == "__main__":
    if USE_PRECOMPUTED:
        LIM, MEAN = PRECOMP_LIMITS, PRECOMP_MEANS
        print("Using pre-computed limits & means.")
    else:
        print("➤ Scanning to compute limits & means …")
        LIM, MEAN = compute_limits_and_means()
        print("\nComputed limits:")
        for c in 'uvw':
            print(f" {c.upper()} {LIM[c][0]:.5f} → {LIM[c][1]:.5f}")
        print("Computed channel means:")
        for c in 'uvw':
            print(f" {c.upper()}: {MEAN[c]:.5f}")
        input("\nPress <Enter> to build the dataset …")

    build_dataset(LIM, MEAN)
    print("\nDataset complete. Check the 'out/' directory.")
# ╰─────────────────────────────────────────────────────────────────╯

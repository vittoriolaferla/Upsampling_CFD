#!/usr/bin/env python3
# build_master_components_from_triplets.py
#
# • scans ROOT_DIR for triplets  <stem>_{U,V,W}.csv
# • finds a global min / max across u, v, w
# • (crop → optional zoom) exactly like your PNG pipeline
# • dumps three master component files in OUT_DIR

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import zoom

# ──────────────────────────────────────────────────────────────
ROOT_DIR  = Path('/home/vittorio/Documenti/Upsampling_CFD/datasets/HighRes_5x5x1.25')
OUT_DIR   = Path('/home/vittorio/Documenti/Upsampling_CFD/velocity_components_normalized/ThirdDataset')
CROP_ROWS = slice(40, 91)        # 51 rows  (MATLAB 41:91)
CROP_COLS = slice(20, 71)        # 51 cols  (MATLAB 21:71)
TARGET_SZ = 256                  # set to None to skip up-sampling
# ──────────────────────────────────────────────────────────────


def load_triplet(base: Path):
    """Return cropped (and optionally up-sampled) u, v, w arrays."""
    u = pd.read_csv(base.with_name(base.name + '_U.csv'),
                    header=None, dtype=float).to_numpy()
    v = pd.read_csv(base.with_name(base.name + '_V.csv'),
                    header=None, dtype=float).to_numpy()
    w = pd.read_csv(base.with_name(base.name + '_W.csv'),
                    header=None, dtype=float).to_numpy()

    # crop to the same window used for the images
    u, v, w = u[CROP_ROWS, CROP_COLS], v[CROP_ROWS, CROP_COLS], w[CROP_ROWS, CROP_COLS]

    if TARGET_SZ is not None and u.shape != (TARGET_SZ, TARGET_SZ):
        zf = (TARGET_SZ / u.shape[0], TARGET_SZ / u.shape[1])
        u = zoom(u, zf, order=1)
        v = zoom(v, zf, order=1)
        w = zoom(w, zf, order=1)

    return u, v, w


# ──────────── pass 1  • find global min / max ────────────
triplet_bases = [p.with_suffix('').with_name(p.stem[:-2])
                 for p in ROOT_DIR.rglob('*_U.csv')]
triplet_bases = [b for b in triplet_bases
                 if (b.with_name(b.name + '_V.csv').exists()
                     and b.with_name(b.name + '_W.csv').exists())]

if not triplet_bases:
    raise RuntimeError(f'No complete U/V/W triplets found under {ROOT_DIR}')

gmin, gmax = np.inf, -np.inf
for base in tqdm(triplet_bases, desc='Scanning min/max'):
    u, v, w = load_triplet(base)
    gmin = min(gmin, u.min(), v.min(), w.min())
    gmax = max(gmax, u.max(), v.max(), w.max())

print(f'→ global component range:  {gmin:.6f}  …  {gmax:.6f}')

# ──────────── pass 2  • collect & write ────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
path_u = OUT_DIR / 'master_u_norm.csv'
path_v = OUT_DIR / 'master_v_norm.csv'
path_w = OUT_DIR / 'master_w_norm.csv'

# open the three files once in append-text mode
with path_u.open('w') as fu, path_v.open('w') as fv, path_w.open('w') as fw:
    fu.write('u_norm\n');  fv.write('v_norm\n');  fw.write('w_norm\n')

    for base in tqdm(triplet_bases, desc='Normalising & appending'):
        u, v, w = load_triplet(base)

        u_n = (u - gmin) / (gmax - gmin)
        v_n = (v - gmin) / (gmax - gmin)
        w_n = (w - gmin) / (gmax - gmin)
        u_n, v_n, w_n = [np.clip(a, 0.0, 1.0).ravel()
                         for a in (u_n, v_n, w_n)]

        # write one value per line – fastest, least RAM
        fu.writelines(f'{x}\n' for x in u_n)
        fv.writelines(f'{x}\n' for x in v_n)
        fw.writelines(f'{x}\n' for x in w_n)

print(f'✔  Master component files ready in {OUT_DIR}')

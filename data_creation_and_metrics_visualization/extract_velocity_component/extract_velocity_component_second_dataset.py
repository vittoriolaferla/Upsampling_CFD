#!/usr/bin/env python
# make_master_component_csvs.py
#
# • Pass 1 : scan every .h5 -> global min / max across u, v, w
# • Pass 2 : centre-Y slice from each file, normalise, append to
#              running Python lists
# • Finish : dump three single-column CSVs:
#              master_u_norm.csv , master_v_norm.csv , master_w_norm.csv
#
#  Requirements: h5py, numpy, pandas, tqdm

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
SRC_DIR = Path("/home/vittorio/Scrivania/ETH/Upsampling/Upsampling_CFD/datasets/dataset_obstacles/hdf-files")
OUT_DIR = Path("/home/vittorio/Documenti/Upsampling_CFD/velocity_components_normalized/SecondDataset")
VELOCITY_KEY = "data_B"           # (flattened) velocity field per file
VOL_SHAPE = (64, 64, 64, 3)       # reshape order='C' → (x, y, z, comp)
# ────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
h5_files = sorted(SRC_DIR.glob("*.h5"))
if not h5_files:
    raise FileNotFoundError(f"No .h5 files found in {SRC_DIR}")

# ───────────── 1 ◉  pass – global min / max ─────────────
gmin, gmax = np.inf, -np.inf
print("Scanning global velocity range …")
for f in tqdm(h5_files):
    with h5py.File(f, "r") as h5:
        vel = h5[VELOCITY_KEY][:]
        vel = vel.reshape(VOL_SHAPE, order="C")
        gmin = min(gmin, vel.min())
        gmax = max(gmax, vel.max())
print(f"→ global component range:  {gmin:.6f}  …  {gmax:.6f}")

# ───────────── 2 ◉  pass – gather Y-slice data ──────────
all_u, all_v, all_w = [], [], []          # running Python lists

for f in tqdm(h5_files, desc="Collecting centre-Y slices"):
    with h5py.File(f, "r") as h5:
        vel = h5[VELOCITY_KEY][:]
        vel = vel.reshape(VOL_SHAPE, order="C")
        u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]

    cy = u.shape[1] // 2                  # centre Y index
    slice_u = u[:, cy, :]
    slice_v = v[:, cy, :]
    slice_w = w[:, cy, :]

    # normalise to [0, 1] using global min/max
    u_n = (slice_u - gmin) / (gmax - gmin)
    v_n = (slice_v - gmin) / (gmax - gmin)
    w_n = (slice_w - gmin) / (gmax - gmin)
    u_n, v_n, w_n = [np.clip(a, 0.0, 1.0) for a in (u_n, v_n, w_n)]

    # flatten and extend the running lists
    all_u.extend(u_n.ravel())
    all_v.extend(v_n.ravel())
    all_w.extend(w_n.ravel())

# ───────────── 3 ◉  write the three master CSVs ─────────
pd.Series(all_u, name="u_norm").to_csv(OUT_DIR / "master_u_norm.csv", index=False)
pd.Series(all_v, name="v_norm").to_csv(OUT_DIR / "master_v_norm.csv", index=False)
pd.Series(all_w, name="w_norm").to_csv(OUT_DIR / "master_w_norm.csv", index=False)

print(f"✔  Finished. 3 component files saved in  {OUT_DIR}")
print(f"   rows per file: {len(all_u):,}  (4096 × {len(h5_files)} slices)")

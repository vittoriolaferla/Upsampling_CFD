#!/usr/bin/env python
# build_master_components_from_csv.py
#
# 1. Pass over all *.csv in SRC_DIR once: find a single global min / max
#    across the u, v, w columns.
# 2. Re-read each file, normalise to that common [0, 1] scale, append
#    the results to running Python lists.
# 3. Dump three one-column CSVs in OUT_DIR:
#       master_u_norm.csv , master_v_norm.csv , master_w_norm.csv
#
# Requirements: numpy, pandas, pathlib, tqdm  (pip install tqdm)

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

# ────────────────────────────────────────────────────────────────
SRC_DIR = Path('/home/vittorio/Scaricati/Final_Steps_47')
OUT_DIR = Path('/home/vittorio/Documenti/Upsampling_CFD/velocity_components_normalized/FirstDataset')
# ────────────────────────────────────────────────────────────────


def read_uvws(path: Path) -> np.ndarray:
    """Return an (N × 3) array holding u, v, w for one CFD CSV."""
    return (
        pd.read_csv(path,
                    skiprows=2,          # skip label + units lines
                    header=None,
                    usecols=[4, 5, 6]    # u, v, w columns
                   )
        .to_numpy(float)                # shape (N, 3)
    )


def global_min_max(csv_files) -> Tuple[float, float]:
    """Scan every file once to discover the absolute min / max."""
    gmin, gmax = np.inf, -np.inf
    for f in tqdm(csv_files, desc='Scanning min/max'):
        uvw = read_uvws(f)
        gmin = min(gmin, uvw.min())
        gmax = max(gmax, uvw.max())
    return gmin, gmax


def main():
    csv_files = sorted(SRC_DIR.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No CSV files found in {SRC_DIR}')

    print(f'→ Found {len(csv_files)} source files')
    gmin, gmax = global_min_max(csv_files)
    print(f'→ Global velocity range:  {gmin:.6f}  …  {gmax:.6f}')

    # running containers for every value in every file
    all_u, all_v, all_w = [], [], []

    for f in tqdm(csv_files, desc='Collecting & normalising'):
        uvw = read_uvws(f)
        uvw_norm = (uvw - gmin) / (gmax - gmin)
        uvw_norm = np.clip(uvw_norm, 0.0, 1.0)

        # extend the master lists
        all_u.extend(uvw_norm[:, 0])
        all_v.extend(uvw_norm[:, 1])
        all_w.extend(uvw_norm[:, 2])

    # write the three master columns
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.Series(all_u, name='u_norm').to_csv(OUT_DIR / 'master_u_norm.csv', index=False)
    pd.Series(all_v, name='v_norm').to_csv(OUT_DIR / 'master_v_norm.csv', index=False)
    pd.Series(all_w, name='w_norm').to_csv(OUT_DIR / 'master_w_norm.csv', index=False)

    print(f'✔  Master component files written to {OUT_DIR}')
    print(f'   total rows per file: {len(all_u):,}')


if __name__ == '__main__':
    main()

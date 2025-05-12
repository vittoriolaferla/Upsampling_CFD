# density_3x3.py  –  triple-panel KDE plot for u, v, w components
#
# Edit ONLY the three *_DIR variables below.
# Each directory must contain:
#     master_u_norm.csv
#     master_v_norm.csv
#     master_w_norm.csv
# produced by the “master CSV” scripts we wrote earlier.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ───────────────────────  EDIT THESE  ─────────────────────────
EXP1_DIR = Path("/home/vittorio/Documenti/Upsampling_CFD/velocity_components_normalized/FirstDataset")     # CFD Final_Steps_47
EXP2_DIR = Path("/home/vittorio/Documenti/Upsampling_CFD/velocity_components_normalized/SecondDataset")    # HDF-5 Y-slices
EXP3_DIR = Path("/home/vittorio/Documenti/Upsampling_CFD/velocity_components_normalized/ThirdDataset")     # HighRes 5×5×1.25
# ↓ Optional: customise the labels that appear in the legend
LABEL1, LABEL2, LABEL3 = "Experiment 1", "Experiment 2", "Experiment 3"
# ──────────────────────────────────────────────────────────────

# no edits needed below this line ---------------------------------------
COMPONENTS = ["u", "v", "w"]

EXPERIMENTS = {
    LABEL1: EXP1_DIR,
    LABEL2: EXP2_DIR,
    LABEL3: EXP3_DIR,
}

# read the nine files
data = {comp: {} for comp in COMPONENTS}
for label, folder in EXPERIMENTS.items():
    for comp in COMPONENTS:
        csv_path = folder / f"master_{comp}_norm.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing file: {csv_path}")
        data[comp][label] = pd.read_csv(csv_path)[f"{comp}_norm"]

# plot
sns.set_theme("paper", style="whitegrid")
palette = sns.color_palette("colorblind", n_colors=len(EXPERIMENTS))
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True, constrained_layout=True)

for ax, comp in zip(axes, COMPONENTS):
    for idx, (label, series) in enumerate(data[comp].items()):
        sns.kdeplot(
            series,
            ax=ax,
            label=label,
            fill=True,
            alpha=0.30,
            linewidth=1.4,
            color=palette[idx],
        )
    ax.set_title(f"{comp.upper()} component")
    ax.set_xlabel("normalised value (0–1)")
    if ax is axes[0]:
        ax.set_ylabel("density")
        ax.legend(frameon=False)
    else:
        ax.set_ylabel("")

plt.show()

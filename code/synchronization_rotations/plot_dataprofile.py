import shutil
import sys
import os
import pickle
import itertools

import matplotlib.pyplot as plt
import matplotlib as mpl

from dataprofile import dataprofile

sys.path.insert(0, os.path.abspath(".."))

plt.rcParams["text.usetex"] = shutil.which("latex") is not None

# Load latest experiment automatically
exp_dir = "experiments"
latest_exp = sorted(os.listdir(exp_dir))[-1]
results_dir = os.path.join(exp_dir, latest_exp, "results")
latest_pkl = sorted(f for f in os.listdir(results_dir) if f.endswith(".pkl"))[-1]
path_frame = os.path.join(results_dir, latest_pkl)

with open(path_frame, "rb") as f:
    df = pickle.load(f)

df = df.drop(["idpb", "alpha_0", "gamma", "Gamma", "alpha_max", "idalgo"], axis=1)

# Plot formatting
cmapproj = mpl.colormaps["Set1"].colors[1:4]
cmapnoproj = mpl.colormaps["Set1"].colors[1:4]
cmap = [cmapnoproj, cmapproj]
linestyles = ["-", "--"]
linewidths = [2, 2]
variants = ["intrinsic", "projected"]
titlefonts = 28

vtaus = [0.01, 0.05, 0.1]
rotation_labels = ["No rotation", "Rotation"]

fig, ax = plt.subplots(2, 3, figsize=(15, 8))

for i_r, rotation in enumerate([0, 1]):
    for i_t, tau in enumerate(vtaus):
        dp, scalingdim, _ = dataprofile(df, tau=tau, euclideansimplex=0, lbdfoeuclideansimplex=0)
        for projection, psstype in itertools.product([0, 1], [1, 2, 3]):
            thedp = dp[
                (dp["projection"] == projection)
                & (dp["rotation"] == rotation)
                & (dp["psstype"] == psstype)
            ]
            vx = thedp["alpha"].values * scalingdim
            vf = thedp["ratio_solved"].values
            ax[i_r, i_t].plot(
                vx, vf,
                color=cmap[projection][psstype - 1],
                linestyle=linestyles[projection],
                linewidth=linewidths[projection],
                label=f"PSS{psstype} ({variants[projection]})",
            )
        ax[i_r, i_t].grid()
        ax[i_r, i_t].set_title(f"tau: {tau}")

for i_r, label in enumerate(rotation_labels):
    ax[i_r, 0].set_ylabel(label, fontsize=16)


fig.supxlabel("Number of simplex gradient evaluations", fontsize=titlefonts)
fig.supylabel("Ratio of problems solved", fontsize=titlefonts)
plt.tight_layout()

pkl_prefix = os.path.splitext(latest_pkl)[0]
dataprofiles_dir = os.path.join(results_dir, "dataprofiles")
os.makedirs(dataprofiles_dir, exist_ok=True)
out_path = os.path.join(dataprofiles_dir, f"dataprofile-{pkl_prefix}.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")

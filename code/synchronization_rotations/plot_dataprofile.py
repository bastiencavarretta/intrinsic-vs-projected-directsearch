import argparse
import shutil
import sys
import os
import pickle
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

from dataprofile import dataprofile

sys.path.insert(0, os.path.abspath(".."))

plt.rcParams["text.usetex"] = shutil.which("latex") is not None

parser = argparse.ArgumentParser()
parser.add_argument("--exppath", type=str, required=True, help="Path to experiment directory")
parser.add_argument("--dryrun", action="store_true", help="Load dryrun results and suffix output with _dryrun")
args = parser.parse_args()

results_dir = os.path.join(args.exppath, "results")
all_pkls = sorted(f for f in os.listdir(results_dir) if f.endswith(".pkl"))
if args.dryrun:
    all_pkls = [f for f in all_pkls if "_dryrun" in f]
else:
    all_pkls = [f for f in all_pkls if "_dryrun" not in f]
latest_pkl = all_pkls[-1]
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

dataprofiles_dir = os.path.join(args.exppath, "plots")
os.makedirs(dataprofiles_dir, exist_ok=True)
dr_suffix = "_dryrun" if args.dryrun else ""
out_path = os.path.join(dataprofiles_dir, f"dataprofile{dr_suffix}.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")

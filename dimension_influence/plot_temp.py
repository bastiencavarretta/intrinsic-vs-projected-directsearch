import sys
import os

sys.path.append(os.path.abspath(".."))
# print(sys.path)

# Importing python packages
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..")))
print(sys.path[0])
# Importing everything
from problems.newmanifolds import *
from tools.utils_cm_hypersphere import *
from tools.perform_directsearch import perform_ds, saveperform_ds, ds_key, directsearch
from tools.plotandsave_dataprofile import plotting_dp
from tools.saveprint_tables import performancetable

# Plotting parameters
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cm"],
        "text.usetex": True,
        # 'font.size' : 10,
        # "axes.labelsize": 10,
        # "legend.fontsize": 8,
        # "xtick.labelsize": 8,
        # "ytick.labelsize": 8,
        "figure.figsize": (6.4, 4.0),
        # "pgf.texsystem": "pdflatex",
    }
)

# Fixing the seed
np.random.seed(1)


expnumber = 1
maniftypeobj = [(1, 2), (1, 1), (1, 3), (2, 1)]
saving = True

for rotation in [0, 1]:
    for lbdfoeuclideansimplex in [0, 1]:
        if lbdfoeuclideansimplex == 1:
            euclideansimplex = 1
            plotting_dp(
                expnumber,
                maniftypeobj=maniftypeobj,
                rotations=[rotation],
                psstypes=[1, 2, 3],
                projections=[0, 1],
                euclideansimplex=euclideansimplex,
                lbdfoeuclideansimplex=lbdfoeuclideansimplex,
                tau=1e-2,
                N=100,
                plotcodimsev=True,
                plotmdimsev=True,
                saving=saving,
            )
        euclideansimplex = 0
        plotting_dp(
            expnumber,
            maniftypeobj=maniftypeobj,
            rotations=[rotation],
            psstypes=[1, 2, 3],
            projections=[0, 1],
            euclideansimplex=euclideansimplex,
            lbdfoeuclideansimplex=lbdfoeuclideansimplex,
            tau=1e-2,
            N=100,
            plotcodimsev=True,
            plotmdimsev=True,
            saving=saving,
        )

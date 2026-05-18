import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dataprofile import plotting_dp


import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from dataprofile import plotting_dp

parser = argparse.ArgumentParser()
parser.add_argument("exp", type=str, help="Path to experiment directory")
args = parser.parse_args()

EXP = args.exp

plotting_dp(
    exppath=EXP,
    plottingworld="codimev",
    fixed_dim=4,  # mdim held fixed at 4
    projections=[0, 1],
    psstypes=[1, 2, 3],
    rotations=[0],
    tau=1e-2,
    saving=True,
)

plotting_dp(
    exppath=EXP,
    plottingworld="mdimev",
    fixed_dim=4,
    projections=[0, 1],
    psstypes=[1, 2, 3],
    rotations=[0],
    tau=1e-2,
    saving=True,
)

plotting_dp(
    exppath=EXP,
    plottingworld="codimev",
    fixed_dim=4,
    projections=[0, 1],
    psstypes=[1, 2, 3],
    rotations=[1],
    tau=1e-2,
    saving=True,
)

plotting_dp(
    exppath=EXP,
    plottingworld="mdimev",
    fixed_dim=4,
    projections=[0, 1],
    psstypes=[1, 2, 3],
    rotations=[1],
    tau=1e-2,
    saving=True,
)

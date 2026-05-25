import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from dataprofile import plotting_dp

parser = argparse.ArgumentParser()
parser.add_argument("--exppath", type=str, required=True, help="Path to experiment directory")
parser.add_argument("--dryrun", action="store_true", help="Load dryrun results and suffix output with _dryrun")
args = parser.parse_args()

EXP = args.exppath

plotting_dp(
    exppath=EXP,
    plottingworld="codimev",
    fixed_dim=4,
    projections=[0, 1],
    psstypes=[1, 2, 3],
    rotations=[0],
    tau=1e-2,
    saving=True,
    dryrun=args.dryrun,
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
    dryrun=args.dryrun,
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
    dryrun=args.dryrun,
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
    dryrun=args.dryrun,
)

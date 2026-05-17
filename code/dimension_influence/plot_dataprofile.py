import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from code.dimension_influence.plottingdataprofile import plotting_dp

EXP = "experiments/2026-05-17_fixedmdim_budg100_v1"

plotting_dp(
    exppath=EXP,
    plottingworld="codimev",
    fixed_dim=4,  # mdim held fixed at 4
    projections=[0, 1],
    psstypes=[1, 2, 3],
    rotations=[0],
    euclsimplex=1,
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
    euclsimplex=1,
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
    euclsimplex=1,
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
    euclsimplex=1,
    tau=1e-2,
    saving=True,
)

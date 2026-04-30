# # Plooting parameters
# from psutil.tests.test_posix import df

from ctypes.util import test
from math import e

import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import dill
import matplotlib as mpl
import itertools
import pandas as pd


# Computing data profiles
def dataprofile(df_run, tau=1e-2, euclideansimplex=0, lbdfoeuclideansimplex=1):

    # df is the dataprofile of
    df = df_run.copy()
    N = df["simplexbudget"].iloc[0]  # supposed to be constant

    nbinstance = len(df["seed"].unique())

    if (euclideansimplex == 1 or lbdfoeuclideansimplex == 1) and df[
        "euclideansimplex"
    ].iloc[0] == 0:
        raise ValueError(
            "euclideansimplex and lbdfoeuclideansimplex must be 0 if the directsearch data was obtained by running perform_directsearch.saveperform_ds with euclideansimplex = 0."
        )

    df["lastf"] = df["vf"].transform(lambda x: x[-1] if len(x) > 0 else None)
    df["rb_lastf"] = df["rb_vf"].transform(lambda x: x[-1] if len(x) > 0 else None)

    df["lbdfo"] = df.groupby("seed")["lastf"].transform("min")
    df["rb_lbdfo"] = df.groupby("seed")["rb_lastf"].transform("min")
    nb_problems = 1
    nbinstances = len(df["seed"].unique())

    vfalpha = []

    mdim, codim = df.iloc[0]["problem"].mdim, df.iloc[0]["problem"].codim
    scalingdim = mdim + codim if euclideansimplex == 1 else mdim

    for seed in df["seed"].unique():
        df_seed = df[df["seed"] == seed]
        for projection, rotation, psstype in itertools.product(
            [0, 1], [0, 1], [1, 2, 3]
        ):
            lbdfo = (
                df_seed["lbdfo"].iloc[0]
                if lbdfoeuclideansimplex == 1
                else df_seed["rb_lbdfo"].iloc[0]
            )

            df_variant = df_seed[
                (df_seed["projection"] == projection)
                & (df_seed["rotation"] == rotation)
                & (df_seed["psstype"] == psstype)
            ]

            evaluations = (
                df_variant["per_iteration_evaluations"]
                if euclideansimplex == 1
                else df_variant["rb_per_iteration_evaluations"]
            )
            evaluations = np.array(evaluations.tolist())[0]
            evaluations = np.cumsum(evaluations)
            vf = df_variant["vf"] if euclideansimplex == 1 else df_variant["rb_vf"]
            vf = np.array(vf.tolist())[0]

            f0 = vf[0]

            # print("lbdfo = ", lbdfo)
            # print("f0 = ", f0)
            # print("vf = ", vf)

            test_value = (vf - lbdfo) / (f0 - lbdfo)
            indices_solving = np.where(test_value <= tau)[0]
            if len(indices_solving) == 0:
                alpha = N + 2
            else:
                # print("hey")
                index_first_solving = np.min(indices_solving)
                # print(evaluations)
                alpha = int(
                    evaluations[index_first_solving] / (scalingdim + 1)
                )  # alpha is the abscissa of the data profile
                # print("heybis")

            # storing for every problem and solver
            vfalpha.append(
                {
                    "seed": seed,
                    "projection": projection,
                    "rotation": rotation,
                    "psstype": psstype,
                    "alpha": alpha,
                }
            )
    vfalpha = pd.DataFrame(vfalpha)
    # print("hhhhhh")
    # return vfalpha

    # Creating one common dataprofile for all problems
    dp = []
    for projection, rotation, psstype in itertools.product([0, 1], [0, 1], [1, 2, 3]):
        for alpha in range(N + 1):

            test = vfalpha[
                (vfalpha["projection"] == projection)
                & (vfalpha["rotation"] == rotation)
                & (vfalpha["psstype"] == psstype)
            ]
            # print(test)
            ratio_solved = (
                vfalpha[
                    (vfalpha["projection"] == projection)
                    & (vfalpha["rotation"] == rotation)
                    & (vfalpha["psstype"] == psstype)
                ]["alpha"]
                .transform(lambda x: x <= alpha)
                .mean()
            )

            dp.append(
                {
                    "psstype": psstype,
                    "projection": projection,
                    "rotation": rotation,
                    "alpha": alpha,
                    "ratio_solved": ratio_solved,
                }
            )
            # last two entries (alpha and ratio_solved) are the coordinates of the data profile.
    return pd.DataFrame(dp), scalingdim, df

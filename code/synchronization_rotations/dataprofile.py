#Plotting parameters
import numpy as np
import itertools
import pandas as pd


# Computing data profiles
def dataprofile(df_run, tau=1e-2, euclideansimplex=0, lbdfoeuclideansimplex=1):

    # df is the dataprofile of
    df = df_run.copy()
    N = df["simplexbudget"].iloc[0]  # supposed to be constant

    nbinstance = len(df["instance"].unique())

    if (euclideansimplex == 1 or lbdfoeuclideansimplex == 1) and df[
        "euclideansimplex"
    ].iloc[0] == 0:
        raise ValueError(
            "euclideansimplex and lbdfoeuclideansimplex must be 0 if the directsearch data was obtained by running perform_directsearch.saveperform_ds with euclideansimplex = 0."
        )

    df["lastf"] = df["vf"].transform(lambda x: x[-1] if len(x) > 0 else None)
    df["rb_lastf"] = df["rb_vf"].transform(lambda x: x[-1] if len(x) > 0 else None)

    df["lbdfo"] = df.groupby("instance")["lastf"].transform("min")
    df["rb_lbdfo"] = df.groupby("instance")["rb_lastf"].transform("min")
    nb_problems = 1
    nbinstances = len(df["instance"].unique())

    vfalpha = []

    mdim, codim = df.iloc[0]["problem"].mdim, df.iloc[0]["problem"].codim
    scalingdim = mdim + codim if euclideansimplex == 1 else mdim

    for instance in df["instance"].unique():
        df_seed = df[df["instance"] == instance]
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
            evaluations = np.array(evaluations.tolist()[0])

            evaluations = np.cumsum(evaluations)
            vf = df_variant["vf"] if euclideansimplex == 1 else df_variant["rb_vf"]
            vf = np.array(vf.tolist())[0]

            f0 = vf[0]
            test_value = (vf - lbdfo) / (f0 - lbdfo)
            indices_solving = np.where(test_value <= tau)[0]
            if len(indices_solving) == 0:
                alpha = N + 2
            else:
                index_first_solving = np.min(indices_solving)
                alpha = int(
                    evaluations[index_first_solving] / (scalingdim + 1)
                )  # alpha is the abscissa of the data profile

            # storing for every problem and solver
            vfalpha.append(
                {
                    "instance": instance,
                    "projection": projection,
                    "rotation": rotation,
                    "psstype": psstype,
                    "alpha": alpha,
                }
            )
    vfalpha = pd.DataFrame(vfalpha)

    # Creating one common dataprofile for all problems
    dp = []
    for projection, rotation, psstype in itertools.product([0, 1], [0, 1], [1, 2, 3]):
        for alpha in range(N + 1):

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

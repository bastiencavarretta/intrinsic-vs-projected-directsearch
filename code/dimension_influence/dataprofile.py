import glob
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools
import pickle
import matplotlib as mpl

if shutil.which("latex"):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cm"],
            "text.usetex": True,
            "figure.figsize": (6.4, 4.0),
        }
    )


def dataprofile(
    df, tau=1e-2, euclsimplex=0, lbdfoeuclsimplex=0, mdim_fixed=None, codim_fixed=None
):
    """Compute data profiles from a flat directsearch results DataFrame.

    Args:
        df (DataFrame): flat results DataFrame as produced by run.py.
            Must have columns: mdim, codim, mdim_rendering, codim_rendering,
            instance, pbtype, projection, rotation, psstype, euclsimplex,
            simplexbudget, vf, per_iteration_evaluations,
            and optionally rb_vf, rb_per_iteration_evaluations.
        tau (float): tolerance for "problem solved" (relative decrease).
        euclsimplex (int): 0 or 1. Budget scaling for the x-axis.
            0 = riemannian simplex gradients (scale by mdim+1),
            1 = euclidean simplex gradients (scale by adim+1).
        lbdfoeuclsimplex (int): 0 or 1. If 1, the lower bound proxy also
            includes the final values from the other budget type.

    Returns:
        DataFrame with columns:
            projection (int)   : 0 (intrinsic) or 1 (projected)
            rotation   (int)   : 0 or 1
            psstype    (int)   : 1, 2, or 3
            groupby    (str)   : "codim" | "mdim" | "tot"
            dim_value  (int)   : codim_rendering or mdim_rendering value;
                                 NaN for groupby="tot"
            curve (np.ndarray) : data profile values for alpha in 0..N
    """
    df = df.copy()
    if df["simplexbudget"].nunique() == 1:
        N = df["simplexbudget"].iloc[0]
    else:
        raise ValueError("Multiple simplex budgets found in DataFrame.")

    codims_rendering = sorted(df["codim_rendering"].unique())
    mdims_rendering = sorted(df["mdim_rendering"].unique())
    nbinstances = df["instance"].nunique()
    pbtypes = sorted(df["pbtype"].unique())
    nb_problems = len(pbtypes)
    ncodims = len(codims_rendering)
    nmdims = len(mdims_rendering)

    experiment_euclsimplex = df["euclsimplex"].iloc[0]

    if experiment_euclsimplex == 0:
        if euclsimplex == 1 or lbdfoeuclsimplex == 1:
            raise ValueError(
                "euclsimplex and lbdfoeuclsimplex cannot be 1 if the experiment was run with euclsimplex=0."
            )
        vfcol = "vf"
        evcol = "per_iteration_evaluations"
        lbdfo_vfcol = "vf"
    else:
        vfcol = "vf" if euclsimplex == 1 else "rb_vf"
        evcol = (
            "per_iteration_evaluations"
            if euclsimplex == 1
            else "rb_per_iteration_evaluations"
        )
        lbdfo_vfcol = "vf" if lbdfoeuclsimplex == 1 else "rb_vf"

    vfalpha = np.full((nb_problems, 2, 2, 3, ncodims, nmdims, nbinstances), np.nan)

    for nb_problem, (pbtype, subdf) in enumerate(df.groupby("pbtype")):

        def _lbdfo(grp):
            finals = list(grp[lbdfo_vfcol].apply(lambda x: x[-1]))
            return min(finals)

        lbdfo_series = subdf.groupby(["mdim", "codim", "instance", "rotation"]).apply(
            _lbdfo, include_groups=False
        )

        for _, row in subdf.iterrows():
            mdim, codim, k = row["mdim"], row["codim"], row["instance"]
            projection, rotation, psstype = (
                row["projection"],
                row["rotation"],
                row["psstype"],
            )

            imdim_rendering = list(mdims_rendering).index(row["mdim_rendering"])
            icodim_rendering = list(codims_rendering).index(row["codim_rendering"])
            scalingdim = mdim + codim if euclsimplex == 1 else mdim

            fvalues = row[vfcol]
            evaluations = np.cumsum(row[evcol])
            f0 = fvalues[0]
            lbdfo = lbdfo_series[(mdim, codim, k, rotation)]

            if abs(f0 - lbdfo) < tau:
                alpha = 0
            else:
                test_value = (fvalues - lbdfo) / (f0 - lbdfo)
                indices_solving = np.where(test_value <= tau)[0]
                if len(indices_solving) == 0:
                    alpha = np.nan
                else:
                    alpha = int(evaluations[np.min(indices_solving)] / (scalingdim + 1))

            vfalpha[
                nb_problem,
                projection,
                rotation,
                psstype - 1,
                icodim_rendering,
                imdim_rendering,
                k,
            ] = alpha

    alphas = np.arange(N + 1)
    records = []

    for projection, rotation, psstype in itertools.product([0, 1], [0, 1], [1, 2, 3]):
        ps = psstype - 1

        mdims_to_fix = [mdim_fixed] if mdim_fixed is not None else mdims_rendering
        for mdim_fixed_r in mdims_to_fix:
            imdim_fixed = list(mdims_rendering).index(mdim_fixed_r)
            for icodim, codim_r in enumerate(codims_rendering):
                vals = vfalpha[:, projection, rotation, ps, icodim, imdim_fixed, :]
                curve = np.array([np.sum(vals <= a) for a in alphas]) / (
                    nbinstances * nb_problems
                )
                records.append(
                    {
                        "projection": projection,
                        "rotation": rotation,
                        "psstype": psstype,
                        "groupby": "codimev",
                        "dim_value": int(codim_r),
                        "fixed_dim": int(mdim_fixed_r),
                        "curve": curve,
                    }
                )

        codims_to_fix = [codim_fixed] if codim_fixed is not None else codims_rendering
        for codim_fixed_r in codims_to_fix:
            icodim_fixed = list(codims_rendering).index(codim_fixed_r)
            for imdim, mdim_r in enumerate(mdims_rendering):
                vals = vfalpha[:, projection, rotation, ps, icodim_fixed, imdim, :]
                curve = np.array([np.sum(vals <= a) for a in alphas]) / (
                    nbinstances * nb_problems
                )
                records.append(
                    {
                        "projection": projection,
                        "rotation": rotation,
                        "psstype": psstype,
                        "groupby": "mdimev",
                        "dim_value": int(mdim_r),
                        "fixed_dim": int(codim_fixed_r),
                        "curve": curve,
                    }
                )

        vals = vfalpha[:, projection, rotation, ps, :, :, :]
        curve = np.array([np.sum(vals <= a) for a in alphas]) / (
            nbinstances * nb_problems * ncodims * nmdims
        )
        records.append(
            {
                "projection": projection,
                "rotation": rotation,
                "psstype": psstype,
                "groupby": "tot",
                "dim_value": np.nan,
                "fixed_dim": None,
                "curve": curve,
            }
        )

    return pd.DataFrame(records)


def plotting_dp(
    exppath,
    rotations=[0],
    psstypes=[1, 2, 3],
    projections=[0, 1],
    euclsimplex=0,
    lbdfoeuclsimplex=0,
    tau=1e-2,
    plottingworld="mdimev",  # "codimev", "mdimev", or "tot"
    fixed_dim=4,
    saving=False,
    dryrun=False,
):
    """Plot and save data profiles from a run.py experiment directory.

    Args:
        exppath (str): path to the experiment directory (contains results/).
        rotations (list): subset of {0, 1}.
        psstypes (list): subset of {1, 2, 3}.
        projections (list): subset of {0, 1}.
        euclsimplex (int): 0 or 1. Budget scaling for the x-axis.
        lbdfoeuclsimplex (int): 0 or 1. See dataprofile docstring.
        tau (float): tolerance for "problem solved".
        plottingworld (str): "codim", "mdim", or "tot" — which panels to plot.
        saving (bool): save plots to exppath/plots/.
        dryrun (bool): load only *_dryrun.pkl files and suffix output filenames with _dryrun.
    """
    all_pkl = glob.glob(os.path.join(exppath, "results", "*.pkl"))
    if dryrun:
        pkl_files = [f for f in all_pkl if "_dryrun" in os.path.basename(f)]
    else:
        pkl_files = [f for f in all_pkl if "_dryrun" not in os.path.basename(f)]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {exppath}/results/")
    df = pd.concat([pd.read_pickle(f) for f in pkl_files], ignore_index=True)
    df["codim_rendering"] = df["codim_rendering"].replace(1, 0)

    codims_rendering = [int(x) for x in sorted(df["codim_rendering"].unique())]
    mdims_rendering = [int(x) for x in sorted(df["mdim_rendering"].unique())]
    nbinstances = df["instance"].nunique()

    mdim_fixed_arg = fixed_dim if plottingworld in ("codimev", "all") else None
    codim_fixed_arg = fixed_dim if plottingworld in ("mdimev", "all") else None
    dp = dataprofile(
        df,
        tau=tau,
        euclsimplex=euclsimplex,
        lbdfoeuclsimplex=lbdfoeuclsimplex,
        mdim_fixed=mdim_fixed_arg,
        codim_fixed=codim_fixed_arg,
    )
    N = int(df["simplexbudget"].iloc[0])

    def _curve(projection, rotation, psstype, groupby, dim_value=None):
        mask = (
            (dp["projection"] == projection)
            & (dp["rotation"] == rotation)
            & (dp["psstype"] == psstype)
            & (dp["groupby"] == groupby)
            & (dp["fixed_dim"] == fixed_dim)
        )
        if dim_value is not None:
            mask &= dp["dim_value"] == dim_value
        return dp.loc[mask, "curve"].iloc[0]

    titlefonts = 28
    subtitle_fonts = 24
    label_fonts = 12
    cmapproj = mpl.colormaps["Set1"].colors[1:4]
    cmapnoproj = mpl.colormaps["Set1"].colors[1:4]
    cmap = [cmapnoproj, cmapproj]
    linestyles = ["-", "--"]

    exp_id = os.path.basename(os.path.normpath(exppath))

    dr_suffix = "_dryrun" if dryrun else ""

    def _figname(suffix):
        name = (
            f"{exp_id}_{suffix}"
            f"_proj{projections}_pss{psstypes}_rot{rotations}"
            f"_es{euclsimplex}_tau{tau}_mdims{mdims_rendering}_codims{codims_rendering}_nbi{nbinstances}_rotawareproxy{dr_suffix}.pdf"
        )
        return name.replace(" ", "").replace("[", "").replace("]", "")

    if saving:
        plots_dir = os.path.join(exppath, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    if plottingworld in ("codimev", "all"):
        plotnbr = len(codims_rendering)
        linenbr = plotnbr // 3 + (1 if plotnbr % 3 != 0 else 0)
        fig, ax = plt.subplots(linenbr, 3, figsize=(15, 4 * linenbr))
        ax = ax.flatten()
        for k in range(3 * linenbr - plotnbr):
            fig.delaxes(ax[plotnbr + k])

        for icodim, codim_r in enumerate(codims_rendering):
            ax[icodim].set_title("n-m = {:}".format(codim_r), fontsize=subtitle_fonts)
            ax[icodim].grid(True)
            ax[icodim].set_xlim(0, N)
            ax[icodim].set_ylim(0, 1)
            for rotation, projection, psstype in itertools.product(
                rotations, projections, psstypes
            ):
                variant = "intr" if projection == 0 else "proj"
                kwargs = dict(
                    color=cmap[projection][psstype - 1],
                    linestyle=linestyles[projection],
                    linewidth=2,
                )
                if icodim == len(codims_rendering) - 1:
                    ax[icodim].plot(
                        _curve(projection, rotation, psstype, "codimev", codim_r),
                        label=f"PSS{psstype} ({variant})",
                        **kwargs,
                    )
                else:
                    ax[icodim].plot(
                        _curve(projection, rotation, psstype, "codimev", codim_r),
                        **kwargs,
                    )

        fig.supxlabel("Number of simplex gradient evaluations", fontsize=titlefonts)
        fig.supylabel("Ratio of problems solved", fontsize=titlefonts)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.tight_layout()
        if saving:
            path = os.path.join(plots_dir, _figname("codimsdp"))
            plt.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved: {path}")
        # plt.show()

    if plottingworld in ("mdimev", "all"):
        plotnbr = len(mdims_rendering)
        linenbr = plotnbr // 3 + (1 if plotnbr % 3 != 0 else 0)
        fig, ax = plt.subplots(linenbr, 3, figsize=(15, 4 * linenbr))
        ax = ax.flatten()
        for k in range(3 * linenbr - plotnbr):
            fig.delaxes(ax[plotnbr + k])

        for imdim, mdim_r in enumerate(mdims_rendering):
            ax[imdim].set_title("m = {:}".format(mdim_r), fontsize=subtitle_fonts)
            ax[imdim].grid(True)
            ax[imdim].set_xlim(0, N)
            ax[imdim].set_ylim(0, 1)
            for rotation, projection, psstype in itertools.product(
                rotations, projections, psstypes
            ):
                variant = "intr" if projection == 0 else "proj"
                kwargs = dict(
                    color=cmap[projection][psstype - 1],
                    linestyle=linestyles[projection],
                    linewidth=2,
                )
                if imdim == len(mdims_rendering) - 1:
                    ax[imdim].plot(
                        _curve(projection, rotation, psstype, "mdimev", mdim_r),
                        label=f"PSS{psstype} ({variant})",
                        **kwargs,
                    )
                else:
                    ax[imdim].plot(
                        _curve(projection, rotation, psstype, "mdimev", mdim_r),
                        **kwargs,
                    )
        fig.supxlabel("Number of simplex gradient evaluations", fontsize=titlefonts)
        fig.supylabel("Ratio of problems solved", fontsize=titlefonts)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.tight_layout()
        if saving:
            path = os.path.join(plots_dir, _figname("mdimsdp"))
            
            plt.savefig(path, bbox_inches="tight", dpi=300)
            print(f"Saved: {path}")

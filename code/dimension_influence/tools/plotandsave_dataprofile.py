import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import dill
import matplotlib as mpl


def dataprofile(vdsresults, vproblems, tau=1e-2, euclsimplex=0, lbdfoeuclsimplex=0, N=100):
    """Compute data profiles from directsearch experiment DataFrames.

    Args:
        vdsresults (list of DataFrame) : one DataFrame per (maniftype, obj) family,
            as returned by perform_ds.
        vproblems (list of dict) : one problems_meta dict per family.
        tau (float) : tolerance for "problem solved" (relative decrease).
        euclsimplex (int) : 0 or 1. Selects the budget scaling for the x-axis.
            0 = riemannian simplex gradients (scale by mdim+1),
            1 = euclidean simplex gradients (scale by adim+1).
        lbdfoeuclsimplex (int) : 0 or 1. If 1, the lower bound proxy also includes
            the final values from the other budget type (tighter bound). If 0, only
            the budget type selected by euclsimplex is used.
        N (int) : maximum budget in simplex gradients for the x-axis.

    Returns:
        dict with keys:
            "dp_codim" : shape (2, 2, 3, ncodims, N+1) — fixed mdim (index 1), varying codim
            "dp_mdim"  : shape (2, 2, 3, nmdims,  N+1) — fixed codim (index 2), varying mdim
            "dp_tot"   : shape (2, 2, 3, N+1)          — all problems aggregated
    """
    codims = vproblems[0]["codims"]
    mdims = vproblems[0]["mdims"]
    nbinstances = vproblems[0]["nbinstances"]
    nb_problems = len(vproblems)
    ncodims = len(codims)
    nmdims = len(mdims)

    # Determine which result columns to use based on euclsimplex target and how
    # experiments were run (stored in the "euclsimplex" column of the DataFrame).
    experiment_euclsimplex = vdsresults[0]["euclsimplex"].iloc[0]
    if euclsimplex == 1:
        vf_col = "vf"
        ev_col = "per_iteration_evaluations"
    else:
        if experiment_euclsimplex == 1:
            vf_col = "rb_vf"
            ev_col = "rb_per_iteration_evaluations"
        else:
            vf_col = "vf"
            ev_col = "per_iteration_evaluations"

    # vfalpha[nb_problem, projection, rotation, psstype-1, icodim, imdim, k] = alpha
    # alpha = number of simplex gradients to solve the problem, or nan if unsolved.
    vfalpha = np.full((nb_problems, 2, 2, 3, ncodims, nmdims, nbinstances), np.nan)

    for nb_problem, df in enumerate(vdsresults):
        # Lower bound proxy: min final function value across all variants per instance.
        # lbdfoeuclsimplex=1 also pulls in the other budget's final values for a tighter bound.
        other_col = "vf" if vf_col == "rb_vf" else "rb_vf"

        def _lbdfo(grp):
            finals = list(grp[vf_col].apply(lambda x: x[-1]))
            if lbdfoeuclsimplex == 1 and other_col in grp.columns:
                finals += list(grp[other_col].apply(lambda x: x[-1]))
            return min(finals)

        lbdfo_series = df.groupby(["mdim", "codim", "k"]).apply(_lbdfo)

        for _, row in df.iterrows():
            mdim, codim, k = row["mdim"], row["codim"], row["k"]
            projection, rotation, psstype = row["projection"], row["rotation"], row["psstype"]
            imdim = list(mdims).index(mdim)
            icodim = list(codims).index(codim)
            scalingdim = mdim + codim if euclsimplex == 1 else mdim

            fvalues = row[vf_col]
            evaluations = np.cumsum(row[ev_col])
            f0 = fvalues[0]
            lbdfo = lbdfo_series[(mdim, codim, k)]

            if abs(f0 - lbdfo) < 1e-15:
                alpha = 0
            else:
                test_value = (fvalues - lbdfo) / (f0 - lbdfo)
                indices_solving = np.where(test_value <= tau)[0]
                if len(indices_solving) == 0:
                    alpha = np.nan
                else:
                    alpha = int(evaluations[np.min(indices_solving)] / (scalingdim + 1))

            vfalpha[nb_problem, projection, rotation, psstype - 1, icodim, imdim, k] = alpha

    dp_codim = np.zeros((2, 2, 3, ncodims, N + 1))
    dp_mdim = np.zeros((2, 2, 3, nmdims, N + 1))
    dp_tot = np.zeros((2, 2, 3, N + 1))

    imdim_fixed = 1   # fixed manifold dimension index for dp_codim
    icodim_fixed = 2  # fixed codimension index for dp_mdim

    for projection, rotation, psstype, icodim in itertools.product(
        [0, 1], [0, 1], [1, 2, 3], range(ncodims)
    ):
        for alpha in range(N + 1):
            dp_codim[projection, rotation, psstype - 1, icodim, alpha] = np.sum(
                vfalpha[:, projection, rotation, psstype - 1, icodim, imdim_fixed, :] <= alpha
            ) / (nbinstances * nb_problems)

    for projection, rotation, psstype, imdim in itertools.product(
        [0, 1], [0, 1], [1, 2, 3], range(nmdims)
    ):
        for alpha in range(N + 1):
            dp_mdim[projection, rotation, psstype - 1, imdim, alpha] = np.sum(
                vfalpha[:, projection, rotation, psstype - 1, icodim_fixed, imdim, :] <= alpha
            ) / (nbinstances * nb_problems)

    for projection, rotation, psstype in itertools.product([0, 1], [0, 1], [1, 2, 3]):
        for alpha in range(N + 1):
            dp_tot[projection, rotation, psstype - 1, alpha] = np.sum(
                vfalpha[:, projection, rotation, psstype - 1, :, :, :] <= alpha
            ) / (nbinstances * nb_problems * ncodims * nmdims)

    return {"dp_codim": dp_codim, "dp_mdim": dp_mdim, "dp_tot": dp_tot}


def plotting_dp(
    expnumber,
    maniftypeobj=[(1, 2)],
    rotations=[0],
    psstypes=[1, 2, 3],
    projections=[0, 1],
    euclsimplex=0,
    lbdfoeuclsimplex=0,
    tau=1e-2,
    N=100,
    plotcodimsev=True,
    plotmdimsev=True,
    saving=False,
):
    """Plot and save data profiles from directsearch experiments.

    Args:
        expnumber (int) : experiment index (same as saveperform_ds).
        maniftypeobj (list of tuples) : (maniftype, obj) pairs to aggregate.
        rotations (list) : subset of {0, 1}.
        psstypes (list) : subset of {1, 2, 3}.
        projections (list) : subset of {0, 1}.
        euclsimplex (int) : 0 or 1. Budget scaling for the x-axis.
        lbdfoeuclsimplex (int) : 0 or 1. See dataprofile docstring.
        tau (float) : tolerance for "problem solved".
        N (int) : maximum simplex gradient evaluations on the x-axis.
        plotcodimsev (bool) : plot data profiles for varying codim (fixed mdim).
        plotmdimsev (bool) : plot data profiles for varying mdim (fixed codim).
        saving (bool) : save plots to tables_and_plots/dataprofiles/.
    """
    titlefonts = 28
    subtitle_fonts = 24
    label_fonts = 12
    cmapproj = mpl.colormaps["Set1"].colors[1:4]
    cmapnoproj = mpl.colormaps["Set1"].colors[1:4]
    cmap = [cmapnoproj, cmapproj]
    linestyles = ["-", "--"]

    vdsresults = []
    vproblems = []
    for maniftype, obj in maniftypeobj:
        nbr = (
            "exp" + str(expnumber)
            + "_maniftype" + str(maniftype)
            + "_obj" + str(obj) + "_"
        )
        pathdsresults = "dsresults_folder/" + nbr + "dsresults.pkl"
        pathproblems = "dsresults_folder/" + nbr + "problems.pkl"
        with open(pathdsresults, "rb") as f:
            dsresults = pickle.load(f)
        with open(pathproblems, "rb") as f:
            problems_meta = dill.load(f)
        vdsresults.append(dsresults)
        vproblems.append(problems_meta)

    mdims = vproblems[0]["mdims"]
    codims = vproblems[0]["codims"]
    nbinstances = vproblems[0]["nbinstances"]

    dps = dataprofile(vdsresults, vproblems, tau=tau, euclsimplex=euclsimplex, lbdfoeuclsimplex=lbdfoeuclsimplex, N=N)
    dp_codim, dp_mdim, dp_tot = dps["dp_codim"], dps["dp_mdim"], dps["dp_tot"]

    def _figname(suffix):
        path_problems = "".join(f"{mto[0]}-{mto[1]}_" for mto in maniftypeobj)
        name = (
            f"exp{expnumber}_manifobj{path_problems}{suffix}"
            f"_param-proj-{projections}_psstypes{psstypes}_rot{rotations}"
            f"_es{euclsimplex}_tau{tau}_mdims{mdims}_codims{codims}_nbi{nbinstances}.pdf"
        )
        return name.replace(" ", "").replace("[", "").replace("]", "")

    if plotcodimsev:
        plotnbr = len(codims)
        linenbr = plotnbr // 3 + (1 if plotnbr % 3 != 0 else 0)
        fig, ax = plt.subplots(linenbr, 3, figsize=(15, 4 * linenbr))
        ax = ax.flatten()
        for k in range(3 * linenbr - plotnbr):
            fig.delaxes(ax[plotnbr + k])

        for icodim, codim in enumerate(codims):
            ax[icodim].set_title("n-m = {:}".format(codim), fontsize=subtitle_fonts)
            ax[icodim].grid(True)
            ax[icodim].set_xlim(0, N)
            ax[icodim].set_ylim(0, 1)
            for rotation, projection, psstype in itertools.product(rotations, projections, psstypes):
                variant = "intr" if projection == 0 else "proj"
                kwargs = dict(
                    color=cmap[projection][psstype - 1],
                    linestyle=linestyles[projection],
                    linewidth=2,
                )
                if icodim == len(codims) - 1:
                    ax[icodim].plot(
                        dp_codim[projection, rotation, psstype - 1, icodim, :],
                        label=f"PSS{psstype} ({variant})", **kwargs,
                    )
                    ax[icodim].legend(fontsize=label_fonts)
                else:
                    ax[icodim].plot(dp_codim[projection, rotation, psstype - 1, icodim, :], **kwargs)

        fig.supxlabel("Number of simplex gradient evaluations", fontsize=titlefonts)
        fig.supylabel("Ratio of problems solved", fontsize=titlefonts)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.tight_layout()
        if saving:
            plt.savefig("tables_and_plots/dataprofiles/" + _figname("codimsdp"), bbox_inches="tight", dpi=300)
        plt.show()

    if plotmdimsev:
        plotnbr = len(mdims)
        linenbr = plotnbr // 3 + (1 if plotnbr % 3 != 0 else 0)
        fig, ax = plt.subplots(linenbr, 3, figsize=(15, 4 * linenbr))
        ax = ax.flatten()
        for k in range(3 * linenbr - plotnbr):
            fig.delaxes(ax[k])

        for iimdim, mdim in enumerate(mdims):
            imdim = iimdim + 1
            ax[imdim].set_title("m = {:}".format(mdim), fontsize=subtitle_fonts)
            ax[imdim].grid(True)
            ax[imdim].set_xlim(0, N)
            ax[imdim].set_ylim(0, 1)
            for rotation, projection, psstype in itertools.product(rotations, projections, psstypes):
                variant = "intr" if projection == 0 else "proj"
                kwargs = dict(
                    color=cmap[projection][psstype - 1],
                    linestyle=linestyles[projection],
                    linewidth=2,
                )
                if imdim == len(mdims):
                    ax[imdim].plot(
                        dp_mdim[projection, rotation, psstype - 1, iimdim, :],
                        label=f"PSS{psstype} ({variant})", **kwargs,
                    )
                    ax[imdim].legend(fontsize=label_fonts)
                else:
                    ax[imdim].plot(dp_mdim[projection, rotation, psstype - 1, iimdim, :], **kwargs)

        ax[0].axis("off")
        fig.supxlabel("Number of simplex gradient evaluations", fontsize=titlefonts)
        fig.supylabel("Ratio of problems solved", fontsize=titlefonts)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.tight_layout()
        if saving:
            plt.savefig("tables_and_plots/dataprofiles/" + _figname("mdimsdp"), bbox_inches="tight", dpi=300)
        plt.show()

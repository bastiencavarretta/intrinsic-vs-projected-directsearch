import itertools
import pickle
import dill
import pandas as pd

pd.set_option("display.float_format", "{:.2f}".format)


def performancetable(
    expnumber,
    maniftypeobj=[(1, 2)],
    rotations=[0, 1],
    psstypes=[1, 2, 3],
    euclsimplexs=[0, 1],
    printing=True,
    saving=False,
):
    """Compute and print/save performance tables for projected vs intrinsic methods.

    Args:
        expnumber (int) : experiment index (same as saveperform_ds).
        maniftypeobj (list of tuples) : (maniftype, obj) pairs to aggregate.
        rotations (list) : subset of {0, 1}.
        psstypes (list) : subset of {1, 2, 3}.
        euclsimplexs (list) : subset of {0, 1}. Selects which budget scaling to use.
            0 = riemannian (use rb_vf if available, else vf),
            1 = euclidean (use vf).
        printing (bool) : print tables if True.
        saving (bool) : save LaTeX tables to tables_and_plots/performance_tables/ if True.
    """
    vdsresults = []
    vproblems = []
    for maniftype, obj in maniftypeobj:
        nbr = (
            "exp" + str(expnumber)
            + "_maniftype" + str(maniftype)
            + "_obj" + str(obj) + "_"
        )
        with open("dsresults_folder/" + nbr + "dsresults.pkl", "rb") as f:
            vdsresults.append(pickle.load(f))
        with open("dsresults_folder/" + nbr + "problems.pkl", "rb") as f:
            vproblems.append(dill.load(f))

    mdims = vproblems[0]["mdims"]
    codims = vproblems[0]["codims"]
    nbinstances = vproblems[0]["nbinstances"]
    nb_problems = len(maniftypeobj)

    # Concatenate all problem families into one DataFrame, tagged by family index.
    df_all = pd.concat(
        [df.assign(nb_problem=i) for i, df in enumerate(vdsresults)],
        ignore_index=True,
    )
    experiment_euclsimplex = df_all["euclsimplex"].iloc[0]

    tablekeypd = pd.MultiIndex.from_product(
        [mdims, codims, euclsimplexs, [0, 1], [1, 2, 3]],
        names=["mdim", "codim", "euclsimplex", "rotation", "psstype"],
    )
    table = pd.DataFrame(index=tablekeypd, columns=["ratio"])

    for mdim, codim in itertools.product(mdims, codims):
        df_dim = df_all[(df_all.mdim == mdim) & (df_all.codim == codim)]
        for euclsimplex, rotation, psstype in itertools.product(euclsimplexs, [0, 1], [1, 2, 3]):
            # Select which final value to compare
            if euclsimplex == 1:
                vf_col = "vf"
            else:
                vf_col = "rb_vf" if (experiment_euclsimplex == 1 and "rb_vf" in df_all.columns) else "vf"

            df_proj = df_dim[
                (df_dim.projection == 1) &
                (df_dim.rotation == rotation) &
                (df_dim.psstype == psstype)
            ].set_index(["nb_problem", "k"])

            df_intr = df_dim[
                (df_dim.projection == 0) &
                (df_dim.rotation == rotation) &
                (df_dim.psstype == psstype)
            ].set_index(["nb_problem", "k"])

            count = 0
            for idx in df_proj.index:
                gap = df_proj.loc[idx, vf_col][-1] - df_intr.loc[idx, vf_col][-1]
                if gap > 0:
                    count += 1
            table.loc[(mdim, codim, euclsimplex, rotation, psstype), "ratio"] = (
                count / (nbinstances * nb_problems)
            )

    path_problems = "".join(f"{mto[0]}-{mto[1]}_" for mto in maniftypeobj)

    for rotation, euclsimplex, psstype in itertools.product(rotations, euclsimplexs, psstypes):
        subtable = (
            table.xs((euclsimplex, rotation, psstype), level=("euclsimplex", "rotation", "psstype"))
            .reset_index()
            .pivot(index="codim", columns="mdim")
        )
        if saving:
            latex_code = subtable.to_latex(
                index=True,
                float_format="%.2f".__mod__,
                column_format="l|" + len(mdims) * "c",
                caption=None,
                label=None,
                position=0,
                header=True,
            )
            path = (
                "tables_and_plots/performance_tables/"
                + f"exp{expnumber}_manifobj{path_problems}"
                + f"rotation{rotation}_euclideansimplex{euclsimplex}_psstype{psstype}.tex"
            )
            with open(path, "w") as f:
                f.write(latex_code)
        if printing:
            print(f"\nrotation = {rotation}, euclsimplex = {euclsimplex}, psstype = {psstype}")
            print(subtable)

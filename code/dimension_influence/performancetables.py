"""performancetables.py — Generate performance tables from run.py results.

Each table cell (codim, mdim) is the fraction of problem instances where the
intrinsic method (projection=0) achieves a strictly lower final function value
than the projected method (projection=1) — i.e. how often intrinsic wins.

Usage (CLI):
    python performancetables.py path/to/results.pkl
    python performancetables.py path/to/results.pkl --save --savedir tables/
    python performancetables.py path/to/results.pkl --euclsimplexs 0 1 --psstypes 1 2
"""
import argparse
import itertools
import os

import pandas as pd

pd.set_option("display.float_format", "{:.2f}".format)


def _resolve_vf_col(df, euclsimplex_req):
    """Return (vf_col, euclsimplex_filter) for the requested budget type.

    euclsimplex_req=1 (euclidean budget): use vf from rows where euclsimplex=1.
    euclsimplex_req=0 (riemannian budget): prefer rb_vf from euclsimplex=1 rows
        (these were recorded up to the riemannian budget during the euclidean-budget
        run); fall back to vf from euclsimplex=0 rows if rb_vf is absent.
    Returns None if the requested budget type has no data available.
    """
    if euclsimplex_req == 1:
        if (df["euclsimplex"] == 1).any():
            return "vf", 1
        return None
    else:
        if "rb_vf" in df.columns and (df["euclsimplex"] == 1).any():
            return "rb_vf", 1
        if (df["euclsimplex"] == 0).any():
            return "vf", 0
        return None


def performancetable(
    df,
    rotations=None,
    psstypes=None,
    euclsimplexs=None,
    printing=True,
    saving=False,
    savedir="tables_and_plots/performance_tables/",
    exp_label="",
):
    """Compute and print/save performance tables for projected vs intrinsic methods.

    Args:
        df: DataFrame produced by run.py, or a path (str/Path) to a .pkl file.
        rotations: rotation values to include (default: all found in df).
        psstypes: psstype values to include (default: all found in df).
        euclsimplexs: budget types to tabulate — 1=euclidean, 0=riemannian.
            Defaults to all euclsimplex values present in df; if rb_vf is
            available, riemannian budget (0) is also offered from euclsimplex=1 runs.
        printing: print tables to stdout if True.
        saving: write LaTeX .tex files to savedir if True.
        savedir: directory for LaTeX output (created if absent).
        exp_label: filename prefix for saved tables.
    """
    if isinstance(df, (str, os.PathLike)):
        df = pd.read_pickle(df)

    mdims = sorted(df["mdim"].unique())
    codims = sorted(df["codim_rendering"].unique())

    if rotations is None:
        rotations = sorted(df["rotation"].unique())
    if psstypes is None:
        psstypes = sorted(df["psstype"].unique())
    if euclsimplexs is None:
        euclsimplexs = sorted(df["euclsimplex"].unique().tolist())
        if "rb_vf" in df.columns and (df["euclsimplex"] == 1).any() and 0 not in euclsimplexs:
            euclsimplexs = [0] + euclsimplexs

    for euclsimplex_req in euclsimplexs:
        resolved = _resolve_vf_col(df, euclsimplex_req)
        if resolved is None:
            print(f"[performancetable] euclsimplex={euclsimplex_req}: no data, skipping.")
            continue
        vf_col, euclsimplex_filter = resolved
        df_bud = df[df["euclsimplex"] == euclsimplex_filter]

        for rotation, psstype in itertools.product(rotations, psstypes):
            rows = []
            for codim in codims:
                row = {"codim": codim}
                for mdim in mdims:
                    df_cell = df_bud[
                        (df_bud["mdim"] == mdim)
                        & (df_bud["codim_rendering"] == codim)
                        & (df_bud["rotation"] == rotation)
                        & (df_bud["psstype"] == psstype)
                    ]

                    df_proj = df_cell[df_cell["projection"] == 1].set_index("idpb")
                    df_intr = df_cell[df_cell["projection"] == 0].set_index("idpb")

                    # Match on idpb: same problem instance, different projection strategy.
                    common_ids = df_proj.index.intersection(df_intr.index)
                    if len(common_ids) == 0:
                        row[mdim] = float("nan")
                        continue

                    n_intr_wins = sum(
                        df_proj.loc[i, vf_col][-1] > df_intr.loc[i, vf_col][-1]
                        for i in common_ids
                    )
                    row[mdim] = n_intr_wins / len(common_ids)

                rows.append(row)

            subtable = pd.DataFrame(rows).set_index("codim")
            subtable.columns.name = "mdim"

            if printing:
                print(
                    f"\nrotation={rotation}, euclsimplex={euclsimplex_req},"
                    f" psstype={psstype}  [vf_col={vf_col}]"
                )
                print(subtable)

            if saving:
                os.makedirs(savedir, exist_ok=True)
                latex_code = subtable.to_latex(
                    index=True,
                    float_format="%.2f".__mod__,
                    column_format="l|" + len(mdims) * "c",
                    caption=None,
                    label=None,
                    header=True,
                )
                fname = (
                    f"{exp_label}"
                    f"rotation{rotation}_euclideansimplex{euclsimplex_req}_psstype{psstype}.tex"
                )
                with open(os.path.join(savedir, fname), "w") as f:
                    f.write(latex_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate performance tables from run.py results."
    )
    parser.add_argument(
        "pkl", type=str, help="Path to results .pkl file produced by run.py."
    )
    parser.add_argument(
        "--rotations",
        type=int,
        nargs="+",
        default=None,
        help="Rotation values to tabulate (default: all in data).",
    )
    parser.add_argument(
        "--psstypes",
        type=int,
        nargs="+",
        default=None,
        help="PSS types to tabulate (default: all in data).",
    )
    parser.add_argument(
        "--euclsimplexs",
        type=int,
        nargs="+",
        default=None,
        help="Budget types: 1=euclidean, 0=riemannian (default: all available).",
    )
    parser.add_argument("--no-print", action="store_true", help="Suppress console output.")
    parser.add_argument("--save", action="store_true", help="Save LaTeX tables to disk.")
    parser.add_argument(
        "--savedir",
        type=str,
        default="tables_and_plots/performance_tables/",
        help="Directory for LaTeX output.",
    )
    parser.add_argument(
        "--label", type=str, default="", help="Filename prefix for saved tables."
    )
    args = parser.parse_args()

    performancetable(
        df=args.pkl,
        rotations=args.rotations,
        psstypes=args.psstypes,
        euclsimplexs=args.euclsimplexs,
        printing=not args.no_print,
        saving=args.save,
        savedir=args.savedir,
        exp_label=args.label,
    )

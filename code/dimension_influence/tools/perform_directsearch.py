import numpy as np
import time
import pandas as pd
from pymanopt import manifolds as man
import pickle
import dill

from code.dimension_influence.problems.ProblemEigh import ProblemEigh
from code.dimension_influence.problems.ProblemLinearSubspace import ProblemLinearSubspace


def directsearch(
    problem,
    simplexbudget=100,
    projection=1,
    psstype=1,
    rotation=1,
    euclsimplex=1,
    renormalize_tangent_vec=True,
    gamma=0.5,
    Gamma=2,
    alpha_0=1.0,
    alpha_max=1.0,
    c=1.0,
    eps=0,
    itmax=np.inf,
    printing=False,
):
    """directsearch algorithm on a Riemannian manifold.
    Args:
        problem (Problem object in tools.problems.py)
        simplexbudget (int) : number of simplex gradients. Actual budget = simplexbudget*(adim+1) if euclsimplex=1, simplexbudget*(mdim+1) if euclsimplex=0.
        projection (int) : 0 or 1. See the method "build_pss" in the tools.problems.py file.
        psstype (int) : 1 or 2 or 3. See the method "build_pss" in the tools.problems.py file.
        rotation (int) : 0 or 1. See the method "build_pss" in the tools.problems.py file.
        euclsimplex (int) : 0 or 1. If 1, budget is scaled by adim+1 and rb_* keys are added to the result for the riemannian sub-budget run. If 0, budget is scaled by mdim+1.
        renormalize_tangent_vec (bool) : See the argument "renormalize" of method "build_pss" in the tools.problems.py file.
        gamma, Gamma, alpha_0, alpha_max, c (float): Direct-search stepsize update parameters.
        eps (float) : minimum stepsize before stopping
        itmax (int) : maximum number of iterations
        printing (bool) : prints progression if True
    Returns:
        A dict with keys: "vf", "per_iteration_evaluations", "valpha", "stopcriterion",
        "last_iterate". When euclsimplex=1, also includes rb_* keys for the riemannian
        sub-budget run.
    """
    n, m, x0, f, manifold, f_value = (
        problem.adim,
        problem.mdim,
        problem.xstart,
        problem.costf,
        problem.manifold,
        problem.fstart,
    )

    x = x0.copy()
    if euclsimplex == 1:
        budget = (n + 1) * simplexbudget
    elif euclsimplex == 0:
        budget = (m + 1) * simplexbudget
    else:
        raise ValueError("euclsimplex must be 0 or 1")
    rb = (m + 1) * simplexbudget

    alpha = alpha_0
    vf = [f(x0)]
    vevperit = [1]
    valphas = [alpha]
    print("----starting direct-search----") if printing else None
    k = 1
    evaluation_number = 1
    stopcriterion = None
    rb_vf, rb_vevperit, rb_valphas, rb_x, rb_stopcriterion = None, None, None, None, None

    while k < itmax and alpha > eps and evaluation_number < budget:
        success = False
        pss_at_x = problem.build_pss(
            x,
            projection=projection,
            psstype=psstype,
            rotation=rotation,
            renormalize=renormalize_tangent_vec,
        )
        i = 0
        nevaluation_for_k = 0
        while i < len(pss_at_x) and success == False:
            p = pss_at_x[i]
            if type(p) == list or isinstance(p, tuple):
                scaled_p = [alpha * comp for comp in p]
            else:
                scaled_p = alpha * p
            x_poll = manifold.exp(x, scaled_p)
            f_poll = f(x_poll)
            evaluation_number += 1
            if f_poll <= f_value - c * alpha**2 * problem.anorm(p) ** 2:
                x = x_poll
                f_value = f_poll
                alpha = min(alpha_max, Gamma * alpha)
                success = True
            i += 1
            nevaluation_for_k += 1
        if not success:
            alpha = min(alpha_max, gamma * alpha)
        vevperit.append(nevaluation_for_k)
        vf.append(f_value)
        valphas.append(alpha)

        if euclsimplex == 1:
            if evaluation_number <= rb or k == 1:
                rb_vf = vf.copy()
                rb_vevperit = vevperit.copy()
                rb_valphas = valphas.copy()
                rb_x = x.copy()
        k += 1

    if k >= itmax:
        stopcriterion = "full iterations used"
    elif alpha <= eps:
        stopcriterion = "low step size"
    elif evaluation_number >= budget:
        stopcriterion = "full budget used"
    else:
        raise ValueError("directsearch exited loop without a known stop criterion")

    if stopcriterion in ("full iterations used", "low step size"):
        rb_stopcriterion = stopcriterion
    else:
        rb_stopcriterion = "full budget used"

    print("----direct-search finished----") if printing else None

    result = {
        "vf": np.array(vf),
        "per_iteration_evaluations": vevperit,
        "valpha": valphas,
        "stopcriterion": stopcriterion,
        "last_iterate": x,
    }

    if euclsimplex == 1:
        result |= {
            "rb_vf": np.array(rb_vf),
            "rb_per_iteration_evaluations": rb_vevperit,
            "rb_valpha": rb_valphas,
            "rb_stopcriterion": rb_stopcriterion,
            "rb_last_iterate": rb_x,
        }

    return result


def perform_ds(
    mdims,
    codims,
    nbinstances,
    N=100,
    maniftype=1,
    obj=1,
    euclsimplex=1,
    printingprogression=True,
):
    """Run directsearch experiments and return results as a flat DataFrame.
    Args:
        mdims (list of int) : list of manifold dimensions
        codims (list of int) : list of manifold codimensions
        nbinstances (int) : number of problem instances per (mdim, codim) pair
        N (int) : simplexbudget passed to directsearch
        maniftype (int) : type of manifold (1, 1.5, or 2)
        obj (int) : type of objective function
        euclsimplex (int) : 0 or 1. See directsearch docstring.
        printingprogression (bool) : prints timing info if True
    Returns:
        problems_meta (dict) : mdims, codims, nbinstances, nested problems list
        dsresults (DataFrame) : one row per (mdim, codim, k, projection, rotation, psstype) run
    """
    adimsmdims = [[[codim + mdim, mdim] for codim in codims] for mdim in mdims]
    maxi, maxj = len(adimsmdims), len(adimsmdims[0])

    if maniftype in [1, 1.5]:
        problems = [
            [
                [
                    ProblemLinearSubspace(
                        (
                            man.Stiefel(
                                adimsmdims[i][j][0], adimsmdims[i][j][1]
                            ).random_point()
                            if maniftype == 1
                            else np.eye(adimsmdims[i][j][0], adimsmdims[i][j][1])
                        ),
                        obj=obj,
                    )
                    for _ in range(nbinstances)
                ]
                for j in range(maxj)
            ]
            for i in range(maxi)
        ]
    elif maniftype == 2:
        problems = [
            [
                [
                    ProblemEigh(adimsmdims[i][j][0], adimsmdims[i][j][1])
                    for _ in range(nbinstances)
                ]
                for j in range(maxj)
            ]
            for i in range(maxi)
        ]

    rows = []
    fullt0 = time.time()
    t0 = fullt0
    count = 0

    for imdim, mdim in enumerate(mdims):
        for icodim, codim in enumerate(codims):
            adim = mdim + codim
            for k in range(nbinstances):
                problem = problems[imdim][icodim][k]
                for projection in [0, 1]:
                    for rotation in [0, 1]:
                        for psstype in [1, 2, 3]:
                            dsresult = directsearch(
                                problem,
                                simplexbudget=N,
                                projection=projection,
                                rotation=rotation,
                                psstype=psstype,
                                euclsimplex=euclsimplex,
                                eps=0,
                            )
                            rows.append({
                                "mdim": mdim,
                                "codim": codim,
                                "adim": adim,
                                "k": k,
                                "projection": projection,
                                "rotation": rotation,
                                "psstype": psstype,
                                "euclsimplex": euclsimplex,
                                **dsresult,
                            })

            if printingprogression:
                count += 1
                t1 = time.time()
                chours = int((t1 - fullt0) // 3600)
                cminutes = int(((t1 - fullt0) % 3600) // 60)
                cseconds = int(((t1 - fullt0) % 3600) % 60)
                print(
                    "newdim : {:} out of {:}, cumulated_time = {:<2}:{:<2}:{:<2}, newtime = {:.2f}s".format(
                        count,
                        len(mdims) * len(codims),
                        chours,
                        cminutes,
                        cseconds,
                        t1 - t0,
                    )
                )
                t0 = t1

    dsresults = pd.DataFrame(rows)
    problems_meta = {
        "mdims": list(mdims),
        "codims": list(codims),
        "nbinstances": nbinstances,
        "problems": problems,
    }
    return problems_meta, dsresults


def saveperform_ds(
    expnumber,
    mdims,
    codims,
    N,
    nbinstances,
    maniftype,
    obj,
    euclsimplex=1,
    saving=False,
):
    """Performs and saves directsearch experiments.
    Args:
        expnumber (int) : experiment number (for saving purposes)
        mdims (list of int) : list of manifold dimensions
        codims (list of int) : list of manifold codimensions
        N (int) : simplexbudget passed to directsearch
        nbinstances (int) : number of problem instances per (mdim, codim) pair
        maniftype (int) : type of manifold
        obj (int) : type of objective
        euclsimplex (int) : 0 or 1. See directsearch docstring.
        saving (bool) : saves results if True, dry run otherwise.
    Note:
        Saves the DataFrame to dsresults_folder/ with pickle, problems_meta with dill.
    """
    nbr = "exp" + str(expnumber) + "_maniftype" + str(maniftype) + "_obj" + str(obj) + "_"
    print("computing :", nbr, "###########################################################")
    problems_meta, dsresults = perform_ds(
        mdims, codims, nbinstances, N=N, maniftype=maniftype, obj=obj, euclsimplex=euclsimplex,
    )
    if saving:
        pathdsresults = "dsresults_folder/" + nbr + "dsresults.pkl"
        pathproblems = "dsresults_folder/" + nbr + "problems.pkl"
        with open(pathproblems, "wb") as f:
            dill.dump(problems_meta, f)
        with open(pathdsresults, "wb") as f:
            pickle.dump(dsresults, f)

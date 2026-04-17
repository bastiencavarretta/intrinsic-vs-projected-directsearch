from tracemalloc import stop

import numpy as np
import numpy.linalg as lg

import time
from dataclasses import dataclass
from pymanopt import manifolds as man
import pickle
import dill


def directsearch(
    problem,
    simplexbudget=100,
    projection=1,
    psstype=1,
    rotation=1,
    euclsimplex=0,
    renormalize_tangent_vec=True,
    gamma=0.5,
    Gamma=2,
    alpha_0=1.0,
    alpha_max=1.0,
    c=1.0,
    eps=0.0,
    itmax=np.inf,
    printing=False,
):  # -> dict[str, Any] | list[dict[str, Any]] | None:

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
        raise ValueError("euclsimplex should be either 0 or 1")
    # riemannian simplex budget, for extracting the subsresults from an "euclidean simplex
    rb = (m + 1) * simplexbudget
    searchdim = m if projection == 0 else n
    (
        print(
            f"\n----starting direct-search - budget: {budget} - search dim: {searchdim}----"
        )
        if printing
        else None
    )
    # Initializations
    alpha = alpha_0
    vf = [f(x0)]
    vevperit = [1]
    valphas = [alpha]  # the value of alpha at each iteration
    success_indices, failure_indices = [], []
    k = 1
    evaluation_number = 1
    stopcriterion = None
    while (
        k < itmax and alpha > eps and evaluation_number < budget
    ):  # try and batch the evaluation later
        print("----entering main loop----") if printing and k == 1 else None

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
        len_polls = len(pss_at_x)
        while i < len_polls and success == False and evaluation_number < budget:
            p = pss_at_x[i]
            if type(p) == list or isinstance(
                p, tuple
            ):  # if the problem.manifold is a product manifold
                scaled_p = [alpha * comp for comp in p]
            else:
                scaled_p = alpha * p

            x_poll = manifold.exp(x, scaled_p)
            f_poll = f(x_poll)
            evaluation_number += 1
            nevaluation_for_k += 1

            # sufficient decrease with ambient norm (embedded submanifold case)
            if f_poll <= f_value - c * alpha**2 * problem.anorm(p) ** 2:
                x = x_poll
                f_value = f_poll
                alpha = min(alpha_max, Gamma * alpha)
                success = True
                success_indices.append(k)

            if (
                budget != np.inf
                and (evaluation_number + 1) % (int(0.1 * budget + 1)) == 1
                and printing
            ):  # printer 100 sorties en tout

                print(
                    f"it:{k:}, total evals: {evaluation_number:}/{budget:}, loss {vf[-1]:.3e}, alpha_ds: {valphas[-1]:.3e}, prop. polled: {vevperit[-1]:}/{len_polls:}"
                )

            i = i + 1

        if success == False:
            alpha = min(alpha_max, gamma * alpha)
            failure_indices.append(k)
        vevperit.append(nevaluation_for_k)

        vf.append(f_value)
        valphas.append(alpha)

        # Copying the directsearch data untill the rbudget if out.
        if euclsimplex == 1:
            if (
                evaluation_number <= rb or k == 1
            ):  # avoid weird behavior, when all the budget is out during the first iteration.
                rb_vf = vf.copy()  # rb means "riemannian budget"
                rb_vevperit = vevperit.copy()
                rb_valphas = valphas.copy()
                rb_x = x.copy()
                # rb_f_value = f_value.copy()
                # rb_success_indices = success_indices.copy()
                # rb_failure_indices = failure_indices.copy()

        k = k + 1

    if k >= itmax:
        stopcriterion = "full iterations used"
    elif alpha <= eps:
        stopcriterion = "low step size"
    elif evaluation_number >= budget:
        stopcriterion = "full budget used"
    else:
        raise ValueError("This should not happen if process exited the loop")

    if stopcriterion == "full iterations used" or stopcriterion == "low step size":
        rb_stopcriterion = stopcriterion
    elif stopcriterion == "full budget used":
        rb_stopcriterion = "full budget used"

    print("----direct-search finished----") if printing else None
    vf = np.array(vf)

    #    index of success/failures peut se retrouver grace au tableau des valphas
    if (
        euclsimplex == 0
    ):  # means I only performed computations in a riemannian simplex setting (lest costly). BUT budget is still absolute, so this function applies to non simplex budget anyway !!!!!
        return {
            "euclideansimplex": 0,
            "vf": vf,
            "per_iteration_evaluations": vevperit,
            "valpha": valphas,
            "stopcriterion": stopcriterion,
            "last_iterate": x,
        }
    if (
        euclsimplex == 1
    ):  # computations were run for euclidean simplex budget (scaled by adim+1). We return

        # First dictionnary corresponds to the Riemannian simplex budget. It is a subsample of the second (Euclidean simplex budget).
        return [
            {
                "euclideansimplex": 0,
                "vf": rb_vf,
                "per_iteration_evaluations": rb_vevperit,
                "valpha": rb_valphas,
                "stopcriterion": rb_stopcriterion,
                "last_iterate": rb_x,
            },
            {
                "euclideansimplex": 1,
                "vf": vf,
                "per_iteration_evaluations": vevperit,
                "valpha": valphas,
                "stopcriterion": stopcriterion,
                "last_iterate": x,
            },
        ]

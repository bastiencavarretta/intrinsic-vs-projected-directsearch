import numpy as np
import numpy.linalg as lg

import time
from dataclasses import dataclass
from pymanopt import manifolds as man
import pickle
import dill


def directsearch(
    problem,
    budget=100,
    projection=1,
    psstype=1,
    rotation=1,
    returnforeuclsimplex=1,
    renormalize_tangent_vec=True,
    gamma=0.5,
    Gamma=2,
    alpha_0=1.0,
    alpha_max=1.0,
    c=1.0,
    eps=0,
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
    if returnforeuclsimplex == 1:
        rb = (
            (m + 1) / (n + 1) * budget
        )  # the budget for the riemannian simplex setting (to truncate the directsearch ran with euclidean simplex budget)

    # Initializations
    alpha = alpha_0
    vf = [f(x0)]
    vevperit = [1]
    valphas = [alpha]  # the value of alpha at each iteration
    success_indices, failure_indices = [], []
    print("----starting direct-search----") if printing else None
    k = 1
    evaluation_number = 1

    while (
        k < itmax and alpha > eps and evaluation_number < budget
    ):  # try and batch the evaluation later
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
        while i < len_polls and success == False:
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

            # sufficient decrease with ambient norm (embedded submanifold case)
            if f_poll <= f_value - c * alpha**2 * problem.anorm(p) ** 2:
                x = x_poll
                f_value = f_poll
                alpha = min(alpha_max, Gamma * alpha)
                success = True
                success_indices.append(k)
            i = i + 1
            nevaluation_for_k += 1

        if success == False:
            alpha = min(alpha_max, gamma * alpha)
            failure_indices.append(k)
        vevperit.append(nevaluation_for_k)

        vf.append(f_value)
        valphas.append(alpha)

        if (
            budget != np.inf
            and (evaluation_number + 1) % (int(0.1 * budget + 1)) == 1
            and printing
        ):  # printer 100 sorties en tout

            print(
                f"it:{k:}, total evals: {evaluation_number:}/{budget:}, loss_per: {vf[-1]:.3e}, alpha_ds: {valphas[-1]:.3e}, total polls: {vevperit[-1]:}/{len_polls:}"
            )

        if k >= itmax:
            stopcriterion = "full iterations used"
        elif alpha <= eps:
            stopcriterion = "low step size"
        elif evaluation_number >= budget:
            stopcriterion = "full budget used"

        # Copying the directsearch data untill the rbudget if out.
        if returnforeuclsimplex == 1:
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

    if stopcriterion == "full iterations used" or stopcriterion == "low step size":
        rb_stopcriterion = stopcriterion
    else:
        rb_stopcriterion = "full budget used"

    print("----direct-search finished----") if printing else None
    vf = np.array(vf)

    #    index of success/failures peut se retrouver grace au tableau des valphas
    if (
        returnforeuclsimplex == 0
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
        returnforeuclsimplex == 1
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

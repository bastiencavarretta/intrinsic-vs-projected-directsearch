
import numpy as np
import numpy.linalg as lg

import time
from dataclasses import dataclass
from pymanopt import manifolds as man
import pickle
import dill

from tools.problems import ProblemLinearSubspace, ProblemEigh


def directsearch(problem, budget=100, projection = 1, psstype = 1, rotation =1,returnforeuclsimplex=1, renormalize_tangent_vec = True, gamma= 0.5 , Gamma=2 , alpha_0=1.0, alpha_max = 1.0, c=1.0, eps = 0, itmax = np.inf, printing = False):
    """ directsearch algorithm on a Riemannian manifold.
    Args:
        problem (Problem object in tools.problems.py)
        budget (int) : evaluation budget of the direct-search
        projection (int) : 0 or 1. See the method "build_pss" in the tools.problems.py file.
        psstype (int) : 1 or 2 or 3. See the method "build_pss" in the tools.problems.py file.
        rotation (int) : 0 or 1. See the method "build_pss" in the tools.problems.py file.
        returnforeuclsimplex (int) : 0 or 1. If 1, the budget is assumed to be scaled in an euclidean way (budget = N*(adim+1)), and the we store "riemannian budget" directsearch results as well as "riemannian budget" directsearch. Else the budget is assumed to be scaled in an riemannian way and we store only "riemannian budget" directsearch results.
        renormalize_tangent_vec (bool) : See the argument "renormalize" of method "build_pss" in the tools.problems.py file.
        gamma, Gamma, alpha_0, alpha_max,c (float): Direct-search stepsize update parameter.
        eps (float) : minimum stepsize before stopping
        itmax (int) : maximum number of iterations
        printing (bool) : prints progression if True
    Returns:
        If returnforeuclsimplex == 0 :
            dict with keys :
                "vf" : np.array of function values at each iteration
                "v_ev_per_it" : list of number of evaluations per iteration
                "vstep_sizes" : list of step sizes at each iteration
                "stopping_criterion" : string describing the stopping criterion
                "last_iterate" : last iterate of the direct-search
                "last_fvalue": last function value of the direct-search
                "sucess_indices" : list of iteration indices where the step was successful
                "failure_indices" : list of iteration indices where the step was unsuccessful
                "euclideansimplex" : 0
        If returnforeuclsimplex == 1 :
            list of two dicts. The first corresponds to the "riemannian budget" directsearch results, the second to the "euclidean budget" directsearch results. Each dict has the same keys as above, with "euclideansimplex" : 0 for the first dict, and "euclideansimplex" : 1 for the second dict.
    """
    
    n,m,x0,f,manifold,f_value = problem.adim, problem.mdim, problem.xstart, problem.costf, problem.manifold,problem.fstart


    x = x0.copy()
    if returnforeuclsimplex == 1:
        rbudget = (m+1)/(n+1)*budget  # the budget for the riemannian simplex setting (to truncate the directsearch ran with euclidean simplex budget)
    # Initializations

    alpha = alpha_0
    fvalues = [f(x0)]
    evaluation_per_its = [1] 
    step_sizes = [alpha] # the value of alpha at each iteration
    success_indices,failure_indices = [],[]
    print("----starting direct-search----") if printing else None
    k = 1 
    evaluation_number = 1
    while k < itmax and alpha>eps and evaluation_number < budget:
        success = False
        pss_at_x = problem.build_pss(x,projection = projection, psstype = psstype, rotation = rotation,renormalize = renormalize_tangent_vec)
        i = 0
        nevaluation_for_k = 0
        while i<len(pss_at_x) and success == False:
            p = pss_at_x[i]

            if type(p) == list: # if the problem.manifold is a product manifold
                scaled_p = [alpha*comp for comp in p]  
            else : 
                scaled_p = alpha*p 

            x_poll = manifold.exp(x,scaled_p) 
            f_poll = f(x_poll)
            evaluation_number +=1
            if f_poll <= f_value - c*alpha**2*problem.anorm(p)**2: # sufficient decrease with ambient norm (embedded submanifold case)
                x = x_poll
                f_value = f_poll
                alpha = min(alpha_max,Gamma*alpha)
                success = True
                success_indices.append(k)
            i = i+1
            nevaluation_for_k += 1
        if success == False : 
            alpha = min(alpha_max,gamma*alpha)
            failure_indices.append(k)
        evaluation_per_its.append(nevaluation_for_k)
        fvalues.append(f_value)
        step_sizes.append(alpha)

        if k >= itmax:
            stopping_criterion = "full iterations used"
        elif alpha<=eps:
            stopping_criterion = "low step size"
        elif evaluation_number >= budget:
            stopping_criterion = "full budget used"
        
        # Copying the directsearch data untill the rbudget if out.
        if returnforeuclsimplex == 1:
            if evaluation_number <= rbudget or k ==1: # avoid weird behavior, when all the budget is out during the first iteration.
                rbudget_fvalues = fvalues.copy()
                rbudget_evaluation_per_its = evaluation_per_its.copy()
                rbudget_step_sizes = step_sizes.copy()
                rbudget_last_iterate = x.copy()
                rbudget_f_value = f_value.copy()
                rbudget_success_indices = success_indices.copy()
                rbudget_failure_indices = failure_indices.copy()
        k = k+1
        

    if stopping_criterion == "full iterations used" or stopping_criterion == "low step size":
        rbudget_stopping_criterion = stopping_criterion
    else:
        rbudget_stopping_criterion = "full budget used"

    print("----direct-search finished----") if printing else None
    fvalues = np.array(fvalues)

    #    index of success/failures peut se retrouver grace au tableau des step_sizes
    if returnforeuclsimplex == 0: # means I only performed computations in a riemannian simplex setting (lest costly). BUT budget is still absolute, so this function applies to non simplex budget anyway !!!!!
        return {"vf" : fvalues,
            "v_ev_per_it" : evaluation_per_its,
            "vstep_sizes" : step_sizes,
            "stopping_criterion" : stopping_criterion,
            "last_iterate" : x,
            "last_fvalue": f_value,
            "sucess_indices" : success_indices,
            "failure_indices" : failure_indices,
            "euclideansimplex" : 0
            }
    if returnforeuclsimplex == 1: # computations were run for euclidean simplex budget (scaled by adim+1). We return 
        
        # First dictionnary corresponds to the Riemannian simplex budget. It is a subsample of the second (Euclidean simplex budget).
        return [{"vf" : rbudget_fvalues,
            "v_ev_per_it" : rbudget_evaluation_per_its,
            "vstep_sizes" : rbudget_step_sizes,
            "stopping_criterion" : rbudget_stopping_criterion,
            "last_iterate" : rbudget_last_iterate,
            "last_fvalue": rbudget_f_value,
            "sucess_indices" : rbudget_success_indices,
            "failure_indices" : rbudget_failure_indices,
            "euclidean_simplex" : 0
            },
            
            {"vf" : fvalues,
            "v_ev_per_it" : evaluation_per_its,
            "vstep_sizes" : step_sizes,
            "stopping_criterion" : stopping_criterion,
            "last_iterate" : x,
            "last_fvalue": f_value,
            "sucess_indices" : success_indices,
            "failure_indices" : failure_indices,
            "euclideansimplex" : 1
            }]




@dataclass(frozen=True)
class ds_key:
    """Key to store which attributes were used in directsearch."""
    euclideansimplex : int #Tuple[int, ...] = (0,1)
    projection : int # Tuple[int, ...] = (0,1)
    rotation : int #Tuple[int, ...] = (0,1)
    psstype : int #Tuple[int, ...] = (1,2,3)
    renormalize_tangent_vector : bool = True


# Function performing directsearch experiments
def perform_ds(mdims,codims,nbinstances, N = 100, maniftype =1, obj = 1,euclideansimplex = 1, printingprogression = True):
    """
    Args:
        mdims (list of int) : list of manifold dimensions
        codims (list of int) : list of manifold codimensions
        nbinstances (int) : number of problem instances per (mdim,codim) pair
        N (int) : evaluation budget factor for directsearch (budget = N*(adim+1) if euclideansimplex = 1, budget = N*(mdim+1) if euclideansimplex = 0)
        maniftype (int) : type of manifold. See tools.problems.py.
        obj (int) : type of objective. See tools.problems.py.
        euclideansimplex (int) : 0 or 1. See directsearch docstring for details.
        printingprogression (bool) : prints progression if True
    """
    
    
    adimsmdims = [[[codim+mdim,mdim] for codim in codims] for mdim in mdims]
    maxi,maxj = len(adimsmdims), len(adimsmdims[0])

    # Generating problems
    if maniftype in [1,1.5]: #subspace or horizontal subspace (vect(e_1, \dots, e_m))
        problems = [[[ProblemLinearSubspace(man.Stiefel(adimsmdims[i][j][0], adimsmdims[i][j][1]).random_point() if maniftype == 1 else np.eye(adimsmdims[i][j][0], adimsmdims[i][j][1]),obj = obj) for k in range(nbinstances)] for j in range(maxj)] for i in range(maxi)]
    elif maniftype == 2: # sphere embedded in Rn
        problems = [[[ProblemEigh(adimsmdims[i][j][0], adimsmdims[i][j][1]) for k in range(nbinstances)] for j in range(maxj)] for i in range(maxi)]  

    dsresults = []
    fullt0 = time.time()
    t0 = fullt0
    count =0
    for imdim,mdim in enumerate(mdims):
        dsresults.append([])
        for icodim,codim in enumerate(codims):
            dsresults[imdim].append([])
            for k in range(nbinstances):
                adim = mdim + codim
                problem = problems[imdim][icodim][k]
                dsresults[imdim][icodim].append({})
                for projection in [0,1]:
                        for rotation in [0,1]:
                            for psstype in [1,2,3]:

                                    budget = N*(problem.mdim*(euclideansimplex == 0)+problem.adim*(euclideansimplex ==1)+1)

                                    # perform directsearch
                                    dsresult = directsearch(problem,budget = budget, projection = projection, rotation = rotation, psstype = psstype, renormalize_tangent_vec= True,returnforeuclsimplex=euclideansimplex,eps = 0)

                                    # the keys thus depend on the computations performed above
                                    if euclideansimplex == 1 :
                                        key1 = ds_key(1, projection,rotation, psstype)
                                        dsresults[imdim][icodim][k][key1] = dsresult[1] # the euclidean simplex computations
                                        key0 = ds_key(0, projection,rotation, psstype)
                                        dsresults[imdim][icodim][k][key0] = dsresult[0] # the riemannian simplex (truncation)
                                    elif euclideansimplex == 0:
                                        key = ds_key(0, projection,rotation, psstype)
                                        dsresults[imdim][icodim][k][key] = dsresult
            # Printing progression
            if printingprogression:
                count = count+1
                t1  = time.time()
                chours= int((t1-fullt0)//3600)
                cminutes = int(((t1-fullt0)%3600)//60)
                cseconds =  int(((t1-fullt0)%3600)%60)
                print("newwdim : {:} out of {:}, cumulated_time = {:<2}:{:<2}:{:<2}, newtime = {:.2f}s".format(count,len(mdims)*len(codims),chours, cminutes, cseconds,t1-t0)) 
                t0 = t1
    return problems, dsresults




# Function saving directsearch experiments
def saveperform_ds(expnumber, mdims, codims, N,nbinstances,maniftype, obj, euclideansimplex =1, saving = False):
    """ Performs and saves directsearch experiments.
    Args:
        expnumber (int) : experiment number (for saving purposes)
        mdims (list of int) : list of manifold dimensions
        codims (list of int) : list of manifold codimensions
        N (int) : evaluation budget factor for directsearch
        nbinstances (int) : number of problem instances per (mdim,codim) pair
        maniftype (int) : type of manifold. See tools.problems.py.
        obj (int) : type of objective. See tools.problems.py.
        euclideansimplex (int) : 0 or 1. See directsearch docstring.
        saving (bool) : saves results if True. dry run else.
    Note :
        saves problems characteristics and results in the folder "dsresults_folder" in pickle and dill format."""
    nbr = "exp" + str(expnumber)+ "_maniftype" + str(maniftype) + "_obj" + str(obj) + "_" 
    print("computing :", nbr,"###########################################################")
    problems, dsresults = perform_ds(mdims, codims, nbinstances, N = N, maniftype = maniftype, obj = obj,euclideansimplex=euclideansimplex)
    if saving:
        pathdsresults = "dsresults_folder/"+ nbr + "dsresults.pkl"
        pathproblems = "dsresults_folder/"+ nbr + "problems.pkl"
        with open(pathproblems, "wb") as f: 
            dill.dump({"mdims" :mdims,"codims" :codims,"nbinstances" :nbinstances,"problems" :problems},f) 
        with open(pathdsresults, "wb") as f: 
            pickle.dump(dsresults,f)
            
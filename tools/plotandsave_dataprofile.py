# Plooting parameters
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle
import dill
import matplotlib as mpl
from tools.perform_directsearch import ds_key
import itertools


# Computing data profiles
def dataprofile(vdsresults,vproblems,tau = 1e-2,euclideansimplex = 0, lbdfoeuclideansimplex = 1, N=100): 
    """Arguments :
            vdsresults (list) : list of directsearch experiments results, for various (manifold, objective) couples. Each element of the list is the dsresults output of tools.perform_directsearch.perform_ds
            vproblems (list) : list of problems corresponding to the vdsresults. Each element of the list is the problems output of tools.perform_directsearch.perform_ds
            tau (float) : tolerance for the data profile
            euclideansimplex (int) : 0 or 1. Whether the data profile is computed with respect to the manifold dimension (0) or the ambient dimension (1)
            lbdfoeuclideansimplex (int) : 0 or 1. Whether the lower bound proxy is computed with respect to the manifold dimension (0) or the ambient dimension (1)
            N (int) : maximum budget (in number of simplex gradients) for the data profile abscissa

        Returns :
            a dictionary with 3 dataprofiles :
                - "dp_codim" : dataprofile for fixed mdim, varying codim
                - "dp_mdim" : dataprofile for fixed codim, varying mdim
                - "dp_tot" : dataprofile for all problems together
        
        Remark : 
            To change which mdim (or codim) is fixed for "dp_codim" (or "dp_mdim"), change the line imdim = 1 (or icodim = 2) in the loops below, to some other viable index.
        """

    ncodims = len(vproblems[0]["codims"])
    nmdims = len(vproblems[0]["mdims"])
    nbinstances = vproblems[0]["nbinstances"]
    nb_problems = len(vproblems) # number of different (manifold, objective) considered


    vlbdfo = [[[[None for k in range(nbinstances)] for icodim in range(ncodims) ] for imdim in range(nmdims)] for nb_problem in range(nb_problems)] # lower bound proxy for each problem instance
    vfalpha = np.zeros((nb_problems,2,2,3,ncodims,nmdims, nbinstances)) 
    dp_codim = np.zeros((2,2,3,ncodims,N+1)) # one dataprofile per codim (fixed mdim)
    dp_mdim = np.zeros((2,2,3,nmdims,N+1)) # one dataprofile per mdim (fixed codim)
    dp_tot = np.zeros((2,2,3,N+1)) # one dataprofile for all dimensions of all problems

    for nb_problem, dsresults in enumerate(vdsresults): 
        codims = vproblems[nb_problem]["codims"]
        mdims = vproblems[nb_problem]["mdims"]
        for icodim, codim in enumerate(codims): 
            for imdim,mdim in enumerate(mdims): 
                scalingdim = mdim + codim if euclideansimplex == 1 else mdim
                for k in range(nbinstances):
                    
                    # computing the lower bound proxy
                    key0 = ds_key(euclideansimplex=euclideansimplex, projection=0, rotation=0, psstype=1)
                    dsresult = dsresults[imdim][icodim][k]
                    lbdfo = dsresult[key0]["vf"][-1] # intialization
                    for projection, rotation, psstype, euclideansimplexexplo in itertools.product([0,1], [0,1], [1,2,3],range(lbdfoeuclideansimplex)): # if lbdfoeuclideansimplex = 1, this explores the values given by a direct-search with budget scaled by the ambient dimension. Else only by the manifold dimension.
                        key = ds_key(euclideansimplex=euclideansimplexexplo,projection=projection,rotation=rotation,psstype = psstype)
                        min_ik = dsresult[key]["last_fvalue"] # assumes the DFO method yields only decreasing function values
                        if min_ik < lbdfo :
                            lbdfo = min_ik
                    vlbdfo[nb_problem][imdim][icodim][k] = lbdfo


                    # Computing the index int(#eval/budget) that solves a problem with tolerance tau
                    for projection, rotation, psstype in itertools.product([0,1], [0,1], [1,2,3]):
                        key = ds_key(euclideansimplex=euclideansimplex,projection=projection,rotation=rotation,psstype = psstype)
                        evaluations = np.cumsum(dsresult[key]["v_ev_per_it"])  #cumulated evaluations at each iterate 
                        fvalues = dsresult[key]["vf"]
                        f0 = fvalues[0]
                        test_value = (fvalues - lbdfo)/(f0-lbdfo)
                        indices_solving = np.where(test_value <= tau)[0]
                        if len(indices_solving) == 0:
                            alpha = None
                        else :
                            index_first_solving = np.min(indices_solving)
                            alpha = int(evaluations[index_first_solving]/(scalingdim + 1)) # alpha is the abscissa of the data profile

                        #storing for every problem and solver
                        vfalpha[nb_problem,projection, rotation,psstype-1, icodim,imdim,k]= alpha 
                        
    
    # Creating the dataprofile for fixed mdim, and every codim.
    for projection, rotation, psstype,icodim in itertools.product([0,1], [0,1], [1,2,3],np.arange(ncodims)):
        imdim = 1 # manifold dimension fixed (4 in our experiments)
        for alpha in range(N+1): 
            # cumulating all problems solved for a given abscissa alpha
            ratio_solved = np.sum(vfalpha[:,projection, rotation,psstype-1, icodim,imdim,:]<=alpha)/(nbinstances*nb_problems)
            dp_codim[projection, rotation, psstype-1, icodim,alpha] = ratio_solved
    
    # Creating the dataprofile for fixed codim, and every mdim.
    for projection, rotation, psstype,imdim in itertools.product([0,1], [0,1], [1,2,3],np.arange(nmdims)):
        icodim = 2 # codimension fixed (4 in our experiments)
        for alpha in range(N+1):
            ratio_solved = np.sum(vfalpha[:,projection, rotation,psstype-1, icodim,imdim,:]<=alpha)/(nbinstances*nb_problems)
            dp_mdim[projection, rotation, psstype-1, imdim,alpha] = ratio_solved
    
    # Creating one common dataprofile for all problems
    for projection, rotation, psstype,imdim,icodim in itertools.product([0,1], [0,1], [1,2,3],np.arange(nmdims),np.arange(ncodims)):
        for alpha in range(N+1):
            ratio_solved = np.sum(vfalpha[:,projection, rotation,psstype-1, :,:,:]<=alpha)/(nbinstances*nb_problems*ncodims*nmdims)
            dp_tot[projection, rotation, psstype-1,alpha] = ratio_solved
    
    return {"dp_codim" :dp_codim, "dp_mdim" : dp_mdim, "dp_tot":dp_tot}



# Plotting data profiles
def plotting_dp(expnumber, maniftypeobj = [(1,2)], rotations = [0], psstypes = [1,2,3], projections = [0,1], euclideansimplex = 0,lbdfoeuclideansimplex = 1, tau = 1e-2,N = 100,plotcodimsev = True, plotmdimsev = True, saving = False):
    """Plotting and saving data profiles from directsearch experiments.
    
    Args:
        expnumber (int) : the index of the experiment (same argument as tools.perform_directsearch.saveperform_ds)
        maniftypeobj (list) : list of tuples of type (index of the manifold)x(index of the objective) for which experiment "expnumber" ran the directsearch.
        
        #### The following arguments tell which variant of the directsearch to plot : ####
        rotations (list): any subset of {0,1}.  
        psstypes (list): any subset of {1,2,3}.
        projections (list): any subset of {0,1}.

        euclideansimplex (int) : 0 or 1. Total budget of the data profile.
        lbdfoeuclideansimplex (int) : 0 or 1. Tells which computations are used to compute a proxy of the lower bound of each problem. 
        tau (float) : tolerance for considering a problem as solved.
        N (int) : maximum number of simplex gradient evaluations for the data profile.
        plotcodimsev (bool) : If True, plots data profiles for various codims, fixed mdim (imdim = 1).
        plotmdimsev (bool) : If True, plots data profiles for various mdims, fixed codim (icodim = 2).
        saving (bool) : saves plots if True. Only plotting else.

    Preconditions : 
        all experiments of index "expnumber" have the same values for codims and mdims.
        lbdfoeuclideansimplex must be 0 if euclideansimplex is 0. 
        euclideansimplex must be 0 if the directsearch data was obtained by running perform_directsearch.saveperform_ds with euclideansimplex = 0.
        N must be the same as the one used in perform_directsearch.saveperform_ds.

    Results :
        plots and saves the data profiles of the projected vs intrinsic methods for every combination in rotations x psstypes x projections. 
    """

    # Plotting parameters
    titlefonts = 28
    subtitle_fonts = 24
    label_fonts = 12
    cmapproj = mpl.colormaps['Set1'].colors[1:4]
    cmapnoproj = mpl.colormaps['Set1'].colors[1:4]
    cmap = [cmapnoproj,cmapproj]
    linestyles = ["-","--"]

    # Opening and storing the data from directsearch experiments
    vdsresults = []
    vproblems = []
    for maniftype, obj in maniftypeobj:
        nbr = "exp" + str(expnumber)+ "_maniftype" + str(maniftype) + "_obj" + str(obj) + "_" 
        pathdsresults = "dsresults_folder/"+ nbr + "dsresults.pkl"
        pathproblems = "dsresults_folder/"+ nbr + "problems.pkl"
        with open(pathdsresults, "rb") as f:
            dsresults = pickle.load(f)

        with open(pathproblems, "rb") as f: 
            loadedproblems = dill.load(f)
        problems = loadedproblems
        vdsresults.append(dsresults)
        vproblems.append(problems)

    # storing the experiments common charateristics
    problem0 = vproblems[0]
    mdims, codims, nbinstances, problems = problem0["mdims"],problem0["codims"],problem0["nbinstances"],problem0["problems"]

    # Computing, storing data profile
    dps = dataprofile(vdsresults,vproblems,tau = tau,euclideansimplex = euclideansimplex, lbdfoeuclideansimplex=lbdfoeuclideansimplex)
    dp_codim,dp_mdim,dp_tot = dps["dp_codim"],dps["dp_mdim"], dps["dp_tot"]
    print("computation number = ", nbr) 

    ## Displaying for various codims, fixed mdim #################################################################################
    if plotcodimsev:
        plotnbr = len(codims)
        linenbr = plotnbr//3 
        if plotnbr%3 != 0:
            linenbr = linenbr+1
        fig, ax = plt.subplots(linenbr, 3,figsize = (15,4*linenbr))
        ax = ax.flatten()
        nbr_unused_subplots = 3*linenbr - plotnbr 
        for k in range(nbr_unused_subplots):
            fig.delaxes(ax[plotnbr+k])

        #plotting
        for icodim,codim in enumerate(codims):
            ax[icodim].set_title("n-m = {:}".format(codim),fontsize=subtitle_fonts)
            ax[icodim].grid(True)
            ax[icodim].set_xlim(0, N)
            ax[icodim].set_ylim(0, 1)
            for rotation, projection,psstype in itertools.product(rotations, projections, psstypes):
                variant = "intr"*(1-projection)+"proj"*projection

                # plotting with labels in the last subfigure
                if icodim == len(codims)-1:
                    ax[icodim].plot(dp_codim[projection, rotation, psstype-1, icodim,:], color =cmap[projection][psstype-1],linestyle = linestyles[projection],linewidth = 2, label = "PSS{:} ({:})".format(psstype,variant))
                    ax[icodim].legend(fontsize=label_fonts) 
                # plotting without labels
                else:
                    ax[icodim].plot(dp_codim[projection, rotation, psstype-1, icodim,:], color =cmap[projection][psstype-1],linestyle = linestyles[projection],linewidth = 2)

        # layout
        fig.supxlabel('Number of simplex gradient evaluations', fontsize=titlefonts)
        fig.supylabel('Ratio of problems solved', fontsize=titlefonts)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.tight_layout()


        if saving:
            path_problems = ""
            for mto in maniftypeobj:
                path_problems = path_problems + str(mto[0])+"-"+str(mto[1]) + "_"
            figname = "exp"+ str(expnumber) + "_manifobj" + path_problems + "codimsdp" + "_param-proj-" + str(projections) + "_psstypes" + str(psstypes) + "_rot" + str(rotations) + "_es" + str(euclideansimplex) + "_eslbdfo" + str(lbdfoeuclideansimplex) + "_tau" + str(tau)+ "_mdims" + str(mdims) +"_codims"+ str(codims) + "_nbi"+ str(nbinstances) +".pdf"
            figname = figname.replace(" ","").replace("[","").replace("]","")
            plt.savefig("tables_and_plots/dataprofiles/"+figname,bbox_inches='tight', dpi=300)
        plt.show()


    ## Displaying for various mdims, fixed codim ##############################################
    ##############################################################################
    if plotmdimsev:
        plotnbr = len(mdims)
        linenbr = plotnbr//3 
        if plotnbr%3 != 0:
            linenbr = linenbr+1
        fig, ax = plt.subplots(linenbr, 3,figsize = (15,4*linenbr))
        ax = ax.flatten()
        nbr_unused_subplots = 3*linenbr - plotnbr 
        for k in range(nbr_unused_subplots): 
            fig.delaxes(ax[k])

        # plotting
        for iimdim,mdim in enumerate(mdims):
            imdim = iimdim +1   # slight shift, to make a symmetry with the codim plots
            ax[imdim].set_title("m = {:}".format(mdim), fontsize=subtitle_fonts)
            ax[imdim].grid(True)
            ax[imdim].set_xlim(0, N)
            ax[imdim].set_ylim(0, 1)
            for rotation, projection,psstype in itertools.product(rotations, projections,psstypes):
                variant = "intr"*(1-projection)+"proj"*projection

                # plotting with labels in the last subfigure
                if imdim == len(mdims):
                    ax[imdim].plot(dp_mdim[projection, rotation, psstype-1, iimdim,:], color =cmap[projection][psstype-1],linestyle = linestyles[projection],linewidth = 2,label = "PSS{:} ({:})".format(psstype,variant))
                    ax[imdim].legend(fontsize=label_fonts)
                # plotting without labels
                else:
                    ax[imdim].plot(dp_mdim[projection, rotation, psstype-1, iimdim,:], color =cmap[projection][psstype-1],linestyle = linestyles[projection],linewidth = 2) 


        # layout
        ax[0].axis('off')
        fig.supxlabel('Number of simplex gradient evaluations', fontsize=titlefonts)
        fig.supylabel('Ratio of problems solved', fontsize=titlefonts)
        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        plt.tight_layout()

        if saving: 
            path_problems = ""
            for mto in maniftypeobj:
                path_problems = path_problems + str(mto[0])+"-"+str(mto[1]) + "_"

            figname = "exp"+ str(expnumber) + "_manifobj" + path_problems + "mdims" + "_param-proj-" + str(projections) + "_psstypes" + str(psstypes) + "_rot" + str(rotations) + "_es" + str(euclideansimplex) + "_eslbdfo" + str(lbdfoeuclideansimplex) + "_tau" + str(tau)+ "_mdims" + str(mdims) +"_codims"+ str(codims) + "_nbi"+ str(nbinstances) +".pdf"
            figname = figname.replace(" ","").replace("[","").replace("]","")
            plt.savefig("tables_and_plots/dataprofiles/"+figname,bbox_inches='tight', dpi=300)
        plt.show()


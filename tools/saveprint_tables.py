import itertools
import pickle
import dill
from tools.perform_directsearch import ds_key #key to read the directsearch data.
import pandas as pd
# formatting tables
pd.set_option('display.float_format', '{:.2f}'.format)


# plotting data profiles
def performancetable(expnumber, maniftypeobj = [(1,2)], rotations = [0,1], psstypes = [1,2,3], euclideansimplexs = [0,1], printing = True, saving = False):
    """Arguments : 
            expnumber (int) : the index of the experiment (same argument as tools.perform_directsearch.saveperform_ds)
            maniftypeobj (list) : list of tuples of type (index of the manifold)x(index of the objective) for which experiment "expnumber" ran the directsearch. See "tools.problems" for viable tuples 
            rotations (list): [0] or [1] or [0,1].  
            psstypes (list): any subset of {1,2,3}.
            euclideansimplexs (list) : [0] or [1] or [0,1] 

            printing (bool) : prints tables if True
            saving (bool) : saves tables if True

        Preconditions :
            - all experiments of index "expnumber" have the same number of codims and mdims values 
            - all experiments of index "expnumber"
            
        Results :
            prints and saves the performance tables of the projected vs intrinsic methods for every combination in rotations x psstypes x euclideansimplexs
    """


    # Opening the datas
    nb_problems = len(maniftypeobj)
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
        vdsresults.append(dsresults)
        vproblems.append(loadedproblems)

    # storing the experiments common charateristics
    loadedproblem = vproblems[0]
    mdims, codims, nbinstances = loadedproblem["mdims"],loadedproblem["codims"],loadedproblem["nbinstances"]

    # Creating key for performance table 
    keyeuclideansimplexs = euclideansimplexs
    keyrotations = [0,1]
    keypsstypes = [1,2,3]

    tablekeypd = pd.MultiIndex.from_product(
        [mdims, codims,keyeuclideansimplexs, keyrotations, keypsstypes],
        names=["mdim", "codim", "euclideansimplex", "rotation", "psstype"]
    )

    table = pd.DataFrame(index = tablekeypd, columns = ["ratio"]) 

    # Computing, storing performance table
    for imdim,mdim in enumerate(mdims):
        for icodim,codim in enumerate(codims):
            sub = table.xs((mdim, codim), level=("mdim", "codim"))
            
            for (euclideansimplex,rotation,psstype),grp in sub.groupby(level= ["euclideansimplex","rotation", "psstype"],sort = False):
                
                euclideansimplex,rotation,psstype = int(euclideansimplex), int(rotation), int(psstype)
                keyproj = ds_key(euclideansimplex = euclideansimplex, projection = 1,rotation = rotation, psstype = psstype)
                keynoproj = ds_key(euclideansimplex = euclideansimplex, projection = 0,rotation = rotation, psstype = psstype)


                ratio = 0
                for k in range(nb_problems):
                    for l in range(nbinstances):
                        dsresult = vdsresults[k][imdim][icodim][l]
                        gap = (dsresult[keyproj]["last_fvalue"]-dsresult[keynoproj]["last_fvalue"])
                        if gap >0 :
                            ratio += 1
                ratio = ratio/(nbinstances*nb_problems)
                table.loc[(mdim, codim,euclideansimplex,rotation,psstype),"ratio"] = ratio
                
    # printing and saving tables
    for rotation, euclideansimplex, psstype in itertools.product(rotations, euclideansimplexs, psstypes):
        subtable = table.xs((euclideansimplex, rotation, psstype), level=("euclideansimplex", "rotation", "psstype"))
        subtable = subtable.reset_index()
        subtable = subtable.pivot(index="codim", columns="mdim")
        if saving:
            latex_code = subtable.to_latex(
            index=True,
            float_format="%.2f".__mod__,
            column_format="l|"+ len(mdims)*"c",
            caption=None, 
            label=None,
            position = 0,
            header=True,
            )
            path_problems = ""
            for mto in maniftypeobj:
                path_problems = path_problems + str(mto[0])+"-"+str(mto[1]) + "_"
            print(path_problems)
            path = "tables_and_plots/performance_tables/" + "exp" + str(expnumber) + "_manifobj" +path_problems + "rotation" + str(rotation) + "_euclideansimplex" + str(euclideansimplex) + "_psstype" + str(psstype) + ".tex"

            with open(path, "w") as f:
                f.write(latex_code)

        if printing:
            print("\nrotation = ", rotation, ", euclideansimplex = ", euclideansimplex, ", psstype = ", psstype)
            print(subtable)

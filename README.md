# intrinsic-vs-projected-directsearch
This repository contains the Python code for the paper "Complexity guarantees and polling strategies for Riemannian direct-search methods" (https://arxiv.org/abs/2511.15360). In the field of derivative-free optimization, direct-search algorithms operate by polling the variable space along specific direction forming positive spanning sets (PSS). When the problem variables are constrained to lie on a Riemannian manifold, two polling strategies can be defined: either built on _intrinsic_ or _projected_ PSSs. 

We compare the performance of a set of direct-search algorithms based on these PSSs by plotting their data profile against a set of synthetic manifold optimization problems (barycenter problem, or quadratic functions), with a fixed evaluation budget. For each manifold dimension $m$ and codimension $n-m$ configuration, we also display the proportion of problems for which the intrinsic strategy provides a better final function value than its projected analogue, resulting in what we call _performance tables_. We observe that the intrinsic strategies perform better than their projected counterpart, when $\frac{n-m}{m}$ is large enough. These results can be found in the `code/dimension_influence` subdirectory. 

To apply these direct-search method to a more geometrically structured context, we test our methods on a problem of synchronization of rotations, for which $n-m = 30$ and $m=20$. The resulting dataprofile in `code/synchronization_rotations` also favors the use of intrinsic PSSs.



# Reproducing results from the paper
Subdirectory `/code/dimension_influence` (resp. `/code/synchronization_rotations`) allows to reproduce results from Sections 5.2 and 6.2 (resp. Section 6.3) of the paper.

## Data profiles
To reproduce the data profiles and performance tables from the paper, run:
- `cd code/`
- `pip install -r requirements.txt` to install required python packages
- `bash reproduce.sh` or `bash reproduce.sh dryrun` to dryrun first. The run takes approximately four hours with configuration Debian GNU/Linux 13, Intel(R) Core(TM) i7-1265U 10 cores 4.8 GHz, 32 GB RAM.

The resulting data profiles and tables are saved in the `plots` and `tables` subdirectories of,
- `code/dimension_influence/experiments/finalrun_budg100mdimcodimall_v1`,
- `code/dimension_influence/experiments/finalrun_budg200mdim32codim4_uselessother_v1`,
- and `code/synchronization_rotations/experiments/k5_n2_budg500_noise1em6_nowarmstart_v1`.

## Cosine measure heatmap
Run cell by cell the notebook `code/dimension_influence/cosinemeasure_psscost.ipynb` to display the cosine measure heatmap on the hypersphere, as well as its range as a function of the ambient dimension $n$. These results correspond to section 5.2 of the paper.

Notice that, in a following section, this notebook attempts to estimate the computational time to build a given time of a PSS. Results should be interpreted cautiously, as our implementation of PSS may still be improved.

# Testing other experimental configurations
For both subdirectories `dimension_influence` or `synchronization_rotations`, it is possible to change the problems and algorithm parameters, to test other experimental configurations. To do so, modify the USER INPUT section of the `/code/name-of-subdirectory/new_experiments.py` file  (title of the experiment, problems parameters, algorithm parameters). Then,
- `cd name-of-subdirectory`
- `python new_experiments.py --nodate`
- `python run.py --exppath experiments/name-of-your-experiment`
- `python plot_dataprofile.py --exppath experiments/name-of-your-experiment`

The resulting data profile(s) will be saved in `/code/name-of-subdirectory/experiments/name-of-your-experiment/plots`.

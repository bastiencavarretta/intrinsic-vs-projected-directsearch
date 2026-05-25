# intrinsic-vs-projected-directsearch
This repository contains the code for the paper "Complexity guarantees and polling strategies for Riemannian direct-search methods" (https://arxiv.org/abs/2511.15360).

# Use
Directory `/code/dimension_influence` (resp. `/code/synchronization_rotations`) allows to reproduce results from Sections 5 and 6.2 (resp. Section 6.3).

Start by setting up the python environnement (we used a python 3.14.4 virtual environnement):  `pip install -r requirements.txt`.
## Dimension influence and 
1. Run cell by cell the `cosine_measure.ipynb` notebook to display and save the cosine measure heatmap of the projected PSS1 (maximal basis) on the sphere.
2. 



<!-- 
 data profiles and performance tables from section 6. The cell performing the direct-search ran for three hours in our case, storing 1.5Gb of data. -->

To test other parameter values, edit the USER INPUT section in `new_experiment.py`, then,
- `cd code/dimension_influence`
- `python new_experiments.py`: creates a new directory `YYYY-MM-DD_name-you-gave-to-exp` in `code/dimension_influence/experiments` with parameters of your choice in stored in `config.json`
- `python run.py --exxpath experiments/YYYY-MM-DD_name-you-gave-to-exp` to run all direct search algorithms associated to every configuration in the `config.json` and store the raw results in `experiments/YYYY-MM-DD_name-you-gave-to-exp/results`.
- `python plot_dataprofile.py experiments/YYYY-MM-DD_name-you-gave-to-exp` to plot data profiles with aggregated dimension or codimension, and store them in `experiments/YYYY-MM-DD_name-you-gave-to-exp/plots`


# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project
Research code for the paper "Complexity guarantees and polling strategies for Riemannian
      direct-search methods". Implements and benchmarks direct-search optimization algorithms on Riemannian manifolds, comparing polling strategies (positive spanning sets or PSSs) intrinsic vs. projected approaches.

## Setup
```bash
pip install -r requirements.txt
```
Dendencies: `pymanopt` (manifold operations), `numpy`, `scipy`, `pandas`, `matplotlib`, `dill`.

## General setup
The direct-search algorithm has many parameters:
- projection: 1 (PSS are projected to the tangent space), 0 (PSSs are defined intrinsically).
- rotation: 1 (PSSs are randomly rotated),  0 (PSSs are always dependent on the same basis).
- psstype: 1 (orth. basis and opposite), 2 (orth. basis and negated sum), 3 (uniform angles, based on Cholesky decomposition).
- budget: evaluation budget of the optimization solver
- eucl_simplex: 0 (budget is multiplied by manifold dim + 1), 1 (budget is multiplied by ambient dim + 1)


We compare the performance of each version of the algorithm by testing their efficiency on the resolution of a few dozens of instances of a few optimization problems (quadratics, synchronization of rotation, barycenter problem..). Metrics:
- Dataprofiles (Moré, Wild, 2009)
- Performance tables

## Twofold repository
- The `synchronization_rotations` directory benchmarks various polling strategy for a fixed problem, with a fixed funny manifold (product of SO(n)) by plotting various dataprofiles for this problem.
- The `dimension_influence` directory performs similar test on a benchmark composed of various optimization problems of various dimension and codimension. It then aggregates these problems by dimension or codimension, and finally plot dataprofiles for a fixed dimension and codimension.

## Coherence and readability issue
When CLAUDE arrives on the project, they will notice that the first directory (synchronization of rotations) is more clean, readable, reproducible than the second, which is spaghetti, and makes a suboptimal use of dataframes. This is due to an increase in the programming skills of the owner of the project (dir1: april2026, dir2: october2025). 

## Objective of Claude Code
### Clean the `dimension_influence` directory
The first goal of Claude Code is to draw from the logic of the experimental pipeline in `synchronization_rotations` to make `dimension_influence` more reproducible
- Look accross files for inconsistency issues (variables with different names but similar objective).
- Make the output of the `directsearch` function coherent across directories.
- Emancipate from the Jupyter Notebook dependance as much as possible
- Create a similar new_experiment.py file, that ensure systematic experiment metadata storing etc..

### Seed adequatly
Propose a way to seed all experiments so that they are reproducible. 
Important: Detail deeply what you are doing at this moment, and why it is relevant to freezing the seed accross the whole pipeline. The programmer thinks of seeding as black magic.

### Introduce a sanitycheck parameter
A sanitycheck (bool) parameter will be added to the whole pipeline (e.g. in the `new_experiment.py` file primarily) to automatically downgrade budget, dimension so that problems run much faster. This will help debugging.

### Suggest adequate improvements of the whole repository


## Forbidden actions
- You cannot change the colormaps and plotting parameters before asking me explicitly.
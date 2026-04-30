import os
import json
from datetime import datetime
from os import path
import json
import itertools
import subprocess
import numpy as np

# ====== USER INPUT ======
experiment_name = "k5_n3_budget200"
description = "In this fourth experiment, the matrix sizes is 5 and we take 3 of them, and 100 instances of the problem are considered. No warmstart. We take a budget of 200 simplex gradients. We run with and without warmstart. We need to extract the required subtable to perform the data profiles"
pb_parameters = {
    "seed": np.arange(0, 1, 1).tolist(),
    "n": [3],
    "d": [5],  # [3, 5, 10],
    "noise": [1e-6],  # ordre de grandeur de la distorsion : noise * d
    "warmstart": [0, 0.2],
}


algo_parameters = {
    "psstype": [1, 2, 3],
    "projection": [0, 1],
    "rotation": [0, 1],
    "simplexbudget": [200],  # number of simplex gradients
    "euclsimplex": [1],  # 0 means riemannian simplex, 1 euclidean.
    "gamma": [0.5],
    "Gamma": [2.0],
    "alpha_0": [1.0],
    "alpha_max": [1.0],
    "c": [1.0],
}


# =======================

# Auto date
date_str = datetime.now().strftime("%Y-%m-%d")
exp_id = f"{date_str}_{experiment_name}_v1"

base_dir = os.path.join("experiments", exp_id)
os.makedirs(base_dir, exist_ok=True)

# ---- metadata.json ----
metadata = {
    "experiment_name": experiment_name,
    "date": date_str,
    "description": description,
    "pb_parameters": pb_parameters,
    "algo_parameters": algo_parameters,
    "tags": [experiment_name],
    "version": "v1",
}

with open(os.path.join(base_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ---- config.json (empty for now) ----
with open(os.path.join(base_dir, "config.json"), "w") as f:
    json.dump({"experiments": []}, f, indent=2)

# ---- generate_config.py ----

pb_keys = pb_parameters.keys()
pb_values = pb_parameters.values()
algo_keys = algo_parameters.keys()
algo_values = algo_parameters.values()

vpb_parameters = []
for idx, combo in enumerate(itertools.product(*pb_values)):
    instance = dict(zip(pb_keys, combo))
    instance["idpb"] = idx
    vpb_parameters.append(instance)

valgo_parameters = []
for idx, combo in enumerate(itertools.product(*algo_values)):
    instance = dict(zip(algo_keys, combo))
    instance["idalgo"] = idx
    valgo_parameters.append(instance)

config = {"pb_parameters": vpb_parameters, "algo_parameters": valgo_parameters}

path_to_dump_json = os.path.join(base_dir, "config.json")

with open(path_to_dump_json, "w") as f:
    json.dump(config, f, indent=2)

print(
    f"Generated {{len(vpb_parameters)}} problem and {{len(valgo_parameters)}} algorithm configurations."
)


# ---- notes.md ----
notes = f"""# {exp_id}

## Description
{description}

## Parameters
{json.dumps(pb_parameters, indent=2)} {json.dumps(algo_parameters, indent=2)}

## Notes
- Add observations here
"""

with open(os.path.join(base_dir, "notes.md"), "w") as f:
    f.write(notes)

print(f"Experiment created at: {base_dir}")

import argparse
import os
import json
from datetime import datetime
import itertools
import numpy as np

# ====== USER INPUT ======
experiment_name = (
    "finalrun_budg100mdimcodimall"
)
description = "Budget =100 simplex gradients. codim = 0,2,4,8,16,32. mdim = 2,4,8,1632. (mdim, codim) pairs are all combinations of these values."


pb_parameters = {
    "seed": [22],
    "mdimcodim": [
        list(pair)
        for pair in itertools.product(
            [2, 4, 8, 16, 32], [0,2,4,8,16,32]
        )  
    ],
    "pbtype": [
        "linear_extrinsic_barycenter",
        "linear_intrinsic_barycenter",
        "linear_extrinsic_quadratic",
        "eigh",
    ],
    "instance": np.arange(0, 100, 1).tolist(),
}


algo_parameters = {
    "psstype": [1, 2, 3],
    "projection": [0, 1],
    "rotation": [0, 1],
    "simplexbudget": [100],  # number of simplex gradients
    "euclsimplex": [0],  # 0 means riemannian simplex, 1 euclidean.
    "gamma": [0.5],
    "Gamma": [2.0],
    "alpha_0": [1.0],
    "alpha_max": [1.0],
    "c": [1.0],
}


# =======================

parser = argparse.ArgumentParser()
parser.add_argument("--nodate", action="store_true", help="Omit date prefix from experiment folder name (use for final experiments)")
args = parser.parse_args()

date_str = datetime.now().strftime("%Y-%m-%d")
exp_id = f"{experiment_name}_v1" if args.nodate else f"{date_str}_{experiment_name}_v1"

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

# ---- config.json  ----
with open(os.path.join(base_dir, "config.json"), "w") as f:
    json.dump({"experiments": []}, f, indent=2)

pb_keys = pb_parameters.keys()
pb_values = pb_parameters.values()
algo_keys = algo_parameters.keys()
algo_values = algo_parameters.values()
vpb_parameters = []
for idx, combo in enumerate(itertools.product(*pb_values)):
    instance = dict(zip(pb_keys, combo))
    instance["mdimcodim"] = list(instance["mdimcodim"])
    instance["idpb"] = idx
    instance["mdim"] = instance["mdimcodim"][0]
    instance["codim"] = instance["mdimcodim"][1]
    instance["codim_rendering"] = instance["codim"]
    instance["mdim_rendering"] = instance["mdim"]
    instance["mdimcodim_rendering"] = [
        instance["mdim_rendering"],
        instance["codim_rendering"],
    ]

    if (
        instance["codim"] == 0 and instance["pbtype"] == "eigh"
    ):  
        instance["codim"] = 1
        instance["mdimcodim"] = list(instance["mdimcodim"])
        instance["mdimcodim"][1] = 1
    instance["adim"] = instance["mdim"] + instance["codim"]

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
    f"Generated {len(vpb_parameters)} problem and {len(valgo_parameters)} algorithm configurations."
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

import os
import json
from datetime import datetime

# ====== USER INPUT ======
experiment_name = "dim5_codim4_budg100"
description = "Benchmark on linear subspace and sphere problems, budget = 100 simplex gradients."

# Each (maniftype, obj) pair is one sub-experiment (one call to saveperform_ds).
# maniftype: 1 = random linear subspace, 1.5 = horizontal subspace, 2 = sphere
# obj:       1 = barycenter in subspace, 2 = barycenter in ambient space, 3 = quadratic
maniftypeobj = [
    (1,   1),
    (1,   2),
    (1.5, 2),
    (1,   3),
    (2,   1),
]

pb_parameters = {
    "mdims": [2, 4, 8, 16, 32],
    "codims": [0, 2, 4, 8, 16, 32],
    "nbinstances": 100,
    "seed": 42,
}

algo_parameters = {
    "N": 100,
    "euclsimplex": 1,
    "gamma": 0.5,
    "Gamma": 2.0,
    "alpha_0": 1.0,
    "alpha_max": 1.0,
    "c": 1.0,
}
# =======================

date_str = datetime.now().strftime("%Y-%m-%d")
exp_id = f"{date_str}_{experiment_name}_v1"
base_dir = os.path.join("dimension_influence", "experiments", exp_id)
os.makedirs(base_dir, exist_ok=True)

# Results will be saved here by saveperform_ds (one .pkl pair per sub-experiment).
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

# ---- metadata.json ----
metadata = {
    "experiment_name": experiment_name,
    "date": date_str,
    "description": description,
    "maniftypeobj": maniftypeobj,
    "pb_parameters": pb_parameters,
    "algo_parameters": algo_parameters,
    "tags": [experiment_name],
    "version": "v1",
}

with open(os.path.join(base_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ---- config.json ----
# One entry per (maniftype, obj) pair — each maps to one saveperform_ds call.
sub_experiments = []
for idx, (maniftype, obj) in enumerate(maniftypeobj):
    sub_experiments.append({
        "id": idx,
        "maniftype": maniftype,
        "obj": obj,
        **pb_parameters,
        **algo_parameters,
        "results_dir": results_dir,
    })

config = {"sub_experiments": sub_experiments}

with open(os.path.join(base_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# ---- notes.md ----
notes = f"""# {exp_id}

## Description
{description}

## Problem families
{chr(10).join(f"  - maniftype={mf}, obj={ob}" for mf, ob in maniftypeobj)}

## Problem parameters
{json.dumps(pb_parameters, indent=2)}

## Algorithm parameters
{json.dumps(algo_parameters, indent=2)}

## Notes
- Add observations here.
"""

with open(os.path.join(base_dir, "notes.md"), "w") as f:
    f.write(notes)

print(f"Experiment '{exp_id}' created at: {base_dir}")
print(f"  {len(sub_experiments)} sub-experiments (one per maniftype/obj pair).")
print(f"  Results will be saved to: {results_dir}")

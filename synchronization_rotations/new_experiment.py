import os
import json
from datetime import datetime
from os import path

from numpy import arange

# ====== USER INPUT ======
experiment_name = "matrixsizek2"
description = "This is a first test to check if the pipeline works."
pb_parameters = {
    "vseed": arange(0, 10, 1).tolist(),
    "vn": [2],
    "vd": [3, 5, 10, 20],
}

algo_parameters = {
    "vpsstype": [1, 2, 3],
    "vproj": [0, 1],
    "vrot": [0, 1],
    "vbudg": [100],  # number of simplex gradients
    "veuclsimpl": [0],  # 0 means riemannian simplex, 1 euclidean.
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
generate_script = f"""import json
import itertools
import subprocess

pb_parameters = {pb_parameters}
algo_parameters = {algo_parameters}

pb_keys = pb_parameters.keys()
pb_values = pb_parameters.values()
algo_keys = algo_parameters.keys()
algo_values = algo_parameters.values()

vpb_parameters = []
for idx, combo in enumerate(itertools.product(*pb_values)):
    instance = dict(zip(pb_keys, combo))
    instance["id"] = idx
    vpb_parameters.append(instance)

valgo_parameters = []
for idx, combo in enumerate(itertools.product(*algo_values)):
    instance = dict(zip(algo_keys, combo))
    instance["id"] = idx
    valgo_parameters.append(instance)

config = {{"pb_parameters": vpb_parameters, "algo_parameters": valgo_parameters}}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Generated {{len(vpb_parameters)}} problem and {{len(valgo_parameters)}} algorithm configurations.")
"""

with open(os.path.join(base_dir, "generate_config.py"), "w") as f:
    f.write(generate_script.strip())

path_to_generate = os.path.join(base_dir, "generate_config.py")

# ---- IGNORE ---- There exists a global run.py
# # ---- run.py ----
# run_script = """
# import argparse
# import json
# from os import path
# import time
# import os
# import sys
# from datetime import datetime
# from main import run_experiment


# class Tee:
#     def __init__(self, filename):
#         self.file = open(filename, "w")
#         self.stdout = sys.stdout

#     def write(self, message):
#         self.stdout.write(message)
#         self.file.write(message)
#         self.stdout.flush()
#         self.file.flush()

#     def flush(self):
#         self.stdout.flush()
#         self.file.flush()


# def setup_auto_log(logdir="results"):
#     sys.path.insert(0, "..")

#     os.makedirs(logdir, exist_ok=True)
#     starttime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
#     filename = f"run{starttime}.log"
#     logpath = os.path.join(logdir, filename)

#     sys.stdout = Tee(logpath)
#     sys.stderr = sys.stdout

#     return logdir, logpath, starttime


# def main():
#     # parser = argparse.ArgumentParser(description="Run experiment")

#     with open("config.json", "r") as f:
#         config = json.load(f)
#     experiment = config["experiments"].copy()  # get experiment config

#     logdir, logpath, starttime = setup_auto_log()

#     config["logpath"] = logpath
#     config["starttime"] = starttime
#     configcopyname = f"run{starttime}.json"

#     configcopypath = os.path.join(logdir, configcopyname)
#     with open(configcopypath, "w") as f:
#         json.dump(config, f, indent=2)

#     print("Running experiment:", experiment)
#     run_experiment(experiment)


# if __name__ == "__main__":
#     main()

# """

# with open(os.path.join(base_dir, "run.py"), "w") as f:
#     f.write(run_script.strip())

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

print(f"✅ Experiment created at: {base_dir}")

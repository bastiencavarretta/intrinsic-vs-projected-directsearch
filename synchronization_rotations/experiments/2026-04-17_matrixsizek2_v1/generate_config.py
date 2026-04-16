import json
import itertools
import subprocess

pb_parameters = {'vseed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'vn': [2], 'vd': [3, 5, 10, 20]}
algo_parameters = {'vpsstype': [1, 2, 3], 'vproj': [0, 1], 'vrot': [0, 1], 'vbudg': [100], 'veuclsimpl': [0]}

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

config = {"pb_parameters": vpb_parameters, "algo_parameters": valgo_parameters}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Generated {len(vpb_parameters)} problem and {len(valgo_parameters)} algorithm configurations.")
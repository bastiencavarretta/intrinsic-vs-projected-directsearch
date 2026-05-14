import argparse
import json
from os import path
import pickle
import time
import os
import sys
from datetime import datetime
from tracemalloc import start
import pandas as pd

from code.synchronization_rotations.ProblemSynchRotations import ProblemSynchRotations
from code.synchronization_rotations.directsearch import directsearch


class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.stdout.flush()
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def setup_auto_log(axpdir, logsubdir="results"):
    logdir = os.path.join(axpdir, logsubdir)
    os.makedirs(logdir, exist_ok=True)
    starttime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    filename = f"run{starttime}.log"
    logpath = os.path.join(logdir, filename)

    sys.stdout = Tee(logpath)
    sys.stderr = sys.stdout

    return logdir, logpath, starttime


def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--exppath", type=str, required=True, help="Path to experience directory"
    )

    args = parser.parse_args()
    with open(os.path.join(args.exppath, "config.json"), "r") as f:
        config = json.load(f)
    logdir, logpath, starttime = setup_auto_log(args.exppath)

    config["logpath"] = logpath
    config["starttime"] = starttime
    configcopyname = f"run{starttime}.json"

    configcopypath = os.path.join(logdir, configcopyname)
    with open(configcopypath, "w") as f:
        json.dump(config, f, indent=2)

    pb_parameters = config["pb_parameters"]
    algo_parameters = config["algo_parameters"]
    # device = config["device"]
    # sanitytest = config["sanitytest"]
    results = []

    for pb_parameter in pb_parameters:
        print("=====================================")
        print("Setting problem:", pb_parameter)
        print("=====================================")
        pb_parameter_noid = {
            k: v for k, v in pb_parameter.items() if k != "idpb"
        }  # withdraw the identification key of algo_parameter
        problem = ProblemSynchRotations(**pb_parameter_noid)

        for algo_parameter in algo_parameters:
            parameter = {**pb_parameter, **algo_parameter}
            start_time = time.time()
            print("Running algo with parameters:", parameter)
            algo_parameter_noid = {
                k: v for k, v in algo_parameter.items() if k != "idalgo"
            }  # withdraw the identification key of algo_parameter
            result = directsearch(problem, **algo_parameter_noid)

            end_time = time.time()
            result = {**parameter, **result}
            result["duration"] = end_time - start_time
            result["problem"] = problem
            results.append(result)

    df = pd.DataFrame(results)
    results_path = os.path.join(logdir, f"results_{starttime}.pkl")
    df.to_pickle(results_path)
    print("Results saved to:", results_path)

    print(
        "Experiment completed\n",
        "Start:",
        starttime,
        "\n",
        "End:",
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        "\n",
        # "Result below: \n ",
    )


if __name__ == "__main__":
    main()

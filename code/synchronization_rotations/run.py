import argparse
import json
import time
import os
import sys
import numpy as np
import pandas as pd

from ProblemSynchRotations import ProblemSynchRotations
from directsearch import directsearch


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


def setup_auto_log(axpdir, logsubdir="results", suffix=""):
    logdir = os.path.join(axpdir, logsubdir)
    os.makedirs(logdir, exist_ok=True)
    starttime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    filename = f"run{starttime}{suffix}.log"
    logpath = os.path.join(logdir, filename)

    sys.stdout = Tee(logpath)
    sys.stderr = sys.stdout

    return logdir, logpath, starttime


def main():
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument(
        "--exppath", type=str, required=True, help="Path to experience directory"
    )
    parser.add_argument(
        "--dryrun", action="store_true",
        help="Run with 2 instances and budget 2; results saved with _dryrun suffix"
    )

    args = parser.parse_args()
    suffix = "_dryrun" if args.dryrun else ""
    with open(os.path.join(args.exppath, "config.json"), "r") as f:
        config = json.load(f)
    logdir, logpath, starttime = setup_auto_log(args.exppath, suffix=suffix)

    config["logpath"] = logpath
    config["starttime"] = starttime
    configcopyname = f"run{starttime}{suffix}.json"

    configcopypath = os.path.join(logdir, configcopyname)
    with open(configcopypath, "w") as f:
        json.dump(config, f, indent=2)

    pb_parameters = config["pb_parameters"]
    algo_parameters = config["algo_parameters"]

    if args.dryrun:
        kept_instances = sorted({p["instance"] for p in pb_parameters})[:2]
        pb_parameters = [p for p in pb_parameters if p["instance"] in kept_instances]
        for ap in algo_parameters:
            ap["simplexbudget"] = 2

    results = []

    for pb_parameter in pb_parameters:
        print("=====================================")
        print("Setting problem:", pb_parameter)
        print("=====================================")
        pb_seed = pb_parameter["seed"] + pb_parameter["idpb"]
        pb_parameter_noid = {
            k: v for k, v in pb_parameter.items() if k not in ("idpb", "instance")
        }
        pb_parameter_noid["seed"] = pb_seed
        problem = ProblemSynchRotations(**pb_parameter_noid)

        for algo_parameter in algo_parameters:
            parameter = {**pb_parameter, **algo_parameter}
            start_time = time.time()
            print("Running algo with parameters:", parameter)
            algo_parameter_noid = {
                k: v for k, v in algo_parameter.items() if k != "idalgo"
            }  # withdraw the identification key of algo_parameter
            algo_seed = (
                pb_parameter["seed"]
                + pb_parameter["idpb"] * 10000
                + algo_parameter["idalgo"]
            )
            np.random.seed(algo_seed)
            result = directsearch(problem, **algo_parameter_noid)

            end_time = time.time()
            result = {**parameter, **result}
            result["duration"] = end_time - start_time
            result["problem"] = problem
            results.append(result)

    df = pd.DataFrame(results)
    suffix = "_dryrun" if args.dryrun else ""
    results_path = os.path.join(logdir, f"results_{starttime}{suffix}.pkl")
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
    )


if __name__ == "__main__":
    main()

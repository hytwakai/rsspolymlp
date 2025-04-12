"""
Script for performing Random Structure Search (RSS) on multiple tasks in parallel
using polynomial machine learinig potentials (MLPs).
"""

import glob
import multiprocessing
import os
import subprocess
import time

import joblib

from rsspolymlp.parse_arg import ParseArgument
from rsspolymlp.rss_mlp import RandomStructureOptimization


def run():
    args = ParseArgument.get_parallelization_args()

    os.makedirs("log", exist_ok=True)
    os.makedirs("opt_struct", exist_ok=True)
    with open("finish.log", "a") as _, open("success.log", "a") as _:
        pass

    # Check which structures have already been optimized
    poscar_path_all = glob.glob("initial_struct/*")
    poscar_path_all = sorted(poscar_path_all, key=lambda x: int(x.split("_")[-1]))
    with open("finish.log") as f:
        finished_set = set(line.strip() for line in f)
    poscar_path_all = [
        p for p in poscar_path_all if os.path.basename(p) not in finished_set
    ]

    if args.parallel_method == "joblib":
        # Perform parallel optimization with joblib
        time_pra = time.time()
        rssobj = RandomStructureOptimization(args)
        joblib.Parallel(n_jobs=args.num_process, backend=args.backend)(
            joblib.delayed(rssobj.run_optimization)(poscar)
            for poscar in poscar_path_all
        )
        time_pra_fin = time.time() - time_pra

        # Log computational times
        core = multiprocessing.cpu_count()
        if len(poscar_path_all) > 0:
            with open("parallel_time.log", "a") as f:
                print("Number of CPU cores:", core, file=f)
                print("Number of the structures:", len(glob.glob("log/*")), file=f)
                print("Computational time:", time_pra_fin, file=f)
                print("", file=f)

    if args.parallel_method == "srun":
        if args.num_process == -1:
            args.num_process = multiprocessing.cpu_count()
        if len(poscar_path_all) > args.num_process:
            with open("multiprocess.sh", "w") as f:
                print("#!/bin/bash", file=f)
                print(": > start.log", file=f)
                print("case $SLURM_PROCID in", file=f)
                for i in range(args.num_process):
                    first_poscar = i
                    run_ = (
                        f"rss-single-srun "
                        f"--pot {' '.join(args.pot)} "
                        f"--num_opt_str {args.num_opt_str} "
                        f"--pressure {args.pressure} "
                        f"--solver_method {args.solver_method} "
                        f"--maxiter {args.maxiter} "
                        f"--first_poscar {first_poscar} "
                    )
                    print(f"    {i}) {run_} ;;", file=f)
                print("esac", file=f)
                print("rm start.log", file=f)
            subprocess.run(["chmod", "+x", "./multiprocess.sh"], check=True)


def run_single_srun():
    args = ParseArgument.get_optimization_args()

    poscar_path_all = glob.glob("initial_struct/*")
    poscar_path_all = sorted(poscar_path_all, key=lambda x: int(x.split("_")[-1]))
    with open("finish.log") as f:
        finished_set = set(line.strip() for line in f)
    poscar_list = [
        p for p in poscar_path_all if os.path.basename(p) not in finished_set
    ]

    first_run = True
    rssobj = RandomStructureOptimization(args)
    while True:
        if first_run:
            poscar_path = poscar_list[args.first_poscar]
            first_run = False
        else:
            poscar_path = poscar_list[0]
        with open("start.log", "a") as f:
            print(os.path.basename(poscar_path), file=f)

        rssobj.run_optimization(poscar_path)

        time.sleep(1)

        finished_set = set()
        for log in ["finish.log", "start.log"]:
            if os.path.exists(log):
                with open(log) as f:
                    finished_set.update(line.strip() for line in f)
        poscar_list = [
            p for p in poscar_list if os.path.basename(p) not in finished_set
        ]

        if not poscar_list:
            print("All POSCAR files have been processed.")
            break

        with open("success.log") as f:
            success_str = sum(1 for _ in f)
        residual_str = args.num_opt_str - success_str
        if residual_str <= 0:
            print("Reached the target number of optimized structures.")
            break

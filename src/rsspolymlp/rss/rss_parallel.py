"""
Script for performing Random Structure Search (RSS) on multiple tasks in parallel
using polynomial machine learinig potentials (MLPs).
"""

import fcntl
import glob
import multiprocessing
import os
import subprocess
import time

import joblib

from rsspolymlp.rss.rss_mlp import RandomStructureSearch
from rsspolymlp.common.parse_arg import ParseArgument


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
        rssobj = RandomStructureSearch(args)
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
            with open("start.log", "w") as f:
                pass
            with open("multiprocess.sh", "w") as f:
                print("#!/bin/bash", file=f)
                print("case $SLURM_PROCID in", file=f)
                for i in range(args.num_process):
                    run_ = (
                        f"rss-single-srun "
                        f"--pot {' '.join(args.pot)} "
                        f"--num_opt_str {args.num_opt_str} "
                        f"--pressure {args.pressure} "
                        f"--solver_method {args.solver_method} "
                        f"--maxiter {args.maxiter} "
                    )
                    if args.not_stop_rss:
                        run_ += "--not_stop_rss"
                    print(f"    {i}) {run_} ;;", file=f)
                print("esac", file=f)
                print("rm start.log", file=f)
            subprocess.run(["chmod", "+x", "./multiprocess.sh"], check=True)


def run_single_srun():

    def acquire_lock():
        lock_file = open("rss.lock", "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return lock_file

    def release_lock(lock_file):
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    args = ParseArgument.get_optimization_args()

    poscar_path_all = glob.glob("initial_struct/*")
    poscar_path_all = sorted(poscar_path_all, key=lambda x: int(x.split("_")[-1]))
    poscar_list = [p for p in poscar_path_all if os.path.basename(p)]

    rssobj = RandomStructureSearch(args)

    while True:
        lock = acquire_lock()

        finished_set = set()
        for log in ["finish.log", "start.log"]:
            if os.path.exists(log):
                with open(log) as f:
                    finished_set.update(line.strip() for line in f)
        poscar_list = [
            p for p in poscar_list if os.path.basename(p) not in finished_set
        ]

        if not poscar_list:
            release_lock(lock)
            print("All POSCAR files have been processed.")
            break

        poscar_path = poscar_list[0]
        with open("start.log", "a") as f:
            print(os.path.basename(poscar_path), file=f)

        release_lock(lock)

        with open("success.log") as f:
            success_str = sum(1 for _ in f)
        residual_str = args.num_opt_str - success_str
        if residual_str <= 0:
            print("Reached the target number of optimized structures.")
            break

        rssobj.run_optimization(poscar_path)

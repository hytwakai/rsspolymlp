import fcntl
import glob
import multiprocessing
import os
import subprocess
import time

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from rsspolymlp.rss.eliminate_duplicates import RSSResultAnalyzer
from rsspolymlp.rss.optimization_mlp import RandomStructureSearch
from rsspolymlp.rss.random_struct import GenerateRandomStructure


def rss_init_struct(
    elements,
    atom_counts,
    num_init_str=5000,
    least_distance=0.0,
    min_volume=0,
    max_volume=100,
    output_dir="initial_struct",
):
    os.makedirs(output_dir, exist_ok=True)
    pre_str_count = len(glob.glob(os.path.join(output_dir, "*")))
    if num_init_str > pre_str_count:
        gen_str = GenerateRandomStructure(
            elements,
            atom_counts,
            num_init_str,
            least_distance=least_distance,
            pre_str_count=pre_str_count,
        )
        gen_str.random_structure(min_volume=min_volume, max_volume=max_volume)


def rss_parallel(
    pot="polymlp.yaml",
    pressure=0.0,
    solver_method="CG",
    maxiter=100,
    num_opt_str=1000,
    not_stop_rss=False,
    parallel_method="joblib",
    num_process=-1,
    backend="loky",
):
    """
    Performing Random Structure Search (RSS) on multiple tasks in parallel
    using polynomial machine learinig potentials (MLPs).
    """
    os.makedirs("rss_result", exist_ok=True)
    for file in ["rss_result/finish.dat", "rss_result/success.dat"]:
        open(file, "a").close()

    with open("rss_result/success.dat") as f:
        success_set = [line.strip() for line in f]
    if len(success_set) >= num_opt_str:
        print("Target number of optimized structures reached. Exiting.")
        return

    # Check which structures have already been optimized
    poscar_path_all = sorted(
        glob.glob("initial_struct/*"), key=lambda x: int(x.split("_")[-1])
    )
    with open("rss_result/finish.dat") as f:
        finished_set = set(line.strip() for line in f)
    poscar_path_all = [
        p for p in poscar_path_all if os.path.basename(p) not in finished_set
    ]

    rssobj = RandomStructureSearch(
        pot=pot,
        pressure=pressure,
        solver_method=solver_method,
        maxiter=maxiter,
        num_opt_str=num_opt_str,
        not_stop_rss=not_stop_rss,
    )
    if num_process == -1:
        num_process = multiprocessing.cpu_count()

    if parallel_method == "joblib":
        # Perform parallel optimization with joblib
        time_start = time.time()
        Parallel(n_jobs=num_process, backend=backend)(
            delayed(rssobj.run_optimization)(poscar) for poscar in poscar_path_all
        )
        executor = get_reusable_executor(max_workers=num_process)
        executor.shutdown(wait=True)
        elapsed = time.time() - time_start
        with open("rss_result/parallel_time.log", "a") as f:
            print("Number of CPU cores:", num_process, file=f)
            print("Number of the structures:", len(glob.glob("log/*")), file=f)
            print("Computational time:", elapsed, file=f)
            print("", file=f)

    elif parallel_method == "srun":
        if len(poscar_path_all) > num_process:
            with open("rss_result/start.dat", "w") as f:
                pass
            with open("multiprocess.sh", "w") as f:
                print("#!/bin/bash", file=f)
                print("case $SLURM_PROCID in", file=f)
                for i in range(num_process):
                    run_ = (
                        f"rss-single-srun --pot {' '.join(pot)} "
                        f"--num_opt_str {num_opt_str} "
                        f"--pressure {pressure} "
                        f"--solver_method {solver_method} "
                        f"--maxiter {maxiter} "
                    )
                    if not_stop_rss:
                        run_ += " --not_stop_rss"
                    print(f"    {i}) {run_} ;;", file=f)
                print("esac", file=f)
                print("rm rss_result/start.dat", file=f)
            subprocess.run(["chmod", "+x", "./multiprocess.sh"], check=True)


def rss_single_srun(
    pot="polymlp.yaml",
    pressure=0.0,
    solver_method="CG",
    maxiter=100,
    num_opt_str=1000,
    not_stop_rss=False,
):
    def acquire_lock():
        lock_file = open("rss.lock", "w")
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return lock_file

    def release_lock(lock_file):
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    poscar_path_all = sorted(
        glob.glob("initial_struct/*"), key=lambda x: int(x.split("_")[-1])
    )
    poscar_list = [p for p in poscar_path_all if os.path.basename(p)]
    rssobj = RandomStructureSearch(
        pot=pot,
        pressure=pressure,
        solver_method=solver_method,
        maxiter=maxiter,
        num_opt_str=num_opt_str,
        not_stop_rss=not_stop_rss,
    )

    while True:
        lock = acquire_lock()

        finished_set = set()
        for log in ["rss_result/finish.dat", "rss_result/start.dat"]:
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
        with open("rss_result/start.dat", "a") as f:
            print(os.path.basename(poscar_path), file=f)

        release_lock(lock)

        with open("rss_result/success.dat") as f:
            success_str = sum(1 for _ in f)
        residual_str = num_opt_str - success_str
        if residual_str <= 0:
            print("Reached the target number of optimized structures.")
            break

        rssobj.run_optimization(poscar_path)


def rss_uniq_struct(
    num_str=-1,
    cutoff=None,
    pressure=None,
    use_joblib=True,
    num_process=-1,
    backend="loky",
):
    analyzer = RSSResultAnalyzer()
    if cutoff is not None:
        analyzer.cutoff = cutoff
    if pressure is not None:
        analyzer.pressure = pressure

    analyzer.run_rss_uniq_struct(
        num_str=num_str,
        use_joblib=use_joblib,
        num_process=num_process,
        backend=backend,
    )

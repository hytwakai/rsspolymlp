import fcntl
import glob
import multiprocessing
import os
import subprocess
import time

from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

from rsspolymlp.analysis.ghost_minima import (
    detect_actual_ghost_minima,
    ghost_minima_candidates,
)
from rsspolymlp.analysis.phase_analysis import ConvexHullAnalyzer
from rsspolymlp.analysis.rss_summarize import RSSResultSummarizer
from rsspolymlp.rss.eliminate_duplicates import RSSResultAnalyzer
from rsspolymlp.rss.optimization_mlp import RandomStructureSearch
from rsspolymlp.rss.random_struct import GenerateRandomStructure


def rss_init_struct(
    elements,
    atom_counts,
    n_init_str=5000,
    min_volume=0,
    max_volume=100,
    least_distance=0.0,
):
    gen_str = GenerateRandomStructure(
        element_list=elements,
        atom_counts=atom_counts,
        min_volume=min_volume,
        max_volume=max_volume,
        least_distance=least_distance,
    )
    gen_str.random_structure(max_init_struct=n_init_str)


def rss_run_parallel(
    pot="polymlp.yaml",
    pressure=0.0,
    solver_method="CG",
    c_maxiter=100,
    n_opt_str=1000,
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
    if len(success_set) >= n_opt_str:
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

    if len(poscar_path_all) == 0:
        return

    rssobj = RandomStructureSearch(
        pot=pot,
        pressure=pressure,
        solver_method=solver_method,
        c_maxiter=c_maxiter,
        n_opt_str=n_opt_str,
        not_stop_rss=not_stop_rss,
    )
    if num_process == -1:
        num_process = multiprocessing.cpu_count()

    if parallel_method == "joblib":
        time_start = time.time()
        if num_process == 1:
            for poscar in poscar_path_all:
                rssobj.run_optimization(poscar)
        else:
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
                        f"rsspolymlp --rss_single "
                        f"--pot {' '.join(pot)} "
                        f"--n_opt_str {n_opt_str} "
                        f"--pressure {pressure} "
                        f"--solver_method {solver_method} "
                        f"--c_maxiter {c_maxiter} "
                    )
                    if not_stop_rss:
                        run_ += " --not_stop_rss"
                    print(f"    {i}) {run_} ;;", file=f)
                print("esac", file=f)
                print("rm rss_result/start.dat", file=f)
            subprocess.run(["chmod", "+x", "./multiprocess.sh"], check=True)


def rss_run_single(
    pot="polymlp.yaml",
    pressure=0.0,
    solver_method="CG",
    c_maxiter=100,
    n_opt_str=1000,
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
        c_maxiter=c_maxiter,
        n_opt_str=n_opt_str,
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
        residual_str = n_opt_str - success_str
        if residual_str <= 0:
            print("Reached the target number of optimized structures.")
            break

        rssobj.run_optimization(poscar_path)


def rss_uniq_struct(
    num_str=-1,
    cutoff=None,
    use_joblib=True,
    num_process=-1,
    backend="loky",
):
    analyzer = RSSResultAnalyzer()
    if cutoff is not None:
        analyzer.cutoff = cutoff

    analyzer.run_rss_uniq_struct(
        num_str=num_str,
        use_joblib=use_joblib,
        num_process=num_process,
        backend=backend,
    )


def rss_polymlp(
    elements,
    atom_counts,
    pot="polymlp.yaml",
    pressure=0.0,
    c_maxiter=100,
    n_opt_str=1000,
    max_init_str=None,
    min_volume=0,
    max_volume=100,
    least_distance=0.0,
    solver_method="CG",
    not_stop_rss=False,
    num_process=-1,
    backend="loky",
    output_dir="./",
):
    if not output_dir == "./":
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)

    if max_init_str is None and n_opt_str is None:
        raise ValueError
    elif max_init_str is None:
        max_init_str = n_opt_str * 10
    elif n_opt_str is None:
        n_opt_str = max_init_str

    n_init_str = 0

    while True:
        n_init_str += n_opt_str
        if n_init_str > max_init_str:
            print(
                "All randomly generated initial structures have been processed. Stopping."
            )
            break

        rss_init_struct(
            elements=elements,
            atom_counts=atom_counts,
            n_init_str=n_init_str,
            min_volume=min_volume,
            max_volume=max_volume,
            least_distance=least_distance,
        )

        rss_run_parallel(
            pot=pot,
            pressure=pressure,
            solver_method=solver_method,
            c_maxiter=c_maxiter,
            n_opt_str=n_opt_str,
            not_stop_rss=not_stop_rss,
            num_process=num_process,
            backend=backend,
        )

        with open("rss_result/success.dat") as f:
            success_set = [line.strip() for line in f]
        if len(success_set) >= n_opt_str:
            print("Target number of optimized structures reached. Exiting.")
            break

    rss_uniq_struct()


def rss_summarize(
    elements,
    result_paths,
    use_joblib=True,
    num_process=-1,
    backend="loky",
    symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    output_poscar: bool = False,
    thresholds: list[float] = None,
    parse_vasp: bool = False,
    summarize_p: bool = False,
    update_result: bool = False,
):
    analyzer = RSSResultSummarizer(
        elements=elements,
        result_paths=result_paths,
        use_joblib=use_joblib,
        num_process=num_process,
        backend=backend,
        symprec_set=symprec_set,
        output_poscar=output_poscar,
        thresholds=thresholds,
        parse_vasp=parse_vasp,
        update_result=update_result,
    )
    if not summarize_p:
        analyzer.run_summarize()
    else:
        analyzer.run_summarize_p()


def rss_ghost_minima_cands(result_paths):
    dir_path = os.path.dirname(result_paths[0])
    os.makedirs(f"{dir_path}/../ghost_minima/ghost_minima_candidates", exist_ok=True)
    os.chdir(f"{dir_path}/../")
    ghost_minima_candidates(result_paths)


def rss_ghost_minima_validate(dft_dir):
    detect_actual_ghost_minima(dft_dir)


def rss_phase_analysis(
    elements, input_paths, ghost_minima_file=None, parse_vasp=False, thresholds=None
):
    ch_analyzer = ConvexHullAnalyzer(elements=elements)
    ch_analyzer.parse_results(
        input_paths=input_paths,
        ghost_minima_file=ghost_minima_file,
        parse_vasp=parse_vasp,
    )
    ch_analyzer.run_analysis()

    if thresholds is not None:
        for threshold in thresholds:
            ch_analyzer.get_structures_near_hull(threshold)

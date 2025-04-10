import os


def finished_log(poscar, is_log_files):
    """Check if the optimization process for a given structure has finished."""
    logfile = os.path.basename(poscar) + ".log"
    if logfile not in is_log_files:
        return False
    try:
        with open("log/" + logfile, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            while pos > 0:
                pos -= 1
                f.seek(pos, os.SEEK_SET)
                if f.read(1) == b"\n":
                    f.seek(pos + 1, os.SEEK_SET)
                    last_line = f.readline().decode().strip()
                    if last_line:
                        break

        return last_line.startswith("Finished")
    except Exception:
        return False


if __name__ == "__main__":
    """Main script for running Random Structure Search (RSS) using polynomial MLPs."""
    import argparse
    import glob
    import multiprocessing
    import time

    import joblib
    import numpy as np

    from rsspolymlp.initial_str import GenerateInitialStructure
    from rsspolymlp.rss_mlp import RandomStructureOptimization
    from rsspolymlp.sorting_str import SortStructure

    np.set_printoptions(legacy="1.21")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements", type=str, nargs="+", default=None, help="List of element symbols"
    )
    parser.add_argument(
        "--n_atoms",
        type=int,
        nargs="+",
        default=None,
        help="Number of atoms for each element",
    )
    parser.add_argument(
        "--max_opt_str",
        type=int,
        default=1000,
        help="Maximum number of optimized structures obtained from RSS",
    )
    parser.add_argument(
        "--least_distance",
        type=float,
        default=0.5,
        help="Minimum interatomic distance in initial structure (angstrom)",
    )
    parser.add_argument(
        "--max_volume",
        type=float,
        default=100,
        help="Maximum volume of initial structure (A^3/atom)",
    )
    parser.add_argument(
        "--min_volume",
        type=float,
        default=0,
        help="Minimum volume of initial structure (A^3/atom)",
    )
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.yaml",
        help="Potential file for polynomial MLP",
    )
    parser.add_argument(
        "--pressure", type=float, default=0, help="Pressure term (in GPa)"
    )
    parser.add_argument("--method", type=str, default="CG", help="Type of solver")
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100,
        help="Maximum number of iterations when c1 and c2 values are changed",
    )
    args = parser.parse_args()
    args.max_init_str = args.max_opt_str * 5

    os.makedirs("initial_str", exist_ok=True)
    os.makedirs("optimized_str", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    with open("success.log", "a") as f:
        pass

    # Generate new structures if necessary
    pre_str_count = len(glob.glob("initial_str/*"))
    if args.max_init_str > pre_str_count:
        elements = args.elements

        # Creating initial random structures
        gen_str = GenerateInitialStructure(
            elements,
            args.n_atoms,
            args.max_init_str,
            least_distance=args.least_distance,
            pre_str_count=pre_str_count,
        )
        gen_str.random_structure(min_volume=args.min_volume, max_volume=args.max_volume)

    poscar_path_all = glob.glob("initial_str/*")
    poscar_path_all = sorted(poscar_path_all, key=lambda x: int(x.split("_")[-1]))

    # Check which structures have already been optimized
    is_log_files = {os.path.basename(f): f for f in glob.glob("log/*.log")}
    poscar_path_all = [p for p in poscar_path_all if not finished_log(p, is_log_files)]

    # Perform parallel optimization
    time_pra = time.time()
    rssobj = RandomStructureOptimization(args)
    joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(rssobj.run_optimization)(poscar) for poscar in poscar_path_all
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

    # Sort the optimized structures and log computational results
    SortStructure().run_sorting(args)

import glob
import multiprocessing
import os
import time
from contextlib import redirect_stdout

import joblib
import numpy as np

from pypolymlp.calculator.str_opt.optimization import GeometryOptimization
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.spglib_utils import SymCell
from rss_polymlp import variable
from rss_polymlp.initial_str import GenerateInitialStructure
from rss_polymlp.sorting_str import SortStructure

np.set_printoptions(legacy="1.21")


def run_optimization(poscat_path, args):
    poscar_name = poscat_path.split("/")[-1]
    output_file = f"log/{poscar_name}.log"

    with open(output_file, "w") as f, redirect_stdout(f):
        time_initial = time.time()
        print("Selected potential:", args.pot)

        unitcell = Poscar(poscat_path).structure
        energy_keep = None

        max_iteration = 10
        for iteration in range(max_iteration):

            minobj = GeometryOptimization(
                unitcell,
                relax_cell=not args.fix_cell,
                relax_volume=not args.fix_volume,
                relax_positions=not args.fix_positions,
                with_sym=args.symmetry,
                pot=args.pot,
                verbose=True,
            )
            if iteration == 0:
                print("Initial structure")
                minobj.print_structure()

            try:
                minobj.run(gtol=1e-6, method="CG")
            except ValueError:
                time_fin = time.time() - time_initial
                print("Final function value (eV/atom):", minobj._energy / len(unitcell.elements))
                print("Geometry optimization failed: Huge negative or zero enegy value.")
                print("Computational time:", time_fin, file=f)
                print("Finished", file=f)
                return

            minobj.write_poscar(filename=f"optimized_str/{poscar_name}")
            sym = SymCell(poscar_name=f"optimized_str/{poscar_name}", symprec=1e-5)
            cell_copy = sym.refine_cell()
            minobj.structure = cell_copy
            minobj.write_poscar(filename=f"optimized_str/{poscar_name}")

            energy_per_atom = minobj.energy / len(unitcell.elements)
            print("Energy (eV/atom):", energy_per_atom)
            if energy_keep is not None:
                energy_convergence = abs(energy_keep - energy_per_atom)
                print("Energy difference from the previous iteration:")
                print(energy_convergence)
                if energy_convergence < 10**-7:
                    print("Final function value (eV/atom):", energy_per_atom)
                    break
            energy_keep = energy_per_atom

            unitcell = Poscar(f"optimized_str/{poscar_name}").structure

            if iteration == max_iteration - 1:
                print("Maximum number of relaxation iterations has been exceeded")

        if not minobj.relax_cell:
            print("Residuals (force):")
            print(minobj.residual_forces.T)
            print(
                "Maximum absolute value in Residuals (force):",
                np.max(np.abs(minobj.residual_forces.T)),
            )
        else:
            res_f, res_s = minobj.residual_forces
            print("Residuals (force):")
            print(res_f.T)
            print("Maximum absolute value in Residuals (force):", np.max(np.abs(res_f.T)))
            print("Residuals (stress):")
            print(res_s)
            print("Maximum absolute value in Residuals (stress):", np.max(np.abs(res_s)))

        print("Final structure")
        minobj.print_structure()
        minobj.write_poscar(filename=f"optimized_str/{poscar_name}")

        spg_sets = []
        for tol in [10**-5, 10**-4, 10**-3, 10**-2]:
            sym = SymCell(poscar_name=f"optimized_str/{poscar_name}", symprec=tol)
            spg = sym.get_spacegroup()
            spg_sets.append(spg)
            print(f"Space group ({tol}):", spg)
        print("Space group set:")
        print(spg_sets)

        time_fin = time.time() - time_initial
        print("Computational time:", time_fin)
        print("Finished")


def is_finished(poscar, is_log_files):
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

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=str, nargs="+", default=None)
    parser.add_argument("--n_atoms", type=int, nargs="+", default=None)
    parser.add_argument("--max_str", type=int, default=1000)
    parser.add_argument("--least_distance", type=float, default=0.5)
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default="polymlp.yaml",
        help="polymlp file",
    )
    parser.add_argument("--fix_cell", action="store_true", help="Fix cell parameters")
    parser.add_argument("--fix_volume", action="store_true", help="Fix volume")
    parser.add_argument("--fix_positions", action="store_true", help="Fix atomic positions")
    parser.add_argument("--symmetry", action="store_true", help="Consider symmetry")
    args = parser.parse_args()

    os.makedirs("initial_str", exist_ok=True)
    os.makedirs("optimized_str", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    max_str = args.max_str
    pre_str_count = len(glob.glob("initial_str/*"))

    if max_str > pre_str_count:
        elements = args.elements
        atomic_length = None
        for element in elements:
            _atomic_length = variable.atom_variable(element)
            if atomic_length is None:
                atomic_length = _atomic_length
            elif atomic_length < _atomic_length:
                atomic_length = _atomic_length

        gen_str = GenerateInitialStructure(
            elements,
            args.n_atoms,
            max_str,
            atomic_length=atomic_length,
            least_distance=args.least_distance,
            pre_str_count=pre_str_count,
        )
        gen_str.random_structure()

    poscar_path_all = glob.glob("initial_str/*")
    poscar_path_all = sorted(poscar_path_all, key=lambda x: int(x.split("_")[-1]))
    is_log_files = {os.path.basename(f): f for f in glob.glob("log/*.log")}
    poscar_path_all = [p for p in poscar_path_all if not is_finished(p, is_log_files)]

    time_pra = time.time()
    joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(run_optimization)(poscar, args) for poscar in poscar_path_all
    )
    time_pra_fin = time.time() - time_pra
    core = multiprocessing.cpu_count()

    if len(poscar_path_all) > 0:
        with open("parallel_time.log", "a") as f:
            print("Number of CPU cores:", core, file=f)
            print("Number of the structures:", len(poscar_path_all), file=f)
            print("Computational time:", time_pra_fin, file=f)
            print("", file=f)

    SortStructure().run_sorting()

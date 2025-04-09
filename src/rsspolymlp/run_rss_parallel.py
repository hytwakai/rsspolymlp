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
from rsspolymlp.initial_str import GenerateInitialStructure
from rsspolymlp.sorting_str import SortStructure

np.set_printoptions(legacy="1.21")


def is_finished(poscar, is_log_files):
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


class RandomStructureOptimization:

    def __init__(self, args):
        # Parsed command-line arguments containing optimization settings.
        self.args = args

    def run_optimization(self, poscar_path):
        """
        Perform geometry optimization on a given random structure using MLP.

        Parameter
        ----------
        poscar_path : str
            Path to the POSCAR file of the structure to be optimized.
        """
        poscar_name = poscar_path.split("/")[-1]
        output_file = f"log/{poscar_name}.log"

        with open(output_file, "w") as f, redirect_stdout(f):
            time_initial = time.time()
            print("Selected potential:", self.args.pot)

            unitcell = Poscar(poscar_path).structure
            energy_keep = None
            max_iteration = 10
            c1_set = [None, 0.9, 0.99]
            c2_set = [None, 0.99, 0.999]

            for iteration in range(max_iteration):
                minobj = self.minimize(unitcell, iteration, c1_set, c2_set)
                if minobj is None:
                    print(
                        "Geometry optimization failed: Huge negative or zero energy value."
                    )
                    self.log_computation_time(time_initial)
                    return

                self.minobj = minobj
                self.target_poscar = f"optimized_str/{poscar_name}"

                self.minobj.write_poscar(filename=self.target_poscar)
                refined_cell = self.refine_structure(self.target_poscar)
                if refined_cell is None:
                    print("Refining cell failed.")
                    self.log_computation_time(time_initial)
                    return
                self.minobj.structure = refined_cell
                self.minobj.write_poscar(filename=self.target_poscar)

                if self.check_convergence(energy_keep):
                    break
                energy_keep = self.minobj.energy / len(unitcell.elements)

                unitcell = Poscar(self.target_poscar).structure

                if iteration == max_iteration - 1:
                    print("Maximum number of relaxation iterations has been exceeded")

            self.print_final_structure_details()
            self.analyze_space_group(self.target_poscar)

            self.log_computation_time(time_initial)

    def minimize(self, unitcell, iteration, c1_set, c2_set):
        """Run geometry optimization with different parameters until successful."""
        minobj = GeometryOptimization(
            unitcell,
            pot=self.args.pot,
            relax_cell=True,
            relax_volume=True,
            relax_positions=True,
            with_sym=False,
            pressure=self.args.pressure,
            verbose=True,
        )
        if iteration == 0:
            print("Initial structure")
            minobj.print_structure()

        maxiter = 10000
        for c_count in range(3):
            if iteration == 0 and c_count <= 1 or iteration == 1 and c_count == 0:
                maxiter = self.args.maxiter
                continue
            try:
                minobj.run(
                    gtol=1e-6,
                    method=self.args.method,
                    maxiter=maxiter,
                    c1=c1_set[c_count],
                    c2=c2_set[c_count],
                )
                return minobj
            except ValueError:
                if c_count == 2:
                    print(
                        "Final function value (eV/atom):",
                        minobj._energy / len(minobj.structure.elements),
                    )
                    return None
                print("Change [c1, c2] to", c1_set[c_count + 1], c2_set[c_count + 1])
                maxiter = 100

    def refine_structure(self, poscar_path):
        """Refine the crystal structure with increasing symmetry precision."""
        symprec_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        for sp in symprec_list:
            try:
                sym = SymCell(poscar_name=poscar_path, symprec=sp)
                return sym.refine_cell()
            except TypeError:
                print("Change symprec to", sp * 10)
        return None

    def check_convergence(self, energy_keep):
        """Check if the energy difference is below the threshold."""
        energy_per_atom = self.minobj.energy / len(self.minobj.structure.elements)
        print("Energy (eV/atom):", energy_per_atom)
        if energy_keep is not None:
            energy_convergence = energy_per_atom - energy_keep
            print("Energy difference from the previous iteration:", energy_convergence)
            if abs(energy_convergence) < 1e-7:
                print("Final function value (eV/atom):", energy_per_atom)
                return True
        return False

    def analyze_space_group(self, poscar_path):
        """Analyze space group symmetry with different tolerances."""
        spg_sets = []
        for tol in [1e-5, 1e-4, 1e-3, 1e-2]:
            try:
                sym = SymCell(poscar_name=poscar_path, symprec=tol)
                spg = sym.get_spacegroup()
                spg_sets.append(spg)
                print(f"Space group ({tol}):", spg)
            except TypeError:
                continue
        print("Space group set:")
        print(spg_sets)

    def print_final_structure_details(self):
        """Print residual forces, stress, and final structure."""
        if not self.minobj.relax_cell:
            print("Residuals (force):")
            print(self.minobj.residual_forces.T)
            print(
                "Maximum absolute value in Residuals (force):",
                np.max(np.abs(self.minobj.residual_forces.T)),
            )
        else:
            res_f, res_s = self.minobj.residual_forces
            print("Residuals (force):")
            print(res_f.T)
            print(
                "Maximum absolute value in Residuals (force):", np.max(np.abs(res_f.T))
            )
            print("Residuals (stress):")
            print(res_s)
            print(
                "Maximum absolute value in Residuals (stress):", np.max(np.abs(res_s))
            )
        print("Final structure")
        self.minobj.print_structure()

    def log_computation_time(self, start_time):
        """Log computational time."""
        time_fin = time.time() - start_time
        print("Computational time:", time_fin)
        print("Finished")


if __name__ == "__main__":
    """Main script for running Random Structure Search (RSS) using polynomial MLPs."""
    import argparse

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
        "--max_str",
        type=int,
        default=1000,
        help="Maximum number of initial structures for RSS",
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

    os.makedirs("initial_str", exist_ok=True)
    os.makedirs("optimized_str", exist_ok=True)
    os.makedirs("log", exist_ok=True)

    # Check the number of pre-existing structures
    max_str = args.max_str
    pre_str_count = len(glob.glob("initial_str/*"))

    # Generate new structures if necessary
    if max_str > pre_str_count:
        elements = args.elements

        # Creating initial random structures
        gen_str = GenerateInitialStructure(
            elements,
            args.n_atoms,
            max_str,
            least_distance=args.least_distance,
            pre_str_count=pre_str_count,
        )
        gen_str.random_structure(max_volume=args.max_volume)

    poscar_path_all = glob.glob("initial_str/*")
    poscar_path_all = sorted(poscar_path_all, key=lambda x: int(x.split("_")[-1]))

    # Check which structures have already been optimized
    is_log_files = {os.path.basename(f): f for f in glob.glob("log/*.log")}
    poscar_path_all = [p for p in poscar_path_all if not is_finished(p, is_log_files)]

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
            print("Number of the structures:", len(poscar_path_all), file=f)
            print("Computational time:", time_pra_fin, file=f)
            print("", file=f)

    # Sort the optimized structures and log computational results
    SortStructure().run_sorting()

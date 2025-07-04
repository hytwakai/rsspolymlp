import glob
import os
import time
from contextlib import redirect_stdout

import numpy as np

from pypolymlp.calculator.str_opt.optimization import GeometryOptimization
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.spglib_utils import SymCell


class RandomStructureSearch:

    def __init__(
        self,
        pot="polymlp.yaml",
        pressure=0.0,
        solver_method="CG",
        maxiter=100,
        num_opt_str=1000,
        not_stop_rss=False,
    ):
        self.pot = pot
        self.pressure = pressure
        self.solver_method = solver_method
        self.maxiter = maxiter
        self.num_opt_str = num_opt_str
        self.not_stop_rss = not_stop_rss
        self._stop_rss = False
        os.makedirs("log", exist_ok=True)
        os.makedirs("opt_struct", exist_ok=True)
        os.makedirs("rss_result", exist_ok=True)

    def run_optimization(self, poscar_path):
        """
        Perform geometry optimization on a given random structure using MLP.

        Parameter
        ----------
        poscar_path : str
            Path to the POSCAR file of the structure to be optimized.
        """
        self.check_opt_str()
        if self._stop_rss:
            return

        self.poscar_name = poscar_path.split("/")[-1]
        self.opt_poscar = f"opt_struct/{self.poscar_name}"
        output_file = f"log/{self.poscar_name}.log"

        with open(output_file, "w") as f, redirect_stdout(f):
            self.time_initial = time.time()
            energy_keep = None
            max_iteration = 10
            c1_set = [None, 0.9, 0.99]
            c2_set = [None, 0.99, 0.999]

            if isinstance(self.pot, list):
                print("Selected potential:", self.pot)
            else:
                print("Selected potential:", [self.pot])
            print("Pressure (GPa):", self.pressure)
            unitcell = Poscar(poscar_path).structure

            for iteration in range(max_iteration):
                self.check_opt_str()
                if self._stop_rss:
                    print("Number of optimized structures has been reached.")
                    break

                minobj = self.minimize(unitcell, iteration, c1_set, c2_set)
                if minobj is None:
                    return
                self.minobj = minobj

                if not self.write_refine_structure(self.opt_poscar):
                    return

                if self.check_convergence(energy_keep):
                    print("Geometry optimization succeeded")
                    break

                energy_keep = self.minobj.energy / len(unitcell.elements)
                unitcell = Poscar(self.opt_poscar).structure

                if iteration == max_iteration - 1:
                    print("Maximum number of relaxation iterations has been exceeded")

            if not self._stop_rss:
                self.print_final_structure_details()
                judge = self.analyze_space_group(self.opt_poscar)
                if judge is False:
                    return
                self.log_computation_time()
                with open("rss_result/success.dat", "a") as f:
                    print(self.poscar_name, file=f)

    def minimize(self, unitcell, iteration, c1_set, c2_set):
        """Run geometry optimization with different parameters until successful."""
        minobj = GeometryOptimization(
            unitcell,
            pot=self.pot,
            relax_cell=True,
            relax_volume=True,
            relax_positions=True,
            with_sym=False,
            pressure=self.pressure,
            verbose=True,
        )
        if iteration == 0:
            print("Initial structure")
            minobj.print_structure()

        maxiter = 1000
        for c_count in range(3):
            if iteration == 0 and c_count <= 1 or iteration == 1 and c_count == 0:
                maxiter = self.maxiter
                continue

            try:
                minobj.run(
                    gtol=1e-6,
                    method=self.solver_method,
                    maxiter=maxiter,
                    c1=c1_set[c_count],
                    c2=c2_set[c_count],
                )
                energy_per_atom = minobj._energy / len(minobj.structure.elements)
                if energy_per_atom < -50:
                    print("Final function value (eV/atom):", energy_per_atom)
                    print(
                        "Geometry optimization failed: Huge negative or zero energy value."
                    )
                    self.log_computation_time()
                    return None
                return minobj

            except ValueError:
                if c_count == 2:
                    print(
                        "Final function value (eV/atom):",
                        minobj._energy / len(minobj.structure.elements),
                    )
                    print(
                        "Geometry optimization failed: Huge negative or zero energy value."
                    )
                    self.log_computation_time()
                    return None
                print("Change [c1, c2] to", c1_set[c_count + 1], c2_set[c_count + 1])
                maxiter = 100

    def write_refine_structure(self, poscar_path):
        """Refine the crystal structure with increasing symmetry precision."""
        self.minobj.write_poscar(filename=self.opt_poscar)
        if not wait_for_file_lines(self.opt_poscar):
            print("Reading file failed.")
            self.log_computation_time()
            return False

        symprec_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        refine_success = False
        for sp in symprec_list:
            try:
                sym = SymCell(poscar_name=poscar_path, symprec=sp)
                self.minobj.structure = sym.refine_cell()
                refine_success = True
                break
            except TypeError:
                print("TypeError: Change symprec to", sp * 10)
            except IndexError:
                print("IndexError: Change symprec to", sp * 10)
        if not refine_success:
            # Failed structure refinement for all tested symmetry tolerances
            print("Refining cell failed.")
            self.log_computation_time()
            return False
        else:
            self.minobj.write_poscar(filename=self.opt_poscar)
            if not wait_for_file_lines(self.opt_poscar):
                print("Reading file failed.")
                self.log_computation_time()
                return False
            return True

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
            except IndexError:
                continue
        if spg_sets == []:
            print("Analyzing space group failed.")
            self.log_computation_time()
            return False
        print("Space group set:")
        print(spg_sets)
        return True

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

    def log_computation_time(self):
        """Log computational time."""
        time_fin = time.time() - self.time_initial
        print("Computational time:", time_fin)
        print("Finished")
        with open("rss_result/finish.dat", "a") as f:
            print(self.poscar_name, file=f)

    def check_opt_str(self):
        if self.not_stop_rss:
            return
        with open("rss_result/success.dat") as f:
            success_str = sum(1 for _ in f)
        residual_str = self.num_opt_str - success_str
        if residual_str < 0:
            self._stop_rss = True
        if len(glob.glob("initial_struct/*")) == len(glob.glob("log/*")):
            self._stop_rss = True


def wait_for_file_lines(file_path, timeout=20, interval=0.1):
    """
    Waits until its contents (lines) can be read.

    Parameters:
        file_path (str): The path of the file to be read.
        timeout (int, optional): The maximum number of seconds to wait before timing out.
        interval (float, optional): The interval in seconds between checks.
    """
    start_time = time.time()

    while True:
        try:
            with open(file_path, "r") as f:
                lines = f.readlines()
            # Return the lines if the list is not empty
            if lines:
                return True
        except Exception:
            # If there's an error (such as the file being in use),
            # ignore and retry after the interval
            pass

        # Check if the timeout has been exceeded
        if time.time() - start_time > timeout:
            return False

        # Wait for the specified interval before retrying
        time.sleep(interval)

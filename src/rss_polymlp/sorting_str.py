import glob
import os
import re
from collections import Counter, defaultdict
from time import time

import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from rss_polymlp.initial_str import nearest_neighbor_atomic_distance
from rss_polymlp.readfile import ReadFile


class SortStructure:

    def __init__(self):
        """Initialize data structures for sorting structures."""
        self.nondup_str = []  # List to store non-duplicate structures
        self.iter_str = []  # Iteration statistics
        self.fval_str = []  # Function evaluation statistics
        self.gval_str = []  # Gradient evaluation statistics
        self.errors = Counter()  # Error tracking
        self.error_poscar = defaultdict(list)  # POSCAR error details
        self.time_all = 0  # Total computation time accumulator

    def read_and_process(self):
        """Read and process log files, filtering based on convergence criteria."""
        self.logfiles = sorted(
            glob.glob("log/*"), key=lambda x: int(x.split("_")[-1].removesuffix(".log"))
        )

        for logfile in self.logfiles:
            _res, judge = ReadFile(logfile).read_file()

            # Handle different error cases
            if judge in {"iteration", "energy_low", "energy_zero"}:
                self.errors[judge] += 1
                self.error_poscar[judge].append(_res["poscar"])
                continue

            if judge is not True or any(
                _res[key] is None for key in ["time", "spg", "energy", "res_f", "res_s"]
            ):
                self.errors["else_err"] += 1
                self.error_poscar["else_err"].append(_res["poscar"])
                continue

            # Convergence checks
            if _res["res_f"] > 10**-4:
                self.errors["f_conv"] += 1
                self.error_poscar["not_converged_f"].append(_res["poscar"])
                continue
            if _res["res_s"] > 10**-3:
                self.errors["s_conv"] += 1
                self.error_poscar["not_converged_s"].append(_res["poscar"])
                continue

            # Ensure the optimized structure file exists
            optimized_path = f"optimized_str/{_res['poscar']}"
            if not os.path.isfile(optimized_path):
                self.errors["else_err"] += 1
                self.error_poscar["else_err"].append(_res["poscar"])
                continue

            self.process_optimized_structure(_res, optimized_path)
            self.update_nondup_structure(_res)
            self.time_all += _res["time"]

    def process_optimized_structure(self, _res, optimized_path):
        """Extract structural properties from the optimized structure file."""
        polymlp_st = Poscar(optimized_path).structure
        _res["elements"] = polymlp_st.elements
        _res["volume"] = np.linalg.det(polymlp_st.axis) / len(_res["elements"])
        a, b, c = np.array(polymlp_st.axis)
        _res["axis"] = self.axis_to_abc(a, b, c)
        _res["positions"] = polymlp_st.positions.T.tolist()
        _res["distance"] = nearest_neighbor_atomic_distance(
            polymlp_st.axis, polymlp_st.positions
        )

    def angle_between(self, v1, v2):
        """Calculate the angle (in degrees) between two vectors."""
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))

    def axis_to_abc(self, a, b, c):
        """Convert lattice vectors to unit cell parameters."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        norm_c = np.linalg.norm(c)
        alpha = self.angle_between(a, b)
        beta = self.angle_between(b, c)
        gamma = self.angle_between(c, a)
        return np.array([norm_a, norm_b, norm_c, alpha, beta, gamma]).tolist()

    def update_nondup_structure(self, _res):
        """
        Check for duplicate structures and update the list accordingly.

        A structure is considered a duplicate if:
        1. Its energy is within 1e-8 of an existing structure.
        2. It shares at least one space group with an existing structure.

        If a duplicate is found:
        - The duplicate count is incremented.
        - The structure is updated if it has a higher symmmerty of space group.
        Otherwise, the structure is added as a new unique entry.
        """

        app_to_nonduplicate = True
        change_str = False
        for idx, entry in enumerate(self.nondup_str):
            if abs(entry["energy"] - _res["energy"]) < 1e-8 and any(
                spg in _res["spg"] for spg in entry["spg"]
            ):
                app_to_nonduplicate = False
                if self.extract_spg_count(_res["spg"]) > self.extract_spg_count(
                    entry["spg"]
                ):
                    change_str = True
                break

        self.update_iteration_stats(_res)

        if not app_to_nonduplicate:
            # Update duplicate count and replace with better data if necessary
            self.nondup_str[idx]["dup_count"] += 1
            if change_str:
                for key in [
                    "energy",
                    "spg",
                    "axis",
                    "positions",
                    "elements",
                    "volume",
                    "distance",
                ]:
                    self.nondup_str[idx][key] = _res[key]
        else:
            # Append a new unique structure
            self.add_new_structure(_res)

    def extract_spg_count(self, spg_list):
        """Extract and sum space group counts from a list of space group strings."""
        return sum(
            int(re.search(r"\((\d+)\)", s).group(1))
            for s in spg_list
            if re.search(r"\((\d+)\)", s)
        )

    def update_iteration_stats(self, _res):
        """Update iteration statistics."""
        if not self.iter_str:
            self.iter_str.append(_res["iter"])
            self.fval_str.append(_res["fval"])
            self.gval_str.append(_res["gval"])
        else:
            self.iter_str[-1] += _res["iter"]
            self.fval_str[-1] += _res["fval"]
            self.gval_str[-1] += _res["gval"]

    def add_new_structure(self, _res):
        """Add a new structure to the list of non-duplicate structures."""
        self.nondup_str.append(_res)
        self.iter_str.append(self.iter_str[-1])
        self.fval_str.append(self.fval_str[-1])
        self.gval_str.append(self.gval_str[-1])
        _res["iter"] = self.iter_str[-1]
        _res["fval"] = self.fval_str[-1]
        _res["gval"] = self.gval_str[-1]

    def run_sorting(self):
        """Sort structures and write the results to a log file."""
        time_start = time()
        self.read_and_process()
        time_finish = time() - time_start

        # Sort structures by energy
        self.nondup_str.sort(key=lambda x: x["energy"])

        # Get minimum energy value
        energy_min = -10
        for _str in self.nondup_str:
            if energy_min < _str["energy"]:
                energy_min = _str["energy"]
                break

        # Calculate total error count
        error_count = sum(
            [
                self.errors["energy_low"],
                self.errors["energy_zero"],
                self.errors["f_conv"],
                self.errors["s_conv"],
                self.errors["iteration"],
                self.errors["else_err"],
            ]
        )

        # Write results to log file
        with open("sorted_result.log", "w") as f:
            print("---- General outputs ----", file=f)
            print("Sorting time (sec.):           ", round(time_finish, 3), file=f)
            print(
                "Selected potential:            ",
                self.nondup_str[0]["potential"],
                file=f,
            )
            print("Number of initial structures:  ", len(self.logfiles), file=f)
            print(
                "Number of optimized structures:",
                len(self.logfiles) - error_count,
                file=f,
            )
            print("Total computation time (sec.): ", round(self.time_all, 1), file=f)
            print("", file=f)
            print("---- Error counts ----", file=f)
            print("Total error counts:", error_count, file=f)
            print(" - Low energy:        ", self.errors["energy_low"], file=f)
            print(" - Zero energy:       ", self.errors["energy_zero"], file=f)
            print(" - Force convergence: ", self.errors["f_conv"], file=f)
            print(" - Stress convergence:", self.errors["s_conv"], file=f)
            print(" - Max iteration:     ", self.errors["iteration"], file=f)
            print(" - Other reason:      ", self.errors["else_err"], file=f)
            print("", file=f)
            print("---- Nonduplicate structures ----", file=f)
            for idx, _str in enumerate(self.nondup_str):
                e_diff = round((_str["energy"] - energy_min) * 1000, 2)
                print(f"No. {idx+1}", file=f)
                print(
                    f"{_str['poscar']} ({e_diff} meV/atom, {_str['dup_count']} duplicates)",
                    file=f,
                )
                print(" - Enthalpy:   ", _str["energy"], file=f)
                print(" - Axis:       ", _str["axis"], file=f)
                print(" - Postions:   ", _str["positions"], file=f)
                print(" - Elements:   ", _str["elements"], file=f)
                print(" - Space group:", _str["spg"], file=f)
                print(
                    " - Other Info.:",
                    f'{int(len(_str["elements"]))} atom',
                    f'/ distance {round(_str["distance"], 3)} (Ang.)',
                    f'/ volume {round(_str["volume"], 2)} (A^3/atom)',
                    f'/ iteration {_str["iter"]}',
                    file=f,
                )
            print("", file=f)
            print("---- Total evaluation counts ----", file=f)
            print("Iteration:           ", self.iter_str[-1], file=f)
            print("Function evaluations:", self.fval_str[-1], file=f)
            print("Gradient evaluations:", self.gval_str[-1], file=f)
            print("", file=f)
            print("---- Evaluation count per structure ----", file=f)
            print("Iteration (list):           ", self.iter_str, file=f)
            print("Function evaluations (list):", self.fval_str, file=f)
            print("Gradient evaluations (list):", self.gval_str, file=f)
            print("", file=f)
            print("---- Poscar name (failed) ----", file=f)
            print("Low energy:        ", self.error_poscar["energy_low"], file=f)
            print("Zero energy:       ", self.error_poscar["energy_zero"], file=f)
            print("Force convergence: ", self.error_poscar["not_converged_f"], file=f)
            print("Stress convergence:", self.error_poscar["not_converged_s"], file=f)
            print("Max iteration:     ", self.error_poscar["iteration"], file=f)
            print("Other reason:      ", self.error_poscar["else_err"], file=f)

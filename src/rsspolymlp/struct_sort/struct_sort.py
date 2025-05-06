"""
Parse and analyze optimization logs, filter out failed or unconverged results,
identify and retain unique structures based on energy and symmetry,
and write detailed computational statistics to log.
"""

import glob
import os
import re
from collections import Counter, defaultdict
from time import time

import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.io_polymlp import load_mlps
from rsspolymlp.parse_arg import ParseArgument
from rsspolymlp.struct_matcher.struct_matcher import (
    get_irrep_positions,
    get_recommend_symprecs,
    get_distance_cluster,
    struct_match,
)
from rsspolymlp.struct_sort.readfile import ReadFile
from rsspolymlp.utils.property_util import PropUtil


def run():
    args = ParseArgument.get_sorting_args()
    sorter = SortStructure()
    if args.cutoff is not None:
        sorter.cutoff = args.cutoff
    sorter.run_sorting(args)


class SortStructure:

    def __init__(self):
        """Initialize data structures for sorting structures."""
        self.cutoff = None
        self.nondup_str = []  # List to store non-duplicate structures
        self.iter_str = []  # Iteration statistics
        self.fval_str = []  # Function evaluation statistics
        self.gval_str = []  # Gradient evaluation statistics
        self.errors = Counter()  # Error tracking
        self.error_poscar = defaultdict(list)  # POSCAR error details
        self.time_all = 0  # Total computation time accumulator

    def get_result_from_logfiles(self):
        """Read and process log files, filtering based on convergence criteria."""
        for logfile in self.logfiles:
            try:
                _res, judge = ReadFile(logfile).read_file()
            except TypeError:
                self.errors["else_err"] += 1
                self.error_poscar["else_err"].append(
                    logfile.split("/")[-1].removesuffix(".log")
                )
                continue

            # Handle different error cases
            if judge in {"iteration", "energy_low", "energy_zero", "anom_struct"}:
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
            optimized_poscar = f"opt_struct/{_res['poscar']}"
            if not os.path.isfile(optimized_poscar):
                self.errors["else_err"] += 1
                self.error_poscar["else_err"].append(_res["poscar"])
                continue

            self.time_all += _res["time"]
            res_processed = self._preprocess_optimized_struct(_res, optimized_poscar)
            if res_processed is None:
                self.errors["invalid_layer_struct"] += 1
                self.error_poscar["invalid_layer_struct"].append(_res["poscar"])
                continue

            self._update_nondup_structure(res_processed)

    def _preprocess_optimized_struct(self, res, poscar_name):
        """Extract structural properties from the optimized structure file."""
        if self.cutoff is None:
            _params, _ = load_mlps(res["potential"])
            if not isinstance(_params, list):
                self.cutoff = _params.as_dict()["model"]["cutoff"]
            else:
                max_cutoff = 0.0
                for param in _params:
                    model_dict = param.as_dict()
                    cutoff_i = model_dict["model"]["cutoff"]
                    if cutoff_i > max_cutoff:
                        max_cutoff = cutoff_i
                self.cutoff = max_cutoff

        polymlp_st = Poscar(poscar_name).structure
        prop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
        res["elements"] = polymlp_st.elements
        res["volume"] = prop.volume
        res["axis"] = prop.axis_to_abc
        res["positions"] = polymlp_st.positions.T.tolist()
        res["distance"] = prop.least_distance

        distance_cluster = get_distance_cluster(
            struct=polymlp_st, symprec_irrep=1e-5
        )
        if distance_cluster is not None:
            max_layer_diff = max(
                [
                    np.max(distance_cluster[0]) * res["axis"][0],
                    np.max(distance_cluster[1]) * res["axis"][1],
                    np.max(distance_cluster[2]) * res["axis"][2],
                ]
            )
            if max_layer_diff > self.cutoff:
                return None

        struct, recommend_symprecs = get_recommend_symprecs(
            poscar_name=poscar_name, symprec_irrep=1e-5
        )
        symprec_list = [1e-5]
        symprec_list.extend(recommend_symprecs)
        irrep_st = get_irrep_positions(struct=struct, symprec_irreps=symprec_list)
        res["irrep_st"] = irrep_st
        return res

    def _update_nondup_structure(self, _res, energy_diff=1e-8):
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
            if abs(entry["energy"] - _res["energy"]) < energy_diff and any(
                spg in _res["spg"] for spg in entry["spg"]
            ):
                app_to_nonduplicate = False
                if self._extract_spg_count(_res["spg"]) > self._extract_spg_count(
                    entry["spg"]
                ):
                    change_str = True
                break
            elif struct_match(entry["irrep_st"], _res["irrep_st"]):
                app_to_nonduplicate = False
                if self._extract_spg_count(_res["spg"]) > self._extract_spg_count(
                    entry["spg"]
                ):
                    change_str = True
                break

        self._update_iteration_stats(_res)

        if not app_to_nonduplicate:
            # Update duplicate count and replace with better data if necessary
            self.nondup_str[idx]["dup_count"] += 1
            if change_str:
                for key in [
                    "poscar",
                    "energy",
                    "spg",
                    "axis",
                    "positions",
                    "elements",
                    "volume",
                    "distance",
                    "irrep_st",
                ]:
                    self.nondup_str[idx][key] = _res[key]
        else:
            # Append a new unique structure
            self._add_new_structure(_res)

    def _extract_spg_count(self, spg_list):
        """Extract and sum space group counts from a list of space group strings."""
        return sum(
            int(re.search(r"\((\d+)\)", s).group(1))
            for s in spg_list
            if re.search(r"\((\d+)\)", s)
        )

    def _update_iteration_stats(self, _res):
        """Update iteration statistics."""
        if "iter" not in _res:
            return
        if not self.iter_str:
            self.iter_str.append(_res["iter"])
            self.fval_str.append(_res["fval"])
            self.gval_str.append(_res["gval"])
        else:
            self.iter_str[-1] += _res["iter"]
            self.fval_str[-1] += _res["fval"]
            self.gval_str[-1] += _res["gval"]

    def _add_new_structure(self, _res):
        """Add a new structure to the list of non-duplicate structures."""
        self.nondup_str.append(_res)
        if "iter" not in _res:
            return
        self.iter_str.append(self.iter_str[-1])
        self.fval_str.append(self.fval_str[-1])
        self.gval_str.append(self.gval_str[-1])
        _res["iter"] = self.iter_str[-1]
        _res["fval"] = self.fval_str[-1]
        _res["gval"] = self.gval_str[-1]

    def run_sorting(self, args):
        """Sort structures and write the results to a log file."""
        time_start = time()
        with open("finish.log") as f:
            finished_set = [line.strip() for line in f]
        with open("success.log") as f:
            sucessed_set = [line.strip() for line in f]
        if args.num_sort_str is not None:
            sucessed_set = sucessed_set[: args.num_sort_str]
            fin_poscar = sucessed_set[-1]
            index = finished_set.index(fin_poscar)
            finished_set = finished_set[: index + 1]
        self.logfiles = [f"log/{p}.log" for p in finished_set]
        self.get_result_from_logfiles()
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
                self.errors["anom_struct"],
                self.errors["f_conv"],
                self.errors["s_conv"],
                self.errors["iteration"],
                self.errors["else_err"],
            ]
        )

        # Check if optimization is complete
        max_init_str = int(len(glob.glob("initial_struct/*")))
        log_str = int(len(glob.glob("log/*")))
        finish_count = len(finished_set)
        success_count = len(sucessed_set)
        if log_str == max_init_str:
            stop_mes = "All randomly generated initial structures have been processed. Stopping."
        else:
            stop_mes = "Target number of optimized structures reached."
        prop_success = round(success_count / finish_count, 2)

        # Write results to log file
        if args.num_sort_str is not None:
            file_name = f"rss_results_{args.num_sort_str}.log"
        else:
            file_name = "rss_results.log"
        with open(file_name, "w") as f:
            print("---- General outputs ----", file=f)
            print("Sorting time (sec.):            ", round(time_finish, 2), file=f)
            print(
                "Selected potential:             ",
                self.nondup_str[0]["potential"],
                file=f,
            )
            print("Max number of structures in RSS:", max_init_str, file=f)
            print("Number of initial structures:   ", finish_count, file=f)
            print("Number of optimized structures: ", success_count, file=f)
            print("Stopping criterion:             ", stop_mes, file=f)
            print("Optimized str. / Initial str.:  ", prop_success, file=f)
            print(
                "Total computational time (sec.):",
                int(self.time_all),
                file=f,
            )
            print("", file=f)
            print("---- Total evaluation counts ----", file=f)
            print("Iteration:           ", self.iter_str[-1], file=f)
            print("Function evaluations:", self.fval_str[-1], file=f)
            print("Gradient evaluations:", self.gval_str[-1], file=f)
            print("", file=f)
            print("---- Error counts ----", file=f)
            print("Total error counts:", error_count, file=f)
            print(" - Low energy:        ", self.errors["energy_low"], file=f)
            print(" - Zero energy:       ", self.errors["energy_zero"], file=f)
            print(" - Anomalous struct.: ", self.errors["anom_struct"], file=f)
            print(" - Force convergence: ", self.errors["f_conv"], file=f)
            print(" - Stress convergence:", self.errors["s_conv"], file=f)
            print(" - Max iteration:     ", self.errors["iteration"], file=f)
            print(" - Other reason:      ", self.errors["else_err"], file=f)
            print("", file=f)
            print("---- Number of invalid layer structures ----", file=f)
            print("Total counts:", self.errors["invalid_layer_struct"], file=f)
            print(
                "Number of valid structures:",
                success_count - self.errors["invalid_layer_struct"],
                file=f,
            )
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
            print("---- Evaluation count per structure ----", file=f)
            print("Iteration (list):           ", self.iter_str, file=f)
            print("Function evaluations (list):", self.fval_str, file=f)
            print("Gradient evaluations (list):", self.gval_str, file=f)
            print("", file=f)
            print("---- POSCAR name (failed) ----", file=f)
            print("Low energy:        ", self.error_poscar["energy_low"], file=f)
            print("Zero energy:       ", self.error_poscar["energy_zero"], file=f)
            print("Anomalous struct.: ", self.error_poscar["anom_struct"], file=f)
            print("Force convergence: ", self.error_poscar["not_converged_f"], file=f)
            print("Stress convergence:", self.error_poscar["not_converged_s"], file=f)
            print("Max iteration:     ", self.error_poscar["iteration"], file=f)
            print("Other reason:      ", self.error_poscar["else_err"], file=f)
            print("---- POSCAR name (invalid layer structure) ----", file=f)
            print(
                "Layer structure:   ", self.error_poscar["invalid_layer_struct"], file=f
            )

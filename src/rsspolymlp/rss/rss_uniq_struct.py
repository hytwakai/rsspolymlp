"""
Parse optimization logs, filter out failed or unconverged cases,
identify and retain unique structures based on irreducible structure representation,
and write detailed computational statistics to the log.
"""

import glob
import os
from collections import Counter, defaultdict
from time import time

import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.io_polymlp import load_mlps
from rsspolymlp.analysis.outlier_cands import detect_outlier
from rsspolymlp.analysis.struct_matcher.utils import get_distance_cluster
from rsspolymlp.analysis.unique_struct import (
    UniqueStructureAnalyzer,
    generate_unique_structs,
)
from rsspolymlp.common.parse_arg import ParseArgument
from rsspolymlp.common.property import PropUtil
from rsspolymlp.rss.load_logfile import LogfileLoader


def run():
    args = ParseArgument.get_analysis_args()
    analyzer = RSSResultAnalyzer()
    if args.cutoff is not None:
        analyzer.cutoff = args.cutoff
    analyzer.run_rss_uniq_struct(args)


def log_unique_structures(file_name, unique_structs, unique_struct_iters=None):
    # Sort structures by energy
    energies = np.array([s.energy for s in unique_structs])
    sort_indices = np.argsort(energies)
    unique_str = [unique_structs[i] for i in sort_indices]
    if unique_struct_iters is not None:
        _iters = [unique_struct_iters[i] for i in sort_indices]

    # Get minimum energy value
    is_strong_outlier, is_weak_outlier = detect_outlier(energies[sort_indices])
    for i in range(len(unique_str)):
        if not is_weak_outlier[i]:
            energy_min = unique_str[i].energy
            break

    with open(file_name, "a") as f:
        print("---- Unique structures ----", file=f)
        for idx, _str in enumerate(unique_str):
            e_diff = round((_str.energy - energy_min) * 1000, 2)
            print(f"No. {idx+1}", file=f)
            print(
                f"{_str.input_poscar} ({e_diff} meV/atom, {_str.dup_count} duplicates)",
                file=f,
            )
            print(" - Enthalpy:   ", _str.energy, file=f)
            print(" - Axis:       ", _str.axis_abc, file=f)
            print(
                " - Postions:   ", _str.original_structure.positions.T.tolist(), file=f
            )
            print(" - Elements:   ", _str.original_structure.elements, file=f)
            print(" - Space group:", _str.spg_list, file=f)
            info = [
                " - Other Info.:",
                f"{_str.n_atoms} atom",
                f"/ distance {round(_str.least_distance, 3)} (Ang.)",
                f"/ volume {round(_str.volume, 2)} (A^3/atom)",
            ]
            if unique_struct_iters is not None:
                info.append(f"/ iteration {_iters[idx]}")
            print(*info, file=f)
            if is_strong_outlier[idx]:
                print(
                    " - WARNING    : This structure is marked as a strong outlier.",
                    file=f,
                )
            elif is_weak_outlier[idx]:
                print(
                    " - NOTE       : This structure is marked as a weak outlier.",
                    file=f,
                )


class RSSResultAnalyzer:

    def __init__(self):
        """Initialize data structures for sorting structures."""
        self.cutoff = None
        self.potential = None
        self.pressure = None
        self.iter_str = []  # Iteration statistics
        self.fval_str = []  # Function evaluation statistics
        self.gval_str = []  # Gradient evaluation statistics
        self.errors = Counter()  # Error tracking
        self.error_poscar = defaultdict(list)  # POSCAR error details
        self.time_all = 0  # Total computation time accumulator

    def _load_rss_logfiles(self):
        """Read and process log files, filtering based on convergence criteria."""
        struct_properties = []

        for logfile in self.logfiles:
            struct_prop, poscar_name = self._read_and_validate_logfile(logfile)
            if struct_prop is None:
                continue

            # Convergence checks
            if struct_prop["res_f"] > 10**-4:
                self.errors["f_conv"] += 1
                self.error_poscar["not_converged_f"].append(poscar_name)
                continue
            if struct_prop["res_s"] > 10**-3:
                self.errors["s_conv"] += 1
                self.error_poscar["not_converged_s"].append(poscar_name)
                continue

            # Ensure the optimized structure file exists
            optimized_poscar = f"opt_struct/{poscar_name}"
            if (
                not os.path.isfile(optimized_poscar)
                or os.path.getsize(optimized_poscar) == 0
            ):
                self.errors["else_err"] += 1
                self.error_poscar["else_err"].append(poscar_name)
                continue

            struct_prop = self._validate_optimized_struct(optimized_poscar, struct_prop)
            if struct_prop is None:
                self.errors["invalid_layer_struct"] += 1
                self.error_poscar["invalid_layer_struct"].append(poscar_name)
                continue

            struct_properties.append(struct_prop)

        return struct_properties

    def _read_and_validate_logfile(self, logfile):
        try:
            struct_prop, judge = LogfileLoader(logfile).read_file()
        except (TypeError, ValueError):
            self.errors["else_err"] += 1
            self.error_poscar["else_err"].append(
                logfile.split("/")[-1].removesuffix(".log")
            )
            return None, None

        poscar_name = struct_prop["poscar"]

        if judge in {"iteration", "energy_low", "energy_zero", "anom_struct"}:
            self.errors[judge] += 1
            self.error_poscar[judge].append(poscar_name)
            return None, None

        required_keys = [
            "potential",
            "time",
            "spg_list",
            "energy",
            "res_f",
            "res_s",
            "struct",
        ]
        if judge is not True or any(struct_prop.get(k) is None for k in required_keys):
            self.errors["else_err"] += 1
            self.error_poscar["else_err"].append(poscar_name)
            return None, None

        self.time_all += struct_prop["time"]
        self.potential = struct_prop["potential"]
        if struct_prop["pressure"] is not None:
            self.pressure = struct_prop["pressure"]

        return struct_prop, poscar_name

    def _validate_optimized_struct(self, poscar_name, struct_prop):
        if self.cutoff is None:
            _params, _ = load_mlps(self.potential)
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
        objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
        axis_abc = objprop.axis_to_abc
        _struct_prop = struct_prop
        _struct_prop["structure"] = polymlp_st

        distance_cluster = get_distance_cluster(polymlp_st=polymlp_st)
        if distance_cluster is not None:
            max_layer_diff = max(
                [
                    np.max(distance_cluster[0]) * axis_abc[0],
                    np.max(distance_cluster[1]) * axis_abc[1],
                    np.max(distance_cluster[2]) * axis_abc[2],
                ]
            )
            if max_layer_diff > self.cutoff:
                return None

        return _struct_prop

    def _analysis_unique_structure(
        self,
        struct_properties: list[dict],
        use_joblib: bool = True,
        num_process: int = -1,
        backend: str = "locky",
    ):
        analyzer = UniqueStructureAnalyzer()
        unique_struct = generate_unique_structs(
            struct_properties,
            use_joblib=use_joblib,
            num_process=num_process,
            backend=backend,
        )
        for idx, unique_struct in enumerate(unique_struct):
            is_unique, _ = analyzer.identify_duplicate_struct(
                unique_struct,
                other_properties=struct_properties[idx],
            )
            self._update_iteration_stats(struct_properties[idx], is_unique)

        return analyzer.unique_str, analyzer.unique_str_prop

    def _update_iteration_stats(self, _res, is_unique):
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
        if is_unique:
            self.iter_str.append(self.iter_str[-1])
            self.fval_str.append(self.fval_str[-1])
            self.gval_str.append(self.gval_str[-1])
            _res["iter"] = self.iter_str[-1]
            _res["fval"] = self.fval_str[-1]
            _res["gval"] = self.gval_str[-1]

    def run_rss_uniq_struct(self, args):
        """Sort structures and write the results to a log file."""
        if args.pressure is not None:
            self.pressure = args.pressure

        time_start = time()

        with open("finish.log") as f:
            finished_set = [line.strip() for line in f]
        with open("success.log") as f:
            sucessed_set = [line.strip() for line in f]
        if not args.num_str == -1:
            sucessed_set = sucessed_set[: args.num_str]
            fin_poscar = sucessed_set[-1]
            index = finished_set.index(fin_poscar)
            finished_set = finished_set[: index + 1]
        self.logfiles = [f"log/{p}.log" for p in finished_set]

        struct_properties = self._load_rss_logfiles()

        unique_str, unique_str_prop = self._analysis_unique_structure(
            struct_properties, args.use_joblib, args.num_process, args.backend
        )

        time_finish = time() - time_start

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
        file_name = "rss_results.log"
        with open(file_name, "w") as f:
            print("---- General informantion ----", file=f)
            print("Sorting time (sec.):            ", round(time_finish, 2), file=f)
            print(
                "Selected potential:             ",
                self.potential,
                file=f,
            )
            print("Pressure (GPa):                 ", self.pressure, file=f)
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

        _iters = np.array([s["iter"] for s in unique_str_prop])
        log_unique_structures(file_name, unique_str, _iters)

        with open(file_name, "a") as f:
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


if __name__ == "__main__":
    run()

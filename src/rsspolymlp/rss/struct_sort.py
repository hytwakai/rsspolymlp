"""
Parse and analyze optimization logs, filter out failed or unconverged results,
identify and retain unique structures based on energy and symmetry,
and write detailed computational statistics to log.
"""

import glob
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import time
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.io_polymlp import load_mlps
from rsspolymlp.analysis.struct_matcher.struct_match import (
    IrrepStruct,
    get_distance_cluster,
    get_irrep_positions,
    get_recommend_symprecs,
    struct_match,
)
from rsspolymlp.rss.read_logfile import ReadFile
from rsspolymlp.utils.parse_arg import ParseArgument
from rsspolymlp.utils.property import PropUtil


def run():
    args = ParseArgument.get_sorting_args()
    sorter = SortStructure()
    if args.cutoff is not None:
        sorter.cutoff = args.cutoff
    sorter.run_sorting(args)


@dataclass
class SortStruct:
    energy: float
    spg_list: list[str]
    irrep_struct: IrrepStruct
    original_axis: np.ndarray
    original_positions: np.ndarray
    original_elements: np.ndarray
    n_atoms: int
    volume: float
    least_distance: float
    input_poscar: str
    dup_count: int = 1


def generate_sort_struct(
    energy: float,
    spg_list: list[str],
    poscar_name: str,
    original_polymlp_st: Optional[PolymlpStructure] = None,
):
    if original_polymlp_st is None:
        original_polymlp_st = Poscar(poscar_name).structure
    _st = original_polymlp_st
    objprop = PropUtil(_st.axis.T, _st.positions.T)

    struct, recommend_symprecs = get_recommend_symprecs(
        poscar_name=poscar_name, symprec_irrep=1e-5
    )
    symprec_list = [1e-5]
    symprec_list.extend(recommend_symprecs)
    irrep_struct = get_irrep_positions(struct=struct, symprec_irreps=symprec_list)

    return SortStruct(
        energy=energy,
        spg_list=spg_list,
        irrep_struct=irrep_struct,
        original_axis=_st.axis.T,
        original_positions=_st.positions.T,
        original_elements=_st.elements,
        n_atoms=int(len(_st.elements)),
        volume=objprop.volume,
        least_distance=objprop.least_distance,
        input_poscar=poscar_name,
    )


class SortStructure:

    def __init__(self):
        """Initialize data structures for sorting structures."""
        self.cutoff = None
        self.nondup_str = []  # List to store non-duplicate structures
        self.nondup_str_prop = []  # List to store non-duplicate structure properties
        self.iter_str = []  # Iteration statistics
        self.fval_str = []  # Function evaluation statistics
        self.gval_str = []  # Gradient evaluation statistics
        self.errors = Counter()  # Error tracking
        self.error_poscar = defaultdict(list)  # POSCAR error details
        self.time_all = 0  # Total computation time accumulator

    def identify_duplicate_struct(
        self, sort_struct: SortStruct, other_properties: dict = {}, energy_diff=1e-8
    ):
        """
        Identify duplicate structures and update the internal list accordingly.

        A structure is considered a duplicate if:
        1. Its energy is within 1e-8 of an existing structure.
        2. It shares at least one space group with an existing structure.

        If a duplicate is found:
        - The duplicate count is incremented.
        - The structure is updated if it has a higher symmmerty of space group.
        Otherwise, the structure is added as a new unique entry.
        """

        is_nonduplicate = True
        change_struct = False
        _energy = sort_struct.energy
        _spg_list = sort_struct.spg_list
        _irrep_struct = sort_struct.irrep_struct

        for idx, ndstr in enumerate(self.nondup_str):
            if abs(ndstr.energy - _energy) < energy_diff and any(
                spg in _spg_list for spg in ndstr.spg_list
            ):
                is_nonduplicate = False
                if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                    ndstr.spg_list
                ):
                    change_struct = True
                break
            elif struct_match(ndstr.irrep_struct, _irrep_struct):
                is_nonduplicate = False
                if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                    ndstr.spg_list
                ):
                    change_struct = True
                break

        self._update_iteration_stats(other_properties, is_nonduplicate)

        if not is_nonduplicate:
            # Update duplicate count and replace with better data if necessary
            if change_struct:
                self.nondup_str[idx] = sort_struct
                self.nondup_str_prop[idx] = other_properties
            self.nondup_str[idx].dup_count += 1
        else:
            self.nondup_str.append(sort_struct)
            self.nondup_str_prop.append(other_properties)

    def _get_result_from_logfiles(self):
        """Read and process log files, filtering based on convergence criteria."""
        for logfile in self.logfiles:
            try:
                struct_properties, judge = ReadFile(logfile).read_file()
            except (TypeError, ValueError):
                self.errors["else_err"] += 1
                self.error_poscar["else_err"].append(
                    logfile.split("/")[-1].removesuffix(".log")
                )
                continue
            poscar_name = struct_properties["poscar"]

            # Handle different error cases
            if judge in {"iteration", "energy_low", "energy_zero", "anom_struct"}:
                self.errors[judge] += 1
                self.error_poscar[judge].append(poscar_name)
                continue

            if judge is not True or any(
                struct_properties[key] is None
                for key in ["time", "spg", "energy", "res_f", "res_s"]
            ):
                self.errors["else_err"] += 1
                self.error_poscar["else_err"].append(poscar_name)
                continue

            # Convergence checks
            if struct_properties["res_f"] > 10**-4:
                self.errors["f_conv"] += 1
                self.error_poscar["not_converged_f"].append(poscar_name)
                continue
            if struct_properties["res_s"] > 10**-3:
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

            self.time_all += struct_properties["time"]
            sort_struct, struct_properties = self._preprocess_optimized_struct(
                struct_properties, optimized_poscar
            )
            if sort_struct is None:
                self.errors["invalid_layer_struct"] += 1
                self.error_poscar["invalid_layer_struct"].append(
                    struct_properties["poscar"]
                )
                continue

            self.identify_duplicate_struct(sort_struct, struct_properties)

    def _preprocess_optimized_struct(self, struct_properties, poscar_name):
        """Extract structural properties from the optimized structure file."""
        _prop = struct_properties

        if self.cutoff is None:
            _params, _ = load_mlps(_prop["potential"])
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
        _prop["axis_abc"] = objprop.axis_to_abc
        _prop["position_list"] = polymlp_st.positions.T.tolist()

        distance_cluster = get_distance_cluster(struct=polymlp_st, symprec_irrep=1e-5)
        if distance_cluster is not None:
            max_layer_diff = max(
                [
                    np.max(distance_cluster[0]) * _prop["axis_abc"][0],
                    np.max(distance_cluster[1]) * _prop["axis_abc"][1],
                    np.max(distance_cluster[2]) * _prop["axis_abc"][2],
                ]
            )
            if max_layer_diff > self.cutoff:
                return None, _prop

        sort_struct = generate_sort_struct(
            _prop["energy"], _prop["spg"], poscar_name, original_polymlp_st=polymlp_st
        )
        return sort_struct, _prop

    def _extract_spg_count(self, spg_list):
        """Extract and sum space group counts from a list of space group strings."""
        return sum(
            int(re.search(r"\((\d+)\)", s).group(1))
            for s in spg_list
            if re.search(r"\((\d+)\)", s)
        )

    def _update_iteration_stats(self, _res, is_nonduplicate):
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
        if is_nonduplicate:
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
        self._get_result_from_logfiles()
        time_finish = time() - time_start

        # Sort structures by energy
        energies = np.array([s.energy for s in self.nondup_str])
        sort_indices = np.argsort(energies)
        self.nondup_str = [self.nondup_str[i] for i in sort_indices]
        self.nondup_str_prop = [self.nondup_str_prop[i] for i in sort_indices]

        # Get minimum energy value
        energy_min = -10
        for _str in self.nondup_str:
            if energy_min < _str.energy:
                energy_min = _str.energy
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
                self.nondup_str_prop[0]["potential"],
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
                _prop = self.nondup_str_prop[idx]
                e_diff = round((_str.energy - energy_min) * 1000, 2)
                print(f"No. {idx+1}", file=f)
                print(
                    f"{_str.input_poscar} ({e_diff} meV/atom, {_str.dup_count} duplicates)",
                    file=f,
                )
                print(" - Enthalpy:   ", _str.energy, file=f)
                print(" - Axis:       ", _prop["axis_abc"], file=f)
                print(" - Postions:   ", _prop["position_list"], file=f)
                print(" - Elements:   ", _str.original_elements, file=f)
                print(" - Space group:", _str.spg_list, file=f)
                print(
                    " - Other Info.:",
                    f"{_str.n_atoms} atom",
                    f"/ distance {round(_str.least_distance, 3)} (Ang.)",
                    f"/ volume {round(_str.volume, 2)} (A^3/atom)",
                    f'/ iteration {_prop["iter"]}',
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

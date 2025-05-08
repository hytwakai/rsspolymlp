import ast
import os
import re
from collections import defaultdict
from time import time

import joblib
import numpy as np

from rsspolymlp.rss.struct_sort import SortStructure, generate_sort_struct
from rsspolymlp.utils.parse_arg import ParseArgument
from rsspolymlp.utils.property import PropUtil


def run():
    args = ParseArgument.get_sorting_args2()
    sorter_all = SortStructAll(
        args.elements,
        args.result_paths,
        args.use_joblib,
        args.num_process,
        args.backend,
    )
    sorter_all.run_sorting()


def extract_composition_ratio(path_name: str, target_elements: list):
    comp_ratio = None
    with open(path_name) as f:
        for line in f:
            if "- Elements:" in line:
                line_strip = line.strip()
                log = re.search(r"- Elements: (.+)", line_strip)
                element_list = np.array(ast.literal_eval(log[1]))
                counts = np.array(
                    [np.count_nonzero(element_list == el) for el in target_elements]
                )
                if not np.any(counts):
                    raise ValueError(
                        "None of the specified elements were found in the result."
                    )
                g = np.gcd.reduce(counts[counts > 0])
                reduced = counts // g
                comp_ratio = tuple(reduced)
                break

    return comp_ratio


def load_rss_results(result_path: str, absolute_path=False) -> list[dict]:
    rss_results = []
    parent_path = os.path.dirname(result_path)
    with open(result_path) as f:
        lines = [i.strip() for i in f]
    for line_idx in range(len(lines)):
        if "No." in lines[line_idx]:
            _res = {}
            if absolute_path:
                _res["poscar"] = str(lines[line_idx + 1]).split()[0]
            else:
                poscar_name = str(lines[line_idx + 1]).split()[0].split("/")[-1]
                _res["poscar"] = parent_path + "/opt_struct/" + poscar_name
            _res["enthalpy"] = float(lines[line_idx + 2].split()[-1])
            spg = re.search(r"- Space group: (.+)", lines[line_idx + 6])
            _res["spg_list"] = ast.literal_eval(spg[1])
            _res["volume"] = float(lines[line_idx + 7].split("volume")[-1].split()[0])
            rss_results.append(_res)

    return rss_results


def generate_sort_structs(
    rss_results, use_joblib=True, num_process=-1, backend="loky"
) -> list[SortStructure]:
    if use_joblib:
        sort_structs = joblib.Parallel(n_jobs=num_process, backend=backend)(
            joblib.delayed(generate_sort_struct)(
                res["enthalpy"], res["spg_list"], res["poscar"]
            )
            for res in rss_results
        )
    else:
        sort_structs = []
        for res in rss_results:
            sort_structs.append(
                generate_sort_struct(res["enthalpy"], res["spg_list"], res["poscar"])
            )
    return sort_structs


def detect_outlier(energies: np.array, volumes: np.array):
    energy_diff = np.abs(np.diff(energies))
    volume_diff = np.abs(np.diff(volumes))

    maybe_outlier = np.full_like(energies, fill_value=False, dtype=bool)
    for i in range(len(energy_diff)):
        if energy_diff[i] > 0.5 or volume_diff[i] > 4:
            maybe_outlier[i] = True
        else:
            break

    return maybe_outlier


class SortStructAll:

    def __init__(
        self,
        elements,
        result_paths,
        use_joblib,
        num_process: int = -1,
        backend: str = "loky",
    ):
        self.elements = elements
        self.result_paths = result_paths
        self.use_joblib = use_joblib
        self.num_process = num_process
        self.backend = backend

    def run_sorting(self):
        result_path_comp = defaultdict(list)
        for path_name in self.result_paths:
            comp_ratio = extract_composition_ratio(path_name, self.elements)
            result_path_comp[comp_ratio].append(path_name)
        result_path_comp = dict(result_path_comp)

        for comp_ratio, res_paths in result_path_comp.items():
            log_name = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    log_name += f"{self.elements[i]}{comp_ratio[i]}"

            time_start = time()

            nondup_str, num_opt_struct, integrated_res_paths = (
                self._sorting_in_same_comp(comp_ratio, res_paths)
            )

            time_finish = time() - time_start

            energies = np.array([s.energy for s in nondup_str])
            volumes = np.array([s.volume for s in nondup_str])
            sort_indices = np.argsort(energies)
            nondup_str = [nondup_str[i] for i in sort_indices]
            
            maybe_outlier = detect_outlier(
                energies[sort_indices], volumes[sort_indices]
            )
            for i in range(len(nondup_str)):
                if not maybe_outlier[i]:
                    energy_min = nondup_str[i].energy
                    break

            with open(log_name + ".log", "w") as f:
                print("---- General outputs ----", file=f)
                print("Sorting time (sec.):           ", round(time_finish, 2), file=f)
                print("Number of optimized strcts:    ", num_opt_struct, file=f)
                print("Number of nonduplicate structs:", len(nondup_str), file=f)
                print(
                    "Input file names:              ",
                    integrated_res_paths,
                    file=f,
                )
                print("", file=f)
                print("---- Nonduplicate structures ----", file=f)
                for idx, _str in enumerate(nondup_str):
                    objprop = PropUtil(_str.original_axis.T, _str.original_positions.T)
                    e_diff = round((_str.energy - energy_min) * 1000, 2)
                    print(f"No. {idx+1}", file=f)
                    print(
                        f"{_str.input_poscar} ({e_diff} meV/atom, {_str.dup_count} duplicates)",
                        file=f,
                    )
                    print(" - Enthalpy:   ", _str.energy, file=f)
                    print(" - Axis:       ", objprop.axis_to_abc, file=f)
                    print(" - Postions:   ", _str.original_positions.T.tolist(), file=f)
                    print(" - Elements:   ", _str.original_elements, file=f)
                    print(" - Space group:", _str.spg_list, file=f)
                    print(
                        " - Other Info.:",
                        f"{_str.n_atoms} atom",
                        f"/ distance {round(_str.least_distance, 3)} (Ang.)",
                        f"/ volume {round(_str.volume, 2)} (A^3/atom)",
                        file=f,
                    )
                    if maybe_outlier[idx]:
                        print(" - WARNING    : This structure might be an outlier.", file=f)
                print("", file=f)

    def _sorting_in_same_comp(self, comp_ratio, result_paths):
        log_name = ""
        for i in range(len(comp_ratio)):
            if not comp_ratio[i] == 0:
                log_name += f"{self.elements[i]}{comp_ratio[i]}"

        sorter = SortStructure()
        num_opt_struct = 0
        pre_result_paths = []
        if os.path.isfile(log_name + ".log"):
            with open(log_name + ".log") as f:
                for line in f:
                    line_strip = line.strip()
                    if "Number of optimized strcts:" in line_strip:
                        num_opt_struct = int(line_strip.split()[-1])
                    if "Input file names:" in line_strip:
                        paths = re.search(r"Input file names:\s+(.+)", line_strip)
                        pre_result_paths = ast.literal_eval(paths[1])
                        break

            rss_results1 = load_rss_results(log_name + ".log", absolute_path=True)
            sort_structs1 = generate_sort_structs(
                rss_results1,
                use_joblib=self.use_joblib,
                num_process=self.num_process,
                backend=self.backend,
            )
            sorter.nondup_str = sort_structs1

        not_processed_path = list(set(result_paths) - set(pre_result_paths))
        integrated_res_paths = list(set(result_paths) | set(pre_result_paths))

        rss_results2 = []
        for res_path in not_processed_path:
            rss_res = load_rss_results(res_path)
            rss_results2.extend(rss_res)
        sort_structs2 = generate_sort_structs(
            rss_results2,
            use_joblib=self.use_joblib,
            num_process=self.num_process,
            backend=self.backend,
        )
        num_opt_struct += len(sort_structs2)

        for res in sort_structs2:
            sorter.identify_duplicate_struct(res)

        return sorter.nondup_str, num_opt_struct, integrated_res_paths

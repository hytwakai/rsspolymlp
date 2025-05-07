import ast
import os
import re
from collections import defaultdict
from time import time

import numpy as np

from rsspolymlp.struct_sorter.struct_sort import SortStructure
from rsspolymlp.utils.property import PropUtil


def get_results_from_file(path_name: str, target_elements: list):
    rss_results = []
    comp_ratio = None

    with open(path_name) as f:
        parent_path = os.path.dirname(path_name)
        lines = [i.strip() for i in f]
        for line_idx in range(len(lines)):
            if "No." in lines[line_idx]:
                _res = {}
                poscar_name = str(lines[line_idx + 1]).split()[0].split("/")[-1]
                _res["poscar"] = parent_path + "/opt_struct/" + poscar_name
                _res["enthalpy"] = float(lines[line_idx + 2].split()[-1])
                spg = re.search(r"- Space group: (.+)", lines[line_idx + 6])
                _res["spg_list"] = ast.literal_eval(spg.group(1))
                rss_results.append(_res)

                if comp_ratio is None:
                    log = re.search(r"- Elements: (.+)", lines[line_idx + 5])
                    element_list = np.array(ast.literal_eval(log.group(1)))
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

    return rss_results, comp_ratio


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        nargs="*",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--rss_results",
        nargs="*",
        type=str,
        default="rss_results",
        help="",
    )
    args = parser.parse_args()

    elements = args.elements
    rss_result_path = args.rss_results

    rss_result_all = defaultdict(list)
    input_path_all = defaultdict(list)
    for path_name in rss_result_path:
        rss_result_part, comp_ratio = get_results_from_file(path_name, elements)
        rss_result_all[comp_ratio].extend(rss_result_part)
        input_path_all[comp_ratio].append(path_name)
    rss_result_all = dict(rss_result_all)
    input_path_all = dict(input_path_all)

    for comp_ratio, rss_results in rss_result_all.items():
        time_start = time()
        sorter = SortStructure()
        for res in rss_results:
            sort_struct = sorter.get_sort_struct(
                res["enthalpy"], res["spg_list"], res["poscar"]
            )
            sorter.identify_duplicate_struct(sort_struct)
        nondup_str = sorter.nondup_str
        time_finish = time() - time_start

        energies = np.array([s.energy for s in nondup_str])
        sort_indices = np.argsort(energies)
        nondup_str = [nondup_str[i] for i in sort_indices]

        energy_min = -10
        for _str in nondup_str:
            if energy_min < _str.energy:
                energy_min = _str.energy
                break

        log_name = ""
        for i in range(len(comp_ratio)):
            if not comp_ratio[i] == 0:
                log_name += f"{elements[i]}{comp_ratio[i]}"
        with open(log_name + ".log", "w") as f:
            print("---- General outputs ----", file=f)
            print("Sorting time (sec.):              ", round(time_finish, 2), file=f)
            print("Number of optimized strctures:    ", len(rss_results), file=f)
            print("Number of nonduplicate structures:", len(nondup_str), file=f)
            print(
                "Input file names:                 ", input_path_all[comp_ratio], file=f
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
            print("", file=f)

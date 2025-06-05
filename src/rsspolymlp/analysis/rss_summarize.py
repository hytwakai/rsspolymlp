import argparse
import ast
import os
import re
from collections import defaultdict
from time import time

import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from rsspolymlp.analysis.unique_struct import (
    UniqueStructureAnalyzer,
    generate_unique_structs,
)
from rsspolymlp.common.comp_ratio import CompositionResult, compute_composition
from rsspolymlp.common.parse_arg import ParseArgument
from rsspolymlp.rss.rss_uniq_struct import log_unique_structures


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        nargs="*",
        type=str,
        default=None,
        help="List of target element symbols",
    )
    parser.add_argument(
        "--rss_paths",
        nargs="*",
        type=str,
        required=True,
        help="Path(s) to directories where RSS was performed",
    )
    ParseArgument.add_parallelization_arguments(parser)
    ParseArgument.add_analysis_arguments(parser)
    args = parser.parse_args()

    analyzer_all = RSSResultSummarizer(
        args.elements,
        args.rss_paths,
        args.use_joblib,
        args.num_process,
        args.backend,
        args.num_str,
    )
    analyzer_all.run_sorting()


def extract_composition_ratio(path_name: str, element_order: list) -> CompositionResult:
    with open(path_name) as f:
        for line in f:
            if "- Elements:" in line:
                line_strip = line.strip()
                log = re.search(r"- Elements: (.+)", line_strip)
                element_list = np.array(ast.literal_eval(log[1]))
                comp_res = compute_composition(element_list, element_order)
                break

    return comp_res


def load_rss_results(
    result_path: str, absolute_path=False, get_warning=False
) -> list[dict]:
    rss_results = []

    parent_path = os.path.dirname(result_path)
    with open(result_path) as f:
        lines = [i.strip() for i in f]
    pressure = None
    for line in lines:
        if "Pressure (GPa):" in line:
            pressure = float(line.split()[-1])
            break
    struct_no = 1
    for line_idx in range(len(lines)):
        if "No." in lines[line_idx]:
            _res = {}
            if absolute_path:
                _res["poscar"] = str(lines[line_idx + 1]).split()[0]
            else:
                poscar_name = str(lines[line_idx + 1]).split()[0].split("/")[-1]
                _res["poscar"] = parent_path + "/opt_struct/" + poscar_name
            _res["structure"] = Poscar(_res["poscar"]).structure
            _res["energy"] = float(lines[line_idx + 2].split()[-1])
            _res["pressure"] = pressure
            spg = re.search(r"- Space group: (.+)", lines[line_idx + 6])
            _res["spg_list"] = ast.literal_eval(spg[1])
            _res["volume"] = float(lines[line_idx + 7].split("volume")[-1].split()[0])
            if get_warning:
                warning_line = lines[line_idx + 8] if line_idx + 8 < len(lines) else ""
                _res["is_strong_outlier"] = "WARNING" in warning_line
                _res["is_weak_outlier"] = "NOTE" in warning_line
            _res["struct_no"] = struct_no
            rss_results.append(_res)
            struct_no += 1

    return rss_results, pressure


class RSSResultSummarizer:

    def __init__(
        self,
        elements,
        rss_paths,
        use_joblib,
        num_process: int = -1,
        backend: str = "loky",
        num_str: int = -1,
    ):
        self.elements = elements
        self.rss_paths = rss_paths
        self.use_joblib = use_joblib
        self.num_process = num_process
        self.backend = backend
        self.num_str = num_str

    def run_sorting(self):
        result_path_comp = defaultdict(list)
        for path_name in self.rss_paths:
            rss_result_path = f"{path_name}/rss_results.log"
            comp_res = extract_composition_ratio(rss_result_path, self.elements)
            comp_ratio = comp_res.comp_ratio
            result_path_comp[comp_ratio].append(rss_result_path)
        result_path_comp = dict(result_path_comp)

        for comp_ratio, res_paths in result_path_comp.items():
            log_name = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    log_name += f"{self.elements[i]}{comp_ratio[i]}"

            time_start = time()

            unique_str, num_opt_struct, integrated_res_paths, pressure = (
                self._sorting_in_same_comp(comp_ratio, res_paths)
            )

            time_finish = time() - time_start

            with open(log_name + ".log", "w") as f:
                print("---- General informantion ----", file=f)
                print("Sorting time (sec.):       ", round(time_finish, 2), file=f)
                print("Pressure (GPa):            ", pressure, file=f)
                print("Number of optimized strcts:", num_opt_struct, file=f)
                print("Number of unique structs:  ", len(unique_str), file=f)
                print(
                    "Input file names:          ",
                    sorted(integrated_res_paths),
                    file=f,
                )
                print("", file=f)
            log_unique_structures(log_name + ".log", unique_str)
            print(log_name, "finished", flush=True)

    def _sorting_in_same_comp(self, comp_ratio, result_paths):
        log_name = ""
        for i in range(len(comp_ratio)):
            if not comp_ratio[i] == 0:
                log_name += f"{self.elements[i]}{comp_ratio[i]}"

        analyzer = UniqueStructureAnalyzer()
        num_opt_struct = 0
        pressure = None
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

            rss_results1, pressure = load_rss_results(
                log_name + ".log", absolute_path=True
            )
            unique_structs1 = generate_unique_structs(
                rss_results1,
                use_joblib=self.use_joblib,
                num_process=self.num_process,
                backend=self.backend,
            )
            analyzer._initialize_unique_structs(unique_structs1)

        not_processed_path = list(set(result_paths) - set(pre_result_paths))
        integrated_res_paths = list(set(result_paths) | set(pre_result_paths))

        rss_results2 = []
        for res_path in not_processed_path:
            rss_res, pressure = load_rss_results(res_path)
            rss_results2.extend(rss_res)
        unique_structs2 = generate_unique_structs(
            rss_results2,
            use_joblib=self.use_joblib,
            num_process=self.num_process,
            backend=self.backend,
        )
        num_opt_struct += len(unique_structs2)

        for res in unique_structs2:
            analyzer.identify_duplicate_struct(res)

        return analyzer.unique_str, num_opt_struct, integrated_res_paths, pressure


if __name__ == "__main__":
    run()

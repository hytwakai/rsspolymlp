import argparse
import json
import os
from collections import defaultdict
from time import time

import yaml

from rsspolymlp.analysis.unique_struct import (
    UniqueStructureAnalyzer,
    generate_unique_structs,
)
from rsspolymlp.common.parse_arg import ParseArgument
from rsspolymlp.rss.rss_uniq_struct import log_unique_structures
from rsspolymlp.utils.convert_dict import polymlp_struct_from_dict


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
    )
    analyzer_all.run_sorting()


class RSSResultSummarizer:

    def __init__(
        self,
        elements,
        rss_paths,
        use_joblib,
        num_process: int = -1,
        backend: str = "loky",
    ):
        self.elements = elements
        self.rss_paths = rss_paths
        self.use_joblib = use_joblib
        self.num_process = num_process
        self.backend = backend

    def run_sorting(self):
        os.makedirs("json", exist_ok=True)

        paths_same_comp = defaultdict(list)
        results_same_comp = defaultdict(dict)
        for path_name in self.rss_paths:
            rss_result_path = f"{path_name}/rss_result/rss_results.json"
            with open(rss_result_path) as f:
                loaded_dict = json.load(f)

            rel_path = os.path.relpath(f"{path_name}/opt_struct", start=os.getcwd())
            for i in range(len(loaded_dict["rss_results"])):
                poscar_name = loaded_dict["rss_results"][i]["poscar"]
                loaded_dict["rss_results"][i]["poscar"] = f"{rel_path}/{poscar_name}"

            target_elements = loaded_dict["elements"]
            comp_ratio = tuple(loaded_dict["comp_ratio"])
            _dicts = dict(zip(target_elements, comp_ratio))
            comp_ratio_orderd = tuple(_dicts.get(el, 0) for el in self.elements)

            paths_same_comp[comp_ratio_orderd].append(rss_result_path)
            results_same_comp[comp_ratio_orderd][rss_result_path] = loaded_dict

        paths_same_comp = dict(paths_same_comp)
        for comp_ratio, res_paths in paths_same_comp.items():
            log_name = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    log_name += f"{self.elements[i]}{comp_ratio[i]}"

            time_start = time()
            unique_str, num_opt_struct, integrated_res_paths, pressure = (
                self._sorting_in_same_comp(
                    comp_ratio, res_paths, results_same_comp[comp_ratio]
                )
            )
            time_finish = time() - time_start

            with open(log_name + ".yaml", "w") as f:
                print("general_information:", file=f)
                print(f"  sorting_time_sec:      {round(time_finish, 2)}", file=f)
                print(f"  pressure_GPa:          {pressure}", file=f)
                print(f"  num_optimized_structs: {num_opt_struct}", file=f)
                print(f"  num_unique_structs:    {len(unique_str)}", file=f)
                print(f"  input_file_names:      {sorted(integrated_res_paths)}", file=f)
                print("", file=f)

            rss_result_all = log_unique_structures(
                log_name + ".yaml", unique_str, pressure=pressure, detect_outliers=True
            )

            with open(f"json/{log_name}.json", "w") as f:
                json.dump(rss_result_all, f)

            print(log_name, "finished", flush=True)

    def _sorting_in_same_comp(self, comp_ratio, result_paths, rss_result_dict):
        log_name = ""
        for i in range(len(comp_ratio)):
            if not comp_ratio[i] == 0:
                log_name += f"{self.elements[i]}{comp_ratio[i]}"

        analyzer = UniqueStructureAnalyzer()
        num_opt_struct = 0
        pressure = None
        pre_result_paths = []
        if os.path.isfile(log_name + ".yaml"):
            with open(log_name + ".yaml") as f:
                yaml_data = yaml.safe_load(f)
                num_opt_struct = yaml_data["general_information"][
                    "num_optimized_structs"
                ]
                pre_result_paths = yaml_data["general_information"]["input_file_names"]

            with open(f"./json/{log_name}.json") as f:
                loaded_dict = json.load(f)
            rss_results1 = loaded_dict["rss_results"]
            for i in range(len(rss_results1)):
                rss_results1[i]["structure"] = polymlp_struct_from_dict(
                    rss_results1[i]["structure"]
                )
            pressure = loaded_dict["pressure"]

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
            loaded_dict = rss_result_dict[res_path]
            rss_res = loaded_dict["rss_results"]
            for i in range(len(rss_res)):
                rss_res[i]["structure"] = polymlp_struct_from_dict(
                    rss_res[i]["structure"]
                )
            pressure = loaded_dict["pressure"]
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

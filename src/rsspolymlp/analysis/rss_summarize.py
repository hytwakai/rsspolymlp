import json
import os
import shutil
from collections import defaultdict
from time import time

import numpy as np
import yaml

from rsspolymlp.analysis.ghost_minima import detect_ghost_minima
from rsspolymlp.analysis.unique_struct import (
    UniqueStructureAnalyzer,
    generate_unique_structs,
)
from rsspolymlp.common.convert_dict import polymlp_struct_from_dict
from rsspolymlp.rss.eliminate_duplicates import log_unique_structures


class RSSResultSummarizer:

    def __init__(
        self,
        elements,
        rss_paths,
        use_joblib,
        num_process: int = -1,
        backend: str = "loky",
        output_poscar: bool = False,
        threshold: float = None,
    ):
        self.elements = elements
        self.rss_paths = rss_paths
        self.use_joblib = use_joblib
        self.num_process = num_process
        self.backend = backend
        self.output_poscar = output_poscar
        self.threshold = threshold

    def run_sorting(self):
        os.makedirs("json", exist_ok=True)
        os.makedirs("ghost_minima", exist_ok=True)

        paths_same_comp = defaultdict(list)
        results_same_comp = defaultdict(dict)
        for path_name in self.rss_paths:
            rss_result_path = f"{path_name}/rss_result/rss_results.json"
            with open(rss_result_path) as f:
                loaded_dict = json.load(f)

            rel_path = os.path.relpath(f"{path_name}/opt_struct", start=os.getcwd())
            for i in range(len(loaded_dict["rss_results"])):
                poscar_name = loaded_dict["rss_results"][i]["poscar"].split("/")[-1]
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
            unique_structs, num_opt_struct, integrated_res_paths, pressure = (
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
                print(f"  num_unique_structs:    {len(unique_structs)}", file=f)
                print(
                    f"  input_file_names:      {sorted(integrated_res_paths)}", file=f
                )
                print("", file=f)

            energies = np.array([s.energy for s in unique_structs])
            distances = np.array([s.least_distance for s in unique_structs])

            sort_idx = np.argsort(energies)
            unique_str_sorted = [unique_structs[i] for i in sort_idx]

            is_ghost_minima, ghost_minima_info = detect_ghost_minima(
                energies[sort_idx], distances[sort_idx]
            )
            with open("ghost_minima/dist_minE_struct.dat", "a") as f:
                print(f"{ghost_minima_info[0]:.3f}  {log_name}", file=f)
            if len(ghost_minima_info[1]) > 0:
                with open("ghost_minima/dist_ghost_minima.dat", "a") as f:
                    print(log_name, file=f)
                    print(np.round(ghost_minima_info[1], 3), file=f)

            rss_result_all = log_unique_structures(
                log_name + ".yaml",
                unique_str_sorted,
                is_ghost_minima,
                pressure=pressure,
            )

            with open(f"json/{log_name}.json", "w") as f:
                json.dump(rss_result_all, f)

            if self.output_poscar:
                self.generate_poscars(f"json/{log_name}.json", threshold=self.threshold)

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
            num_opt_struct = yaml_data["general_information"]["num_optimized_structs"]
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

    def generate_poscars(self, json_path: str, threshold=None):
        logname = os.path.basename(json_path).split(".json")[0]
        os.makedirs(f"poscars/{logname}", exist_ok=True)

        with open(json_path) as f:
            loaded_dict = json.load(f)
        rss_results = loaded_dict["rss_results"]

        e_min = None
        for res in rss_results:
            if not res.get("is_ghost_minima") and e_min is None:
                e_min = res["energy"]
            if e_min is not None and threshold is not None:
                diff = abs(e_min - res["energy"])
                if diff * 1000 > threshold:
                    continue

            dest = f"poscars/{logname}/POSCAR_{logname}_No{res['struct_no']}"
            shutil.copy(res["poscar"], dest)

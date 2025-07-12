import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from time import time

import numpy as np
import yaml

from pypolymlp.core.interface_vasp import Vasprun
from rsspolymlp.analysis.ghost_minima import detect_ghost_minima
from rsspolymlp.analysis.unique_struct import (
    UniqueStructureAnalyzer,
    generate_unique_structs,
    log_all_unique_structures,
    log_unique_structures,
)
from rsspolymlp.common.atomic_energy import atomic_energy
from rsspolymlp.common.composition import compute_composition
from rsspolymlp.common.convert_dict import polymlp_struct_from_dict
from rsspolymlp.common.property import PropUtil


class RSSResultSummarizer:

    def __init__(
        self,
        elements,
        result_paths,
        use_joblib,
        num_process: int = -1,
        backend: str = "loky",
        output_poscar: bool = False,
        threshold: float = None,
        parse_vasp: bool = False,
    ):
        self.elements = elements
        self.result_paths = result_paths
        self.use_joblib = use_joblib
        self.num_process = num_process
        self.backend = backend
        self.output_poscar = output_poscar
        self.threshold = threshold
        self.parse_vasp = parse_vasp

        self.num_opt_struct = 0
        self.pressure = None
        self.analyzer = UniqueStructureAnalyzer()

    def run_summarize(self):
        os.makedirs("json", exist_ok=True)
        os.makedirs("ghost_minima", exist_ok=True)

        if not self.parse_vasp:
            paths_same_comp, results_same_comp = self._parse_mlp_result()
            axis_tol = 0.01
            pos_tol = 0.01
        else:
            paths_same_comp, results_same_comp = self._parse_vasp_result()
            axis_tol = 0.05
            pos_tol = 0.03

        for comp_ratio, res_paths in paths_same_comp.items():
            self.num_opt_struct = 0
            self.analyzer = UniqueStructureAnalyzer()

            log_name = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    log_name += f"{self.elements[i]}{comp_ratio[i]}"

            time_start = time()

            log_name = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    log_name += f"{self.elements[i]}{comp_ratio[i]}"
            yaml_name = log_name + ".yaml"
            res_paths, integrated_res_paths = self.initialize_uniq_struct(
                yaml_name, res_paths
            )
            self.sort_in_single_comp(
                res_paths,
                results_same_comp[comp_ratio],
                axis_tol=axis_tol,
                pos_tol=pos_tol,
            )

            time_finish = time() - time_start

            unique_structs = self.analyzer.unique_str

            with open(log_name + ".yaml", "w") as f:
                print("general_information:", file=f)
                print(f"  sorting_time_sec:      {round(time_finish, 2)}", file=f)
                print(f"  pressure_GPa:          {self.pressure}", file=f)
                print(f"  num_optimized_structs: {self.num_opt_struct}", file=f)
                print(f"  num_unique_structs:    {len(unique_structs)}", file=f)
                print(
                    f"  input_file_names:      {sorted(integrated_res_paths)}", file=f
                )
                print("", file=f)

            energies = np.array([s.energy for s in unique_structs])
            distances = np.array([s.least_distance for s in unique_structs])

            sort_idx = np.argsort(energies)
            unique_str_sorted = [unique_structs[i] for i in sort_idx]

            if not self.parse_vasp:
                is_ghost_minima, ghost_minima_info = detect_ghost_minima(
                    energies[sort_idx], distances[sort_idx]
                )
                with open("ghost_minima/dist_minE_struct.dat", "a") as f:
                    print(f"{ghost_minima_info[0]:.3f}  {log_name}", file=f)
                if len(ghost_minima_info[1]) > 0:
                    with open("ghost_minima/dist_ghost_minima.dat", "a") as f:
                        print(log_name, file=f)
                        print(np.round(ghost_minima_info[1], 3), file=f)
            else:
                is_ghost_minima = np.full(energies.shape, False, dtype=bool)

            rss_result_all = log_unique_structures(
                log_name + ".yaml",
                unique_str_sorted,
                is_ghost_minima,
                pressure=self.pressure,
            )

            with open(f"json/{log_name}.json", "w") as f:
                json.dump(rss_result_all, f)

            if self.output_poscar:
                self.generate_poscars(f"json/{log_name}.json", threshold=self.threshold)

            print(log_name, "finished", flush=True)

    def run_summarize_p(self):
        os.makedirs("json", exist_ok=True)

        paths_same_comp, results_same_comp = self._parse_json_result()

        for comp_ratio, res_paths in paths_same_comp.items():
            self.num_opt_struct = 0
            self.analyzer = UniqueStructureAnalyzer()

            log_name = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    log_name += f"{self.elements[i]}{comp_ratio[i]}"

            time_start = time()
            self.sort_in_single_comp(
                res_paths,
                results_same_comp[comp_ratio],
                keep_unique=True,
                axis_tol=0.1,
                pos_tol=0.03,
            )
            time_finish = time() - time_start

            unique_structs = self.analyzer.unique_str_keep
            unique_structs = [sorted(s, key=lambda x: x.energy) for s in unique_structs]

            with open(log_name + ".yaml", "w") as f:
                print("general_information:", file=f)
                print(f"  sorting_time_sec:      {round(time_finish, 2)}", file=f)
                print(f"  num_optimized_structs: {self.num_opt_struct}", file=f)
                print(f"  num_unique_structs:    {len(unique_structs)}", file=f)
                print(f"  input_file_names:      {sorted(res_paths)}", file=f)
                print("", file=f)

            energies = np.array([s[0].energy for s in unique_structs])
            sort_idx = np.argsort(energies)
            unique_str_sorted = [unique_structs[i] for i in sort_idx]
            unique_str_sorted = sorted(
                unique_str_sorted, key=lambda x: len(x), reverse=True
            )

            rss_result_all = log_all_unique_structures(
                log_name + ".yaml",
                unique_str_sorted,
            )

            with open(f"json/{log_name}.json", "w") as f:
                json.dump(rss_result_all, f)

            if self.output_poscar:
                self.generate_poscars(f"json/{log_name}.json", threshold=self.threshold)

            print(log_name, "finished", flush=True)

    def _parse_mlp_result(self):
        paths_same_comp = defaultdict(list)
        results_same_comp = defaultdict(dict)
        for path_name in self.result_paths:
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

        return paths_same_comp, results_same_comp

    def _parse_vasp_result(self):
        paths_same_comp = defaultdict(list)
        results_same_comp = defaultdict(dict)
        for path_name in self.result_paths:
            res_dict = {
                "poscar": None,
                "structure": None,
                "energy": None,
                "spg_list": None,
            }
            try:
                vaspobj = Vasprun(path_name + "/vasprun.xml")
            except Exception:
                continue

            polymlp_st = vaspobj.structure
            objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
            spg_list = objprop.analyze_space_group(polymlp_st.elements)

            energy_dft = vaspobj.energy
            for element in polymlp_st.elements:
                energy_dft -= atomic_energy(element)
            energy_dft /= len(polymlp_st.elements)

            res_dict["poscar"] = path_name + "/vasprun.xml"
            res_dict["structure"] = polymlp_st
            res_dict["energy"] = energy_dft
            res_dict["spg_list"] = spg_list

            comp_res = compute_composition(
                polymlp_st.elements, element_order=self.elements
            )
            comp_ratio = comp_res.comp_ratio
            try:
                tree = ET.parse(path_name + "/vasprun.xml")
                root = tree.getroot()
                for incar_item in root.findall(".//incar/i"):
                    if incar_item.get("name") == "PSTRESS":
                        self.pressure = float(incar_item.text.strip()) / 10
            except Exception:
                self.pressure = None

            paths_same_comp[comp_ratio].append(path_name)
            results_same_comp[comp_ratio][path_name] = {
                "pressure": self.pressure,
                "rss_results": [res_dict],
            }

        return paths_same_comp, results_same_comp

    def _parse_json_result(self):
        paths_same_comp = defaultdict(list)
        results_same_comp = defaultdict(dict)
        for path_name in self.result_paths:
            with open(path_name) as f:
                loaded_dict = json.load(f)

            target_elements = loaded_dict["elements"]
            comp_ratio = tuple(loaded_dict["comp_ratio"])
            _dicts = dict(zip(target_elements, comp_ratio))
            comp_ratio_orderd = tuple(_dicts.get(el, 0) for el in self.elements)

            paths_same_comp[comp_ratio_orderd].append(path_name)
            results_same_comp[comp_ratio_orderd][path_name] = loaded_dict

        return paths_same_comp, results_same_comp

    def sort_in_single_comp(
        self,
        result_paths,
        rss_result_dict,
        keep_unique=False,
        axis_tol=0.01,
        pos_tol=0.01,
    ):
        rss_results = []
        for res_path in result_paths:
            loaded_dict = rss_result_dict[res_path]
            rss_res = loaded_dict["rss_results"]
            if not self.parse_vasp:
                for res in rss_res:
                    res["structure"] = polymlp_struct_from_dict(res["structure"])
                    res["struct_no"] = None

            pressure = loaded_dict.get("pressure")
            for res in rss_res:
                res["pressure"] = pressure
            self.pressure = pressure

            rss_results.extend(rss_res)

        unique_structs = generate_unique_structs(
            rss_results,
            use_joblib=self.use_joblib,
            num_process=self.num_process,
            backend=self.backend,
        )
        self.num_opt_struct += len(unique_structs)

        for unique_struct in unique_structs:
            self.analyzer.identify_duplicate_struct(
                unique_struct=unique_struct,
                keep_unique=keep_unique,
                axis_tol=axis_tol,
                pos_tol=pos_tol,
            )

    def initialize_uniq_struct(self, yaml_name, result_paths):
        pre_result_paths = []

        if os.path.isfile(yaml_name):
            with open(yaml_name) as f:
                yaml_data = yaml.safe_load(f)
            self.num_opt_struct = yaml_data["general_information"][
                "num_optimized_structs"
            ]
            pre_result_paths = yaml_data["general_information"]["input_file_names"]

            with open(f"./json/{yaml_name.split('.yaml')[0]}.json") as f:
                loaded_dict = json.load(f)
            rss_results = loaded_dict["rss_results"]
            for i in range(len(rss_results)):
                rss_results[i]["structure"] = polymlp_struct_from_dict(
                    rss_results[i]["structure"]
                )
            self.pressure = loaded_dict["pressure"]

            unique_structs = generate_unique_structs(
                rss_results,
                use_joblib=self.use_joblib,
                num_process=self.num_process,
                backend=self.backend,
            )
            self.analyzer._initialize_unique_structs(unique_structs)

        not_processed_path = list(set(result_paths) - set(pre_result_paths))
        integrated_res_paths = list(set(result_paths) | set(pre_result_paths))

        return not_processed_path, integrated_res_paths

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

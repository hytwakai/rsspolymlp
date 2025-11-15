import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np

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
        result_paths: list = [],
        parent_paths: list = [],
        element_order: list = None,
        num_process: int = -1,
        backend: str = "loky",
        symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
        output_poscar: bool = False,
        thresholds: list[float] = None,
        parse_vasp: bool = False,
        update_parent: bool = False,
    ):
        self.result_paths = result_paths
        self.parent_paths = parent_paths
        self.result_paths.extend(self.parent_paths)

        self.element_order = element_order
        self.num_process = num_process
        self.backend = backend
        self.symprec_set = symprec_set
        self.output_poscar = output_poscar
        self.thresholds = thresholds
        self.parse_vasp = parse_vasp
        self.update_parent = update_parent

        self.pressure = None
        self.analyzer = UniqueStructureAnalyzer()

    def run_summarize(self):
        os.makedirs("json", exist_ok=True)

        parent_path_ref = {}
        if len(self.parent_paths) == 0 and os.path.isdir("json"):
            self.parent_paths = glob.glob("json/*.json")

        for json_path in self.parent_paths:
            with open(json_path) as f:
                loaded_dict = json.load(f)

            target_elements = loaded_dict["elements"]
            comp_ratio = tuple(loaded_dict["comp_ratio"])
            if self.element_order is not None:
                _dicts = dict(zip(target_elements, comp_ratio))
                comp_ratio = tuple(_dicts.get(el, 0) for el in self.element_order)
                target_elements = self.element_order

            composition_tag = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    composition_tag += f"{target_elements[i]}{comp_ratio[i]}"
            parent_path_ref[composition_tag] = json_path

        if not self.parse_vasp:
            paths_same_comp, results_same_comp = self._parse_json_result(
                parse_rss_result=True
            )
            axis_tol = 0.03
            pos_tol = 0.03
        else:
            paths_same_comp, results_same_comp = self._parse_vasp_result()
            axis_tol = 0.1
            pos_tol = 0.1

        for composition_tag, res_paths in paths_same_comp.items():
            print(f"Composition {composition_tag}: summarizing...")
            self.analyzer = UniqueStructureAnalyzer()

            time_start = time()

            processed_paths = set()
            if composition_tag in parent_path_ref:
                print(
                    f" - Processing result file (parent): {parent_path_ref[composition_tag]}"
                )
                processed_paths = self.initialize_uniq_struct(
                    parent_path_ref[composition_tag]
                )

            for res_path in res_paths:
                if (
                    composition_tag not in parent_path_ref
                    or parent_path_ref[composition_tag] != res_path
                ):
                    rss_results = [
                        r
                        for r in results_same_comp[composition_tag][res_path][
                            "rss_results"
                        ]
                        if r["struct_path"] not in processed_paths
                    ]
                    if not len(rss_results) == 0:
                        print(f" - Processing result file: {res_path}", flush=True)
                        self.remove_duplicates_in_comp(
                            results_same_comp[composition_tag][res_path],
                            processed_paths,
                            axis_tol=axis_tol,
                            pos_tol=pos_tol,
                        )

            time_finish = time() - time_start

            unique_structs = self.analyzer.unique_str

            num_opt_str = 0
            for ustr in unique_structs:
                num_opt_str += len(list(ustr.dupstr_paths))

            with open(composition_tag + ".yaml", "w") as f:
                print("general_information:", file=f)
                print(f"  sorting_time_sec:      {round(time_finish, 2)}", file=f)
                print(f"  pressure_GPa:          {self.pressure}", file=f)
                print(f"  num_optimized_structs: {num_opt_str}", file=f)
                print(f"  num_unique_structs:    {len(unique_structs)}", file=f)
                print("", file=f)

            energies = np.array([s.energy for s in unique_structs])
            distances = np.array([s.least_distance for s in unique_structs])

            sort_idx = np.argsort(energies)
            unique_str_sorted = [unique_structs[i] for i in sort_idx]

            if not self.parse_vasp:
                os.makedirs("ghost_minima", exist_ok=True)
                is_ghost_minima, ghost_minima_info = detect_ghost_minima(
                    energies[sort_idx], distances[sort_idx]
                )
                with open("ghost_minima/dist_minE_struct.dat", "a") as f:
                    print(f"{ghost_minima_info[0]:.3f}  {composition_tag}", file=f)
                if len(ghost_minima_info[1]) > 0:
                    with open("ghost_minima/dist_ghost_minima.dat", "a") as f:
                        print(composition_tag, file=f)
                        print(np.round(ghost_minima_info[1], 3), file=f)
            else:
                is_ghost_minima = None

            rss_result_all = log_unique_structures(
                composition_tag + ".yaml",
                unique_str_sorted,
                is_ghost_minima,
                pressure=self.pressure,
            )

            with open(f"json/{composition_tag}.json", "w") as f:
                json.dump(rss_result_all, f)

            if self.thresholds is not None or self.output_poscar is not False:
                self.generate_poscars(
                    f"json/{composition_tag}.json",
                    thresholds=self.thresholds,
                    output_poscar=self.output_poscar,
                )

            print(f"Composition {composition_tag}: finished summarizing.", flush=True)

    def run_summarize_p(self):
        os.makedirs("json", exist_ok=True)

        paths_same_comp, results_same_comp = self._parse_json_result()

        for composition_tag, res_paths in paths_same_comp.items():
            self.analyzer = UniqueStructureAnalyzer()

            time_start = time()
            for res_path in res_paths:
                self.remove_duplicates_in_comp(
                    results_same_comp[composition_tag][res_path],
                    standardize_axis=True,
                    keep_unique=True,
                    axis_tol=0.03,
                    pos_tol=0.03,
                )
            time_finish = time() - time_start

            unique_structs = self.analyzer.unique_str_keep
            unique_structs = [sorted(s, key=lambda x: x.energy) for s in unique_structs]

            with open(composition_tag + ".yaml", "w") as f:
                print("general_information:", file=f)
                print(f"  sorting_time_sec:      {round(time_finish, 2)}", file=f)
                print(f"  num_unique_structs:    {len(unique_structs)}", file=f)
                print("", file=f)

            energies = np.array([s[0].energy for s in unique_structs])
            sort_idx = np.argsort(energies)
            unique_str_sorted = [unique_structs[i] for i in sort_idx]
            unique_str_sorted = sorted(
                unique_str_sorted, key=lambda x: len(x), reverse=True
            )

            rss_result_all = log_all_unique_structures(
                composition_tag + ".yaml",
                unique_str_sorted,
            )

            with open(f"json/{composition_tag}.json", "w") as f:
                json.dump(rss_result_all, f)

            if self.thresholds is not None or self.output_poscar is not False:
                self.generate_poscars(
                    f"json/{composition_tag}.json",
                    thresholds=self.thresholds,
                    output_poscar=self.output_poscar,
                )

            print(composition_tag, "finished", flush=True)

    def remove_duplicates_in_comp(
        self,
        loaded_dict,
        standardize_axis=False,
        keep_unique=False,
        axis_tol=0.01,
        pos_tol=0.01,
    ):
        rss_results = loaded_dict["rss_results"]
        pressure = loaded_dict.get("pressure")
        for res in rss_results:
            res["pressure"] = pressure
        self.pressure = pressure

        print("   - Converting reduced crystal structure representation...")
        unique_structs = generate_unique_structs(
            rss_results,
            num_process=self.num_process,
            backend=self.backend,
            symprec_set1=self.symprec_set,
            standardize_axis=standardize_axis,
        )

        print("   - Eliminating duplicate structures...")
        for unique_struct in unique_structs:
            self.analyzer.identify_duplicate_struct(
                unique_struct=unique_struct,
                keep_unique=keep_unique,
                axis_tol=axis_tol,
                pos_tol=pos_tol,
            )

    def initialize_uniq_struct(self, json_path):
        processed_paths = []

        with open(json_path) as f:
            loaded_dict = json.load(f)
        rss_results = loaded_dict["rss_results"]
        for r in rss_results:
            r["structure"] = polymlp_struct_from_dict(r["structure"])
            processed_paths.extend(r["dupstr_paths"])
            r["dupstr_paths"] = set(r["dupstr_paths"])
        self.pressure = loaded_dict["pressure"]

        print("   - Converting reduced crystal structure representation...")
        unique_structs = generate_unique_structs(
            rss_results,
            num_process=self.num_process,
            backend=self.backend,
            symprec_set1=self.symprec_set,
        )

        if self.update_parent:
            print("   - Eliminating duplicate structures...")
            for unique_struct in unique_structs:
                self.analyzer.identify_duplicate_struct(
                    unique_struct=unique_struct,
                    axis_tol=0.03,
                    pos_tol=0.03,
                )
        else:
            self.analyzer._initialize_unique_structs(unique_structs)

        return set(processed_paths)

    def generate_poscars(self, json_path: str, thresholds=None, output_poscar=False):
        if thresholds is None:
            thresholds = [None]

        struct_counts = []
        logname = os.path.basename(json_path).split(".json")[0]
        for threshold in thresholds:
            if threshold is None:
                dir_name = "poscars"
            else:
                threshold = float(threshold)
                dir_name = f"poscars_{threshold}"
            if output_poscar:
                os.makedirs(f"{dir_name}/{logname}", exist_ok=True)

            print(f"Threshold (meV/atom): {threshold}")

            with open(json_path) as f:
                loaded_dict = json.load(f)
            rss_results = loaded_dict["rss_results"]

            e_min = None
            struct_count = 0
            for res in rss_results:
                if not res.get("is_ghost_minima") and e_min is None:
                    e_min = res["energy"]
                elif res.get("is_ghost_minima") and output_poscar:
                    dest = f"{dir_name}/{logname}/POSCAR_{logname}_No{res['struct_no']}"
                    shutil.copy(res["struct_path"], dest)
                    continue

                if e_min is not None and threshold is not None:
                    diff = abs(e_min - res["energy"])
                    if diff * 1000 > threshold:
                        continue

                if output_poscar:
                    dest = f"{dir_name}/{logname}/POSCAR_{logname}_No{res['struct_no']}"
                    shutil.copy(res["struct_path"], dest)
                struct_count += 1

            struct_counts.append(struct_count)
            print("Number of local minimum structures:", struct_count)

        return struct_counts

    def _parse_json_result(self, parse_rss_result=False):

        def resolve_path(base: Path, p):
            cwd = Path.cwd()
            if p is None:
                return None
            p = Path(p)
            target = (
                base / p if "opt_struct" in p.parts else base / "opt_struct" / p.name
            )
            return os.path.relpath(target, start=cwd)

        paths_same_comp = defaultdict(list)
        results_same_comp = defaultdict(dict)
        for path_name in self.result_paths:
            with open(path_name) as f:
                loaded_dict = json.load(f)

            if parse_rss_result:
                base = Path(path_name).parents[1]
                for r in loaded_dict["rss_results"]:
                    r["struct_path"] = resolve_path(base, r["struct_path"])
                    r["dupstr_paths"] = {
                        resolve_path(base, p) for p in r["dupstr_paths"]
                    }
                    r["structure"] = polymlp_struct_from_dict(r["structure"])
                    r["struct_no"] = None

            target_elements = loaded_dict["elements"]
            comp_ratio = tuple(loaded_dict["comp_ratio"])
            if self.element_order is not None:
                _dicts = dict(zip(target_elements, comp_ratio))
                comp_ratio = tuple(_dicts.get(el, 0) for el in self.element_order)
                target_elements = self.element_order

            composition_tag = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    composition_tag += f"{target_elements[i]}{comp_ratio[i]}"

            paths_same_comp[composition_tag].append(path_name)
            results_same_comp[composition_tag][path_name] = loaded_dict

        return paths_same_comp, results_same_comp

    def _parse_vasp_result(self):
        paths_same_comp = defaultdict(list)
        results_same_comp = defaultdict(dict)
        for path_name in self.result_paths:
            res_dict = {
                "struct_path": None,
                "structure": None,
                "energy": None,
                "spg_list": None,
                "dupstr_paths": None,
            }
            try:
                vaspobj = Vasprun(path_name + "/vasprun.xml")
            except Exception:
                print("ParseError:", path_name + "/vasprun.xml")
                continue

            polymlp_st = vaspobj.structure
            objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
            spg_list = objprop.analyze_space_group(polymlp_st.elements)

            energy_dft = vaspobj.energy
            for element in polymlp_st.elements:
                energy_dft -= atomic_energy(element)
            energy_dft /= len(polymlp_st.elements)

            if energy_dft < -10:
                print(path_name, "exhibits an unphysically low energy. Skipping.")
                continue

            res_dict["struct_path"] = os.path.relpath(
                path_name + "/vasprun.xml", os.getcwd()
            )
            res_dict["structure"] = polymlp_st
            res_dict["energy"] = energy_dft
            res_dict["spg_list"] = spg_list
            res_dict["dupstr_paths"] = {res_dict["struct_path"]}

            comp_res = compute_composition(
                polymlp_st.elements, element_order=self.element_order
            )
            comp_ratio = comp_res.comp_ratio
            if self.element_order is not None:
                target_elements = self.element_order
            else:
                target_elements = comp_res.unique_elements

            try:
                tree = ET.parse(path_name + "/vasprun.xml")
                root = tree.getroot()
                for incar_item in root.findall(".//incar/i"):
                    if incar_item.get("name") == "PSTRESS":
                        self.pressure = float(incar_item.text.strip()) / 10
            except Exception:
                self.pressure = None

            composition_tag = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    composition_tag += f"{target_elements[i]}{comp_ratio[i]}"

            paths_same_comp[composition_tag].append(path_name)
            results_same_comp[composition_tag][path_name] = {
                "pressure": self.pressure,
                "rss_results": [res_dict],
            }

        return paths_same_comp, results_same_comp

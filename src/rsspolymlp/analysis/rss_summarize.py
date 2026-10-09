import glob
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Optional

import numpy as np

from pypolymlp.utils.vasp_utils import write_poscar_file
from rsspolymlp.analysis.ghost_minima import detect_ghost_minima
from rsspolymlp.analysis.struct_matcher.gen_redrep import (
    ReducedStructReps,
    generate_redreps_parallel,
)
from rsspolymlp.analysis.struct_matcher.struct_match import (
    UniqueStructIdentifier,
    identify_unique_structure,
)
from rsspolymlp.common.composition import compute_composition
from rsspolymlp.common.convert_dict import (
    polymlp_struct_from_dict,
    polymlp_struct_to_dict,
)
from rsspolymlp.common.interface_vasp import Vasprun, parse_properties_from_vasprun
from rsspolymlp.common.property import PropUtil


class RSSResultSummarizer:

    def __init__(
        self,
        result_paths: list = [],
        parent_paths: list = [],
        element_order: list = None,
        num_process: int = -1,
        backend: str = "loky",
        primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
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
        self.primitive_symprecs = primitive_symprecs
        self.output_poscar = output_poscar
        self.thresholds = thresholds
        self.parse_vasp = parse_vasp
        self.update_parent = update_parent

        self.pressure = None
        self.analyzer = UniqueStructIdentifier()

    def run_summarize(self):
        os.makedirs("json", exist_ok=True)
        if len(self.parent_paths) == 0:
            self.parent_paths = glob.glob("json/*.json")

        parent_path_ref = {}
        for json_path in self.parent_paths:
            with open(json_path) as f:
                loaded_dict = json.load(f)
            target_elements = loaded_dict["elements"]
            comp_ratio = tuple(loaded_dict["comp_ratio"])
            if self.element_order is not None:
                _dicts = dict(zip(target_elements, comp_ratio))
                comp_ratio = tuple(_dicts.get(el, 0) for el in self.element_order)
                target_elements = self.element_order
            comp_tag = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    comp_tag += f"{target_elements[i]}{comp_ratio[i]}"
            parent_path_ref[comp_tag] = json_path

        if not self.parse_vasp:
            json_paths_c, rss_results_c = self._parse_json_result()
            axis_tol = 0.03
            pos_tol = 0.03
        else:
            json_paths_c, rss_results_c = self._parse_vasp_result()
            axis_tol = 0.1
            pos_tol = 0.1

        for comp_tag, res_paths in json_paths_c.items():
            print(f"Composition {comp_tag}: summarizing...")
            self.analyzer = UniqueStructIdentifier()

            time_start = time()

            processed_paths = set()
            if comp_tag in parent_path_ref:
                print(
                    f" - Processing result file (parent): {parent_path_ref[comp_tag]}"
                )
                processed_paths = self.initialize_uniq_struct(parent_path_ref[comp_tag])

            for res_path in res_paths:
                if (
                    comp_tag not in parent_path_ref
                    or parent_path_ref[comp_tag] != res_path
                ):
                    rss_results = [
                        r
                        for r in rss_results_c[comp_tag][res_path]
                        if r["struct_path"] not in processed_paths
                    ]
                    if len(rss_results) > 0:
                        print(f" - Processing result file: {res_path}", flush=True)
                        self.analyzer = identify_unique_structure(
                            rss_results_c[comp_tag][res_path],
                            axis_tol=axis_tol,
                            pos_tol=pos_tol,
                            pre_analyzer=self.analyzer,
                        )

            time_finish = time() - time_start

            unique_structs = self.analyzer.unique_str
            unique_str_prop = self.analyzer.unique_str_prop
            enthalpies = np.array([s["energy"] for s in unique_str_prop])
            distances = np.array([s["least_distance"] for s in unique_str_prop])
            sort_idx = np.argsort(enthalpies)

            if not self.parse_vasp:
                os.makedirs("ghost_minima", exist_ok=True)
                is_ghost_minima, ghost_minima_info = detect_ghost_minima(
                    enthalpies[sort_idx], distances[sort_idx]
                )
                with open("ghost_minima/dist_minE_struct.dat", "a") as f:
                    print(f"{ghost_minima_info[0]:.3f}  {comp_tag}", file=f)
                if len(ghost_minima_info[1]) > 0:
                    with open("ghost_minima/dist_ghost_minima.dat", "a") as f:
                        print(comp_tag, file=f)
                        print(np.round(ghost_minima_info[1], 3), file=f)
            else:
                is_ghost_minima = None

            unique_str_sorted = [unique_structs[i] for i in sort_idx]
            if self.thresholds is not None:
                e_min = None
                for idx, e in enumerate(enthalpies[sort_idx]):
                    if e_min is None:
                        e_min = e
                    if e - e_min > self.thresholds[0] * 0.001:
                        unique_str_sorted = unique_str_sorted[:idx]
                        break
                output_file = f"{comp_tag}_{self.thresholds[0]}"
            else:
                output_file = comp_tag

            num_opt_str = 0
            for ustr in unique_structs:
                num_opt_str += len(list(ustr.dupstr_paths))

            with open(output_file + ".yaml", "w") as f:
                print("general_information:", file=f)
                print(f"  sorting_time_sec:      {round(time_finish, 2)}", file=f)
                print(f"  pressure_GPa:          {self.pressure}", file=f)
                print(f"  num_optimized_structs: {num_opt_str}", file=f)
                print(f"  num_unique_structs:    {len(unique_structs)}", file=f)
                print("", file=f)
            log_unique_structures(
                output_file + ".yaml",
                unique_str_sorted,
                enthalpies[sort_idx],
                is_ghost_minima,
            )
            dump_json_rss_results(
                f"json/{output_file}.json",
                unique_str_sorted,
                enthalpies[sort_idx],
                is_ghost_minima,
                self.pressure,
            )
            if self.thresholds is not None or self.output_poscar is not False:
                self.generate_poscars(
                    f"json/{output_file}.json",
                    thresholds=self.thresholds,
                    output_poscar=self.output_poscar,
                )
            print(f"Composition {comp_tag}: finished summarizing.", flush=True)

    def run_summarize_p(self):
        os.makedirs("json", exist_ok=True)

        json_paths_c, rss_results_c = self._parse_json_result()

        for comp_tag, res_paths in json_paths_c.items():
            print(f"Composition {comp_tag}: summarizing...")
            self.analyzer = UniqueStructIdentifier()

            if not self.parse_vasp:
                axis_tol = 0.03
                pos_tol = 0.03
            else:
                axis_tol = 0.1
                pos_tol = 0.1

            time_start = time()
            for res_path in res_paths:
                print(f" - Processing result file (parent): {res_path}")
                self.analyzer = identify_unique_structure(
                    rss_results_c[comp_tag][res_path],
                    standardize_axis=True,
                    keep_unique=True,
                    axis_tol=axis_tol,
                    pos_tol=pos_tol,
                    pre_analyzer=self.analyzer,
                )
            time_finish = time() - time_start

            unique_structs = self.analyzer.unique_str_keep
            unique_str_prop = self.analyzer.unique_str_prop_keep
            enthalpies = np.array([s[0]["energy"] for s in unique_str_prop])

            sort_idx = np.argsort(enthalpies)
            sort_idx = sorted(
                sort_idx,
                key=lambda i: len(unique_str_sorted[i]),
                reverse=True,
            )
            unique_str_sorted = [unique_structs[i] for i in sort_idx]
            unique_str_prop_sorted = [unique_str_prop[i] for i in sort_idx]

            with open(comp_tag + ".yaml", "w") as f:
                print("general_information:", file=f)
                print(f"  sorting_time_sec:      {round(time_finish, 2)}", file=f)
                print(f"  num_unique_structs:    {len(unique_structs)}", file=f)
                print("", file=f)
            rss_result_all = log_all_unique_structures(
                comp_tag + ".yaml",
                unique_str_sorted,
                unique_str_prop_sorted,
            )
            with open(f"json/{comp_tag}.json", "w") as f:
                json.dump(rss_result_all, f)
            if self.thresholds is not None or self.output_poscar is not False:
                self.generate_poscars(
                    f"json/{comp_tag}.json",
                    thresholds=self.thresholds,
                    output_poscar=self.output_poscar,
                )
            print(comp_tag, "finished", flush=True)

    def initialize_uniq_struct(self, json_path):
        with open(json_path) as f:
            loaded_dict = json.load(f)
        self.pressure = loaded_dict["pressure"]
        rss_results = loaded_dict["rss_results"]

        processed_paths = []
        for r in rss_results:
            r["structure"] = polymlp_struct_from_dict(r["structure"])
            processed_paths.extend(r["dupstr_paths"])
            r["dupstr_paths"] = set(r["dupstr_paths"])

        print("   - Converting reduced crystal structure representation...")
        redreps_list = generate_redreps_parallel(
            rss_results,
            num_process=self.num_process,
            backend=self.backend,
            symprec_set1=self.symprec_set,
        )
        if self.update_parent:
            print("   - Eliminating duplicate structures...")
            for idx, redreps in enumerate(redreps_list):
                self.analyzer.identify_duplicate_struct(
                    unique_struct=redreps,
                    other_properties=rss_results[idx],
                    axis_tol=0.03,
                    pos_tol=0.03,
                )
        else:
            self.analyzer._initialize_unique_structs(redreps_list, rss_results)

        return set(processed_paths)

    def generate_poscars(self, json_path: str, thresholds=None, output_poscar=False):
        e_min = None
        with open(json_path) as f:
            loaded_dict = json.load(f)
        rss_results = loaded_dict["rss_results"]
        for res in rss_results:
            if not res.get("is_ghost_minima") and e_min is None:
                e_min = res["energy"]
            else:
                continue

        struct_count = 0
        struct_counts = []
        logname = os.path.basename(json_path).split(".json")[0]
        if thresholds is None:
            thresholds = [None]
        for threshold in thresholds:
            print(f"Threshold (meV/atom): {threshold}")
            dir_name = "poscars"
            dir_name_vasp = "vaspruns"
            if threshold is not None:
                threshold = float(threshold)
                dir_name += f"_{threshold}"
                dir_name_vasp += f"_{threshold}"

            if output_poscar:
                os.makedirs(f"{dir_name}/{logname}", exist_ok=True)
                if self.parse_vasp:
                    os.makedirs(f"{dir_name_vasp}/{logname}", exist_ok=True)

            for res in rss_results:
                if threshold is not None:
                    diff = res["energy"] - e_min
                    if diff * 1000 > threshold:
                        continue
                if output_poscar:
                    dest = (
                        f"{dir_name}/{logname}/POSCAR_{logname}_No{res['struct_tag']}"
                    )
                    dest_vasp = (
                        f"{dir_name_vasp}/{logname}/{logname}_No{res['struct_tag']}.xml"
                    )
                    if self.parse_vasp:
                        poscar_path = f'{os.path.dirname(res["struct_path"])}/POSCAR'
                        if os.path.isfile(poscar_path):
                            shutil.copy(poscar_path, dest)
                        else:
                            try:
                                polymlp_st = Vasprun(res["struct_path"]).structure
                            except Exception:
                                print(res["struct_path"], "failed")
                                continue
                            write_poscar_file(polymlp_st, filename=dest)
                        shutil.copy(res["struct_path"], dest_vasp)
                    else:
                        shutil.copy(res["struct_path"], dest)
                struct_count += 1

            struct_counts.append(struct_count)
            print("Number of local minimum structures:", struct_count)
        return struct_counts

    def _parse_json_result(self):

        def resolve_path(base: Path, p):
            cwd = Path.cwd()
            target = Path(p)
            if p is None:
                return None
            if not self.parse_vasp:
                p = Path(p)
                target = (
                    base / p
                    if "opt_struct" in p.parts
                    else base / "opt_struct" / p.name
                )
            return os.path.relpath(target, start=cwd)

        json_paths_c = defaultdict(list)
        rss_results_c = defaultdict(dict)
        for path_name in self.result_paths:
            with open(path_name) as f:
                loaded_dict = json.load(f)
            pressure = loaded_dict.get("pressure")

            base = Path(path_name).parents[1]
            for r in loaded_dict["rss_results"]:
                r["struct_path"] = resolve_path(base, r["struct_path"])
                r["dupstr_paths"] = {resolve_path(base, p) for p in r["dupstr_paths"]}
                r["structure"] = polymlp_struct_from_dict(r["structure"])
                r["struct_tag"] = None
                r["pressure"] = pressure

            target_elements = loaded_dict["elements"]
            comp_ratio = tuple(loaded_dict["comp_ratio"])
            if self.element_order is not None:
                _dicts = dict(zip(target_elements, comp_ratio))
                comp_ratio = tuple(_dicts.get(el, 0) for el in self.element_order)
                target_elements = self.element_order

            comp_tag = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    comp_tag += f"{target_elements[i]}{comp_ratio[i]}"

            json_paths_c[comp_tag].append(path_name)
            rss_results_c[comp_tag][path_name] = loaded_dict["rss_results"]

        return json_paths_c, rss_results_c

    def _parse_vasp_result(self):
        json_paths_c = defaultdict(list)
        rss_results_c = defaultdict(dict)
        for path_name in self.result_paths:
            res_dict = {
                "struct_path": None,
                "structure": None,
                "energy": None,
                "spg_list": None,
                "dupstr_paths": None,
                "pressure": None,
            }
            try:
                polymlp_st, (energy_dft, _, _) = parse_properties_from_vasprun(
                    path_name + "/vasprun.xml",
                )
            except Exception:
                print("ParseError:", path_name + "/vasprun.xml")
                continue

            energy_dft /= len(polymlp_st.elements)
            if energy_dft < -10:
                print(path_name, "exhibits an unphysically low energy. Skipping.")
                continue

            objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
            spg_list = objprop.analyze_space_group(polymlp_st.elements)

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
                        res_dict["pressure"] = self.pressure
            except Exception:
                self.pressure = None

            comp_tag = ""
            for i in range(len(comp_ratio)):
                if not comp_ratio[i] == 0:
                    comp_tag += f"{target_elements[i]}{comp_ratio[i]}"
            json_paths_c[comp_tag].append(path_name)
            rss_results_c[comp_tag][path_name] = [res_dict]

        return json_paths_c, rss_results_c


def dump_json_rss_results(
    file_name: str,
    redreps_list: list[ReducedStructReps],
    enthalpy_lists: list[float],
    is_ghost_minima=None,
    pressure=None,
):
    if is_ghost_minima is None:
        is_ghost_minima = np.full_like(redreps_list, False, dtype=bool)

    rss_results = []
    for idx, _str in enumerate(redreps_list):
        polymlp_st = _str.original_polymlp_st
        objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
        rss_results.append(
            {
                "struct_path": _str.struct_path,
                "structure": polymlp_struct_to_dict(polymlp_st),
                "energy": enthalpy_lists[idx],
                "axis_abc": objprop.abc,
                "least_distance": objprop.least_distance,
                "volume": objprop.volume,
                "pressure": pressure,
                "struct_tag": _str.struct_tag,
                "dupstr_paths": list(_str.dupstr_paths),
                "is_ghost_minima": bool(is_ghost_minima[idx]),
            }
        )
    if len(redreps_list) > 0:
        comp_res = compute_composition(redreps_list[0].original_polymlp_st.elements)
        rss_result_all = {
            "elements": comp_res.unique_elements.tolist(),
            "comp_ratio": comp_res.comp_ratio,
            "pressure": pressure,
            "rss_results": rss_results,
        }
    else:
        rss_result_all = {}

    if not rss_result_all == {}:
        with open(file_name, "w") as f:
            json.dump(rss_result_all, f)

    return rss_result_all


def log_unique_structures(
    file_name: str,
    redreps_list: list[ReducedStructReps],
    enthalpy_lists: Optional[list[float]] = None,
    is_ghost_minima=None,
    unique_struct_iters=None,
):
    if is_ghost_minima is None:
        is_ghost_minima = np.full_like(redreps_list, False, dtype=bool)
    energy_min = None
    if enthalpy_lists is not None:
        for i in range(len(enthalpy_lists)):
            if not is_ghost_minima[i]:
                energy_min = enthalpy_lists[i]
                break

    struct_num_max = max(
        (_s.struct_tag for _s in redreps_list if _s.struct_tag is not None), default=0
    )
    for _s in redreps_list:
        if _s.struct_tag is None:
            struct_num_max += 1
            _s.struct_tag = struct_num_max

    with open(file_name, "a") as f:
        print("unique_structures:", file=f)
        for is_ghost in [False, True]:
            for idx, _str in enumerate(redreps_list):
                if energy_min is not None:
                    e_diff = round((enthalpy_lists[idx] - energy_min) * 1000, 2)
                    if (not is_ghost and e_diff < -300) or (
                        is_ghost and e_diff >= -300
                    ):
                        continue
                elif is_ghost:
                    continue

                if enthalpy_lists is not None:
                    e = enthalpy_lists[idx]
                else:
                    e = None
                polymlp_st = _str.original_polymlp_st
                objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
                dupstr_paths = list(_str.dupstr_paths)

                print(f"  - struct_No: {_str.struct_tag}", file=f)
                print(f"    struct_path: {_str.struct_path}", file=f)
                if energy_min is not None:
                    print(f"    energy_diff_meV_per_atom: {e_diff}", file=f)
                print(f"    n_duplicates: {len(dupstr_paths)}", file=f)
                print(f"    enthalpy: {e}", file=f)
                print(f"    axis: {objprop.abc}", file=f)
                print(
                    f"    positions: {polymlp_st.positions.T.tolist()}",
                    file=f,
                )
                print(f"    elements: {polymlp_st.elements}", file=f)
                print(f"    space_group: {_str.spg_list}", file=f)

                info = [
                    f"{len(polymlp_st.elements)} atom",
                    f"distance {round(objprop.least_distance, 3)} (Ang.)",
                    f"volume {round(objprop.volume, 2)} (A^3/atom)",
                ]
                if unique_struct_iters is not None:
                    info.append(f"iteration {unique_struct_iters[idx]}")
                print(f"    other_info: {' / '.join(info)}", file=f)

                if is_ghost_minima[idx]:
                    print("    ghost_minima_flag: true", file=f)


def log_all_unique_structures(
    file_name,
    unique_structs: list[list[ReducedStructReps]],
    unique_str_prop: Optional[list[list[dict]]] = None,
    unique_other_prop=None,
):
    rss_results = []
    with open(file_name, "a") as f:
        print("unique_structures:", file=f)
        for idx1, _str in enumerate(unique_structs):
            print(f"  - struct_No: {idx1 + 1}", file=f)
            print("    structures:", file=f)
            for idx2, _str in enumerate(unique_structs[idx1]):
                if unique_str_prop is not None:
                    e = unique_str_prop[idx1][idx2]["energy"]
                else:
                    e = None
                polymlp_st = _str.original_polymlp_st
                objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
                print(f"    - sub_struct_No: '{idx1 + 1}_{idx2 + 1}'", file=f)
                print(f"      struct_path: {_str.struct_path}", file=f)
                print(f"      pressure: {_str.pressure}", file=f)
                print(f"      enthalpy: {e}", file=f)
                print(f"      axis: {objprop.abc}", file=f)
                print(
                    f"      positions: {polymlp_st.positions.T.tolist()}",
                    file=f,
                )
                print(f"      elements: {polymlp_st.elements}", file=f)
                print(f"      space_group: {_str.spg_list}", file=f)

                info = [
                    f"{len(polymlp_st.elements)} atom",
                    f"distance {round(objprop.least_distance, 3)} (Ang.)",
                    f"volume {round(objprop.volume, 2)} (A^3/atom)",
                ]
                print(f"      other_info: {' / '.join(info)}", file=f)

                _res = {}
                _res["struct_path"] = _str.struct_path
                polymlp_st_dict = polymlp_struct_to_dict(polymlp_st)
                _res["structure"] = polymlp_st_dict
                _res["energy"] = e
                _res["pressure"] = None
                _res["spg_list"] = _str.spg_list
                _res["struct_tag"] = f"{idx1 + 1}_{idx2 + 1}"
                _res["is_ghost_minima"] = False
                rss_results.append(_res)
            if unique_other_prop is not None:
                print("    properties:", unique_other_prop[idx1], file=f)

    comp_res = compute_composition(unique_structs[0][0].original_structure.elements)

    rss_result_all = {
        "elements": comp_res.unique_elements.tolist(),
        "comp_ratio": comp_res.comp_ratio,
        "pressure": None,
        "rss_results": rss_results,
    }

    return rss_result_all

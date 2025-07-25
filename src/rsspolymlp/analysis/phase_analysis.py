import json
import os
from collections import defaultdict

import numpy as np
import yaml
from scipy.spatial import ConvexHull

from pypolymlp.core.interface_vasp import Vasprun
from rsspolymlp.common.atomic_energy import atomic_energy
from rsspolymlp.common.composition import compute_composition


class ConvexHullAnalyzer:
    """Analyze phase stability via the convex hull of formation energies."""

    def __init__(self, elements: np.ndarray = []):
        self.elements = np.array(elements)
        # Initialize per-composition storage
        self._composition_data = {}
        # Convex hull object
        self.hull = None
        # Results for global minima on the hull
        self.hull_energies = None
        self.hull_compositions = None
        self.hull_paths = None

    @property
    def composition_data(self):
        return self._composition_data

    @composition_data.setter
    def composition_data(self, data):
        """
        composition_data maps each composition ratio (tuple) to a dict with keys:
            'energy': np.ndarray of energies,
            'input_path': np.ndarray of file paths,
            'struct_tag': np.ndarray of structure tags.
        """
        if not isinstance(data, dict):
            raise TypeError("composition_data must be a dictionary.")

        for k, v in data.items():
            if not isinstance(k, tuple):
                raise ValueError(f"Key {k} must be a tuple.")
            if not isinstance(v, dict):
                raise ValueError(f"Value for key {k} must be a dictionary.")
            if not all(key in v for key in ["energy", "input_path", "struct_tag"]):
                raise ValueError(f"Missing required keys in value for {k}.")

            # Convert lists to np.ndarray if necessary
            for key in ["energy", "input_path", "struct_tag"]:
                if isinstance(v[key], list):
                    v[key] = np.array(v[key])
                elif not isinstance(v[key], np.ndarray):
                    raise ValueError(
                        f"Value of '{key}' for key {k} must be a list or ndarray."
                    )

        self._composition_data = data

    def run_analysis(self):
        """Compute formation energies, build convex hull, and evaluate energies above hull."""
        if len(self.elements) == 0:
            raise ValueError("elements is none. Please set elements.")
        if not self._composition_data:
            raise ValueError("composition_data is empty. Please set results first.")
        self.compute_formation_energies()
        self.compute_convex_hull()
        self.compute_fe_above_ch()

    def compute_formation_energies(self):
        """Compute formation energy for each structure relative to pure elements."""
        # Sort compositions for reproducible output
        comps = np.array(list(self._composition_data.keys()))
        order = np.lexsort(comps.T)
        sorted_keys = [tuple(comps[i].tolist()) for i in order]
        self._composition_data = {k: self._composition_data[k] for k in sorted_keys}

        # Reference energies from pure elements (where any ratio == 1)
        pure = comps[np.any(comps == 1, axis=1)]
        pure_sorted = sorted(pure, key=lambda x: np.argmax(x))
        ref_energies = [
            np.min(self._composition_data[tuple(r)]["energy"]) for r in pure_sorted
        ]
        ref_energies = np.array(ref_energies)

        # Calculate and store formation energies
        for ratio, data in self._composition_data.items():
            formation = data["energy"] - np.dot(ref_energies, np.array(ratio))
            data["formation_e"] = formation

        # Save JSON and elements
        os.makedirs("phase_analysis/data", exist_ok=True)
        serial = {
            str(r): {k: v.tolist() for k, v in d.items()}
            for r, d in self._composition_data.items()
        }
        with open("phase_analysis/data/composition_data.json", "w") as f:
            json.dump(serial, f)
        np.save("phase_analysis/data/elements.npy", self.elements)

    def compute_convex_hull(self):
        """Build the convex hull and extract vertex compositions and energies."""
        comps, min_e, paths = [], [], []
        for ratio, data in self._composition_data.items():
            comps.append(ratio)
            idx = np.argmin(data["formation_e"])
            min_e.append(data["formation_e"][idx])
            paths.append(data["input_path"][idx])

        comp_arr = np.array(comps)
        e_arr = np.array(min_e).reshape(-1, 1)
        points = np.hstack([comp_arr[:, 1:], e_arr])
        self.hull = ConvexHull(points)

        verts = np.unique(self.hull.simplices)
        e_hull = e_arr[verts].flatten()
        mask = e_hull <= 1e-10
        self.hull_energies = e_hull[mask]
        self.hull_compositions = comp_arr[verts][mask]
        self.hull_paths = np.array(paths)[verts][mask]

        # Write global minima YAML
        with open("phase_analysis/global_minima.yaml", "w") as f:
            f.write("global_minima:\n")
            for comp, path, energy in zip(
                self.hull_compositions, self.hull_paths, self.hull_energies
            ):
                f.write(f"  - composition: {comp.tolist()}\n")
                f.write(f"    structure:   {path}\n")
                f.write(f"    formation_e: {energy}\n")

        # Save NumPy arrays
        os.makedirs("phase_analysis/data", exist_ok=True)
        np.save("phase_analysis/data/hull_e.npy", self.hull_energies)
        np.save("phase_analysis/data/hull_comp.npy", self.hull_compositions)

    def compute_fe_above_ch(self):
        """Compute each structure's energy above the convex hull."""
        for ratio, data in self._composition_data.items():
            data["fe_above_ch"] = data["formation_e"] - self._evaluate_hull(ratio)

    def _evaluate_hull(self, ratio):
        """Evaluate the hull plane to get reference energy at a given ratio."""
        ratio = np.array(ratio)
        if np.any(ratio == 1):
            return 0.0
        best = -np.inf
        for eq in self.hull.equations:
            num = -(np.dot(eq[:-2], ratio[1:]) + eq[-1])
            val = num / eq[-2]
            if best < val < -1e-8:
                best = val
        return best

    def get_structures_near_hull(self, threshold_meV):
        """Return structures within threshold (meV/atom) above hull."""
        near, far = {}, {}
        for ratio, data in self._composition_data.items():
            mask = data["fe_above_ch"] < (threshold_meV / 1000)
            near[ratio] = {
                "struct_tag": data["struct_tag"][mask],
                "input_path": data["input_path"][mask],
                "fe_above_ch": data["fe_above_ch"][mask],
                "formation_e": data["formation_e"][mask],
            }
            far[ratio] = {
                "struct_tag": data["struct_tag"][~mask],
                "input_path": data["input_path"][~mask],
                "fe_above_ch": data["fe_above_ch"][~mask],
                "formation_e": data["formation_e"][~mask],
            }

        element_count = 0
        multi_count = 0
        for key, res in near.items():
            if len(res["formation_e"]) == 0:
                continue
            if np.any(np.array(key) == 1):
                element_count += len(res["formation_e"])
            else:
                multi_count += len(res["formation_e"])

        thre = float(threshold_meV)
        os.makedirs(f"phase_analysis/threshold_{thre}meV", exist_ok=True)
        with open(f"phase_analysis/threshold_{thre}meV/struct_cands.yaml", "w") as f:
            print("summary:", file=f)
            print(f"  threshold_meV_per_atom: {thre}", file=f)
            print(f"  n_structs_single:       {element_count}", file=f)
            print(f"  n_structs_multi:        {multi_count}", file=f)
            print("", file=f)

            print("near_ch_structures:", file=f)
            for key, res in near.items():
                if len(res["formation_e"]) == 0:
                    continue
                print(f"  - composition: {list(key)}", file=f)
                print("    structures:", file=f)
                for i in range(len(res["formation_e"])):
                    print(f"      - struct_tag: {res['struct_tag'][i]}", file=f)
                    print(f"        input_path: {res['input_path'][i]}", file=f)
                    print(f"        delta_F_ch: {res['fe_above_ch'][i]:.6f}", file=f)
                    print(f"        F_value:    {res['formation_e'][i]:.15f}", file=f)

        far_serial = {
            str(r): {k: v.tolist() for k, v in d.items()} for r, d in far.items()
        }
        near_serial = {
            str(r): {k: v.tolist() for k, v in d.items()} for r, d in near.items()
        }
        with open(f"phase_analysis/threshold_{thre}meV/not_near_ch.json", "w") as f:
            json.dump(far_serial, f)
        with open(f"phase_analysis/threshold_{thre}meV/near_ch.json", "w") as f:
            json.dump(near_serial, f)

        return near, far

    def parse_results(self, input_paths, ghost_minima_file=None, parse_vasp=False):
        """
        Populate composition_data by parsing RSS JSON or VASP vasprun.xml outputs.
        :param input_paths: List of RSS JSON files or VASP directories.
        :param ghost_minima_file: YAML file listing ghost minima to exclude (RSS only).
        :param parse_vasp: If True, parse VASP outputs; otherwise parse RSS JSON.
        """
        if parse_vasp:
            self._parse_vasp_results(input_paths)
        else:
            self._parse_rss_results(input_paths, ghost_minima_file)

    def _parse_rss_results(self, input_paths, ghost_minima_file=None):
        """Internal: parse RSS JSON and filter ghost minima."""
        is_not_ghost_minima = []
        if ghost_minima_file is not None:
            with open(ghost_minima_file) as f:
                ghost_minima_data = yaml.safe_load(f)
            if ghost_minima_data["ghost_minima"] is not None:
                for entry in ghost_minima_data["ghost_minima"]:
                    if entry.get("assessment") == "Not a ghost minimum":
                        is_not_ghost_minima.append(str(entry["structure"]))
        is_not_ghost_minima_set = set(is_not_ghost_minima)

        n_changed = 0
        for res_path in input_paths:
            with open(res_path) as f:
                loaded_dict = json.load(f)

            target_elements = loaded_dict["elements"]
            comp_ratio = loaded_dict["comp_ratio"]
            element_to_ratio = dict(zip(target_elements, comp_ratio))
            comp_ratio_orderd = tuple(
                element_to_ratio.get(el, 0) for el in self.elements
            )
            comp_ratio_array = tuple(
                np.round(
                    np.array(comp_ratio_orderd) / sum(comp_ratio_orderd), 10
                ).tolist()
            )

            logname = os.path.basename(res_path).split(".")[0]
            rss_results = loaded_dict["rss_results"]
            rss_results_array = {
                "energy": np.array([r["energy"] for r in rss_results]),
                "input_path": np.array([r["poscar"] for r in rss_results]),
                "is_ghost_minima": np.array(
                    [r["is_ghost_minima"] for r in rss_results]
                ),
                "struct_tag": np.array(
                    [f"POSCAR_{logname}_No{r['struct_no']}" for r in rss_results]
                ),
            }

            for i in range(len(rss_results_array["is_ghost_minima"])):
                name = rss_results_array["struct_tag"][i]
                if (
                    rss_results_array["is_ghost_minima"][i]
                    and name in is_not_ghost_minima_set
                ):
                    rss_results_array["is_ghost_minima"][i] = False
                    n_changed += 1

            rss_results_valid = {
                key: rss_results_array[key][~rss_results_array["is_ghost_minima"]]
                for key in rss_results_array
            }
            self._composition_data[comp_ratio_array] = rss_results_valid

    def _parse_vasp_results(self, input_paths):
        """Internal: parse vasprun.xml files and map to composition_data."""
        dft_dict = defaultdict(list)
        for dft_path in input_paths:
            vasprun_path = f"{dft_path}/vasprun.xml"
            try:
                vaspobj = Vasprun(vasprun_path)
            except Exception:
                print(vasprun_path, "failed")
                continue

            energy_dft = vaspobj.energy
            structure = vaspobj.structure
            for element in structure.elements:
                energy_dft -= atomic_energy(element)
            energy_dft /= len(structure.elements)

            comp_res = compute_composition(structure.elements, self.elements)
            comp_ratio = tuple(
                np.round(
                    np.array(comp_res.comp_ratio) / sum(comp_res.comp_ratio), 10
                ).tolist()
            )
            dft_dict[comp_ratio].append(
                {
                    "energy": energy_dft,
                    "input_path": vasprun_path,
                    "struct_tag": dft_path.split("/")[-1],
                }
            )

        dft_dict_array = {}
        for comp_ratio, entries in dft_dict.items():
            sorted_entries = sorted(entries, key=lambda x: x["energy"])
            dft_dict_array[comp_ratio] = {
                "energy": np.array([entry["energy"] for entry in sorted_entries]),
                "input_path": np.array(
                    [entry["input_path"] for entry in sorted_entries]
                ),
                "struct_tag": np.array(
                    [entry["struct_tag"] for entry in sorted_entries]
                ),
            }

        self._composition_data = dft_dict_array

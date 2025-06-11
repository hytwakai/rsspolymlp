import argparse
import json
import os

import numpy as np
import yaml
from scipy.spatial import ConvexHull


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        nargs="*",
        type=str,
        required=True,
        help="phase_analysisemical elements, e.g., La Bi",
    )
    parser.add_argument(
        "--result_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to RSS result log files",
    )
    parser.add_argument(
        "--outlier_file",
        type=str,
        default=None,
        help="Path to file listing outlier structure names to be excluded",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Threshold values for energy above the convex hull in meV/atom ",
    )
    args = parser.parse_args()

    ch_analyzer = ConvexHullAnalyzer(
        args.elements, args.result_paths, args.outlier_file
    )
    ch_analyzer.run_calc()

    if args.thresholds is not None:
        threshold_list = args.thresholds
        for threshold in threshold_list:
            ch_analyzer.get_struct_near_ch(threshold)


class ConvexHullAnalyzer:

    def __init__(self, elements, result_paths, outlier_file=None):

        self.elements = elements
        self.result_paths = result_paths
        self.outlier_file = outlier_file
        self.rss_result_fe = {}
        self.ch_obj = None
        self.fe_ch = None
        self.comp_ch = None
        self.poscar_ch = None
        os.makedirs("phase_analysis/data", exist_ok=True)

    def run_calc(self):
        self.calc_formation_e()
        self.calc_convex_hull()
        self.calc_fe_above_convex_hull()

    def calc_formation_e(self):
        is_not_outliers = []
        if self.outlier_file is not None:
            with open(self.outlier_file) as f:
                outlier_data = yaml.safe_load(f)
            for entry in outlier_data["outliers"]:
                if entry.get("assessment") == "Not an outlier":
                    is_not_outliers.append(str(entry["structure"]))
        is_not_outliers_set = set(is_not_outliers)

        n_changed = 0

        for res_path in self.result_paths:
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

            rss_results = loaded_dict["rss_results"]
            rss_results_valid = [r for r in rss_results if not r["is_strong_outlier"]]
            rss_results_array = {
                "formation_e": np.array([r["energy"] for r in rss_results_valid]),
                "poscars": np.array([r["poscar"] for r in rss_results_valid]),
                "is_outliers": np.array(
                    [r["is_weak_outlier"] for r in rss_results_valid]
                ),
                "struct_no": np.array([r["struct_no"] for r in rss_results_valid]),
            }

            logname = os.path.basename(res_path).split(".")[0]
            for i in range(len(rss_results_array["is_outliers"])):
                struct_no = rss_results_array["struct_no"][i]
                name = f"POSCAR_{logname}_No{struct_no}"
                if rss_results_array["is_outliers"][i] and name in is_not_outliers_set:
                    rss_results_array["is_outliers"][i] = False
                    n_changed += 1
            rss_results_array = {
                key: rss_results_array[key][~rss_results_array["is_outliers"]]
                for key in rss_results_array
            }

            self.rss_result_fe[comp_ratio_array] = rss_results_array

        comps = np.array(list(self.rss_result_fe.keys()))
        sort_idx = np.lexsort(comps.T)
        sorted_keys = [list(self.rss_result_fe.keys())[i] for i in sort_idx]
        self.rss_result_fe = {key: self.rss_result_fe[key] for key in sorted_keys}

        e_ends = []
        keys = np.array(list(self.rss_result_fe))
        valid_keys = keys[np.any(keys == 1, axis=1)]
        sorted_keys = sorted(valid_keys, key=lambda x: np.argmax(x))
        for key in sorted_keys:
            key_tuple = tuple(key)
            energy = self.rss_result_fe[key_tuple]["formation_e"][0]
            e_ends.append(energy)
        e_ends = np.array(e_ends)

        for key in self.rss_result_fe:
            self.rss_result_fe[key]["formation_e"] -= np.dot(e_ends, np.array(key))

        rss_result_fe_serial = self._convert_ndarray_to_json(self.rss_result_fe)
        with open("phase_analysis/data/rss_result_fe.json", "w") as f:
            json.dump(rss_result_fe_serial, f)

    def calc_convex_hull(self):
        rss_result_fe = self.rss_result_fe

        comp_list, e_min_list, label_list = [], [], []
        for key, dicts in rss_result_fe.items():
            comp_list.append(key)
            e_min_list.append(dicts["formation_e"][0])
            label_list.append(dicts["poscars"][0])

        comp_array = np.array(comp_list)
        e_min_array = np.array(e_min_list).reshape(-1, 1)
        label_array = np.array(label_list)

        data_ch = np.hstack([comp_array[:, 1:], e_min_array])
        self.ch_obj = ConvexHull(data_ch)

        v_convex = np.unique(self.ch_obj.simplices)
        _fe_ch = e_min_array[v_convex].astype(float)
        mask = np.where(_fe_ch <= 1e-10)[0]

        _comp_ch = comp_array[v_convex][mask]
        sort_idx = np.lexsort(_comp_ch.T)

        self.fe_ch = _fe_ch[mask][sort_idx]
        self.comp_ch = _comp_ch[sort_idx]
        self.poscar_ch = label_array[v_convex][mask][sort_idx]

        with open("phase_analysis/global_minima.yaml", "w") as f:
            print("global_minima:", file=f)
            for i in range(len(self.comp_ch)):
                print("  - composition:      ", self.comp_ch[i], file=f)
                print("    structure:        ", self.poscar_ch[i], file=f)
                print("    formation_energy: ", self.fe_ch[i][0], file=f)

        np.save("phase_analysis/data/fe_ch.npy", self.fe_ch)
        np.save("phase_analysis/data/comp_ch.npy", self.comp_ch)

    def calc_fe_above_convex_hull(self):
        rss_result_fe = self.rss_result_fe
        for key in rss_result_fe:
            _ehull = self._calc_fe_convex_hull(key)
            fe_above_ch = rss_result_fe[key]["formation_e"] - _ehull
            rss_result_fe[key]["fe_above_ch"] = fe_above_ch

    def _calc_fe_convex_hull(self, comp_ratio):
        if np.any(np.array(comp_ratio) == 1):
            return 0

        ehull = -1e10
        for eq in self.ch_obj.equations:
            face_val_comp = -(np.dot(eq[:-2], comp_ratio[1:]) + eq[-1])
            ehull_trial = face_val_comp / eq[-2]
            if ehull_trial > ehull and ehull_trial < -1e-8:
                ehull = ehull_trial

        return ehull

    def get_struct_near_ch(self, threshold):
        near_ch = {}
        not_near_ch = {}

        rss_result_fe = self.rss_result_fe
        for key in rss_result_fe:
            is_near = rss_result_fe[key]["fe_above_ch"] < threshold / 1000
            near_ch[key] = {"formation_e": None, "poscars": None, "fe_above_ch": None}
            near_ch[key]["formation_e"] = rss_result_fe[key]["formation_e"][is_near]
            near_ch[key]["poscars"] = rss_result_fe[key]["poscars"][is_near]
            near_ch[key]["fe_above_ch"] = rss_result_fe[key]["fe_above_ch"][is_near]

            is_not_near = rss_result_fe[key]["fe_above_ch"] >= threshold / 1000
            not_near_ch[key] = {
                "formation_e": None,
                "poscars": None,
                "fe_above_ch": None,
            }
            not_near_ch[key]["formation_e"] = rss_result_fe[key]["formation_e"][
                is_not_near
            ]
            not_near_ch[key]["poscars"] = rss_result_fe[key]["poscars"][is_not_near]
            not_near_ch[key]["fe_above_ch"] = rss_result_fe[key]["fe_above_ch"][
                is_not_near
            ]

        element_count = 0
        multi_count = 0
        for key, res in near_ch.items():
            if len(res["formation_e"]) == 0:
                continue
            if np.any(np.array(key) == 1):
                element_count += len(res["formation_e"])
            else:
                multi_count += len(res["formation_e"])

        os.makedirs(f"phase_analysis/threshold_{threshold}meV", exist_ok=True)
        with open(
            f"phase_analysis/threshold_{threshold}meV/struct_cands.yaml", "w"
        ) as f:
            print("summary:", file=f)
            print(f"  threshold_meV_per_atom: {threshold}", file=f)
            print(f"  n_structs_single:       {element_count}", file=f)
            print(f"  n_structs_multi:        {multi_count}", file=f)
            print("", file=f)

            print("near_ch_structures:", file=f)
            for key, res in near_ch.items():
                if len(res["formation_e"]) == 0:
                    continue
                print(f"  - composition: {key}", file=f)
                print("    structures:", file=f)
                for i in range(len(res["formation_e"])):
                    print(f"      - poscar:     {res['poscars'][i]}", file=f)
                    print(
                        f"        delta_F_ch: {res['fe_above_ch'][i]:.6f}",
                        file=f,
                    )
                    print(
                        f"        F_value:    {res['formation_e'][i]:.15f}",
                        file=f,
                    )

        not_near_ch = self._convert_ndarray_to_json(not_near_ch)
        near_ch = self._convert_ndarray_to_json(near_ch)
        with open(
            f"phase_analysis/threshold_{threshold}meV/not_near_ch.json", "w"
        ) as f:
            json.dump(not_near_ch, f)
        with open(f"phase_analysis/threshold_{threshold}meV/near_ch.json", "w") as f:
            json.dump(near_ch, f)

    def _convert_ndarray_to_json(self, data):
        converted = {
            str(k): {key: val.tolist() for key, val in v.items()}
            for k, v in data.items()
        }
        return converted

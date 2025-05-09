import numpy as np
from scipy.spatial import ConvexHull

from rsspolymlp.analysis.rss_summarize import (
    extract_composition_ratio,
    load_rss_results,
)


class ConvexHullAnalyzer:

    def __init__(self, elements, result_paths):

        self.elements = elements
        self.result_paths = result_paths
        self.rss_result_fe = {}
        self.ch_obj = None
        self.fe_ch = None
        self.comp_ch = None
        self.poscar_ch = None

    def run_calc(self):
        self.calc_formation_e()
        self.calc_convex_hull()

    def calc_formation_e(self):
        for res_path in self.result_paths:
            comp_ratio = extract_composition_ratio(res_path, self.elements)
            comp_ratio = tuple(np.round(np.array(comp_ratio) / sum(comp_ratio), 10))

            rss_results = load_rss_results(
                res_path, absolute_path=True, get_warning=True
            )
            rss_results_array = {
                "energies": np.array([r["enthalpy"] for r in rss_results]),
                "poscars": np.array([r["poscar"] for r in rss_results]),
                "is_outliers": np.array([r["outlier"] for r in rss_results]),
            }
            self.rss_result_fe[comp_ratio] = rss_results_array

        e_ends = []
        keys = np.array(list(self.rss_result_fe))
        valid_keys = keys[np.any(keys == 1, axis=1)]
        sorted_keys = sorted(valid_keys, key=lambda x: np.argmax(x))
        for key in sorted_keys:
            key_tuple = tuple(key)
            is_outlier = self.rss_result_fe[key_tuple]["is_outliers"]
            first_valid_index = np.where(~is_outlier)[0][0]
            energy = self.rss_result_fe[key_tuple]["energies"][first_valid_index]
            e_ends.append(energy)
        e_ends = np.array(e_ends)

        for key in self.rss_result_fe:
            self.rss_result_fe[key]["energies"] -= np.dot(e_ends, np.array(key))

    def calc_convex_hull(self):
        rss_result_fe = self.rss_result_fe

        comp_list, e_min_list, label_list = [], [], []
        for key, dicts in rss_result_fe.items():
            comp_list.append(key)
            first_idx = np.where(~dicts["is_outliers"])[0][0]
            e_min_list.append(dicts["energies"][first_idx])
            label_list.append(dicts["poscars"][first_idx])

        comp_array = np.array(comp_list)
        e_min_array = np.array(e_min_list).reshape(-1, 1)
        label_array = np.array(label_list)

        data_ch = np.hstack([comp_array[:, 1:], e_min_array])
        self.ch_obj = ConvexHull(data_ch)

        v_convex = np.unique(self.ch_obj.simplices)
        _fe_ch = e_min_array[v_convex].astype(float)
        mask = np.where(_fe_ch <= 1e-10)[0]

        _comp_ch = comp_array[v_convex][mask]
        sort_idx = np.lexsort(_comp_ch[mask][:, ::-1].T)

        self.fe_ch = _fe_ch[mask][sort_idx]
        self.comp_ch = _comp_ch[sort_idx]
        self.poscar_ch = label_array[v_convex][mask][sort_idx]

    def calc_fe_convex_hull(self, comp):
        ehull = -1e10
        for eq in self.ch_obj.equations:
            face_val_comp = -(np.dot(eq[:-2], comp[1:]) + eq[-1])
            ehull_trial = face_val_comp / eq[-2]
            if ehull_trial > ehull and abs(ehull_trial) > 1e-8:
                ehull = ehull_trial

        return ehull

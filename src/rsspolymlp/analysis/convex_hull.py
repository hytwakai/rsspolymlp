import numpy as np
from scipy.spatial import ConvexHull, Delaunay

from rsspolymlp.analysis.rss_summarize import (
    extract_composition_ratio,
    load_rss_results,
)
from rsspolymlp.common.parse_arg import ParseArgument


def run():
    args = ParseArgument.get_summarize_args()
    elements = args.elements
    result_paths = args.result_paths
    ch_analyzer = ConvexHullAnalyzer(elements, result_paths)
    ch_analyzer.run_calc()


class ConvexHullAnalyzer:

    def __init__(self, elements, result_paths):

        self.elements = elements
        self.result_paths = result_paths
        self.rss_result_fe = {}

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
            rss_results_sorted = sorted(rss_results, key=lambda r: r["enthalpy"])
            rss_results_array = {
                "energies": np.array([r["enthalpy"] for r in rss_results_sorted]),
                "poscars": np.array([r["poscar"] for r in rss_results_sorted]),
                "is_outliers": np.array([r["outlier"] for r in rss_results_sorted]),
            }
            self.rss_result_fe[comp_ratio] = rss_results_array

        e_ends = []
        keys = np.array(list(self.rss_result_fe))
        valid_keys = keys[np.any(keys == 1, axis=1)]
        for key in valid_keys:
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

        comp_list = []
        e_min_list = []
        label_list = []
        for key, dicts in rss_result_fe.items():
            comp_list.append(key)
            first_idx = np.where(~dicts["is_outliers"])[0][0]
            e_min_list.append(dicts["energies"][first_idx])
            label_list.append(dicts["poscars"][first_idx])
        comp_array = np.array(comp_list)
        e_min_array = np.array(e_min_list).reshape(-1, 1)
        data_ch, v_convex = self.convex_hull(comp_array, e_min_array)
        print(data_ch)
        print(comp_array[v_convex])

    def convex_hull(self, comp, min_f_e):
        data = np.hstack([comp[:, :-1], min_f_e])
        hull = ConvexHull(data)

        v_convex = np.unique(hull.simplices)
        data_ch = min_f_e[v_convex].astype(float)
        lower_convex = np.where(data_ch <= 1e-15)[0]
        data_ch = data_ch[lower_convex]

        return data_ch, v_convex[lower_convex]

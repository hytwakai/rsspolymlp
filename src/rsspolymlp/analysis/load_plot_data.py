import ast
import json

import numpy as np


def load_plot_data(path="./phase_analysis", threshold=None):
    res = {}

    res["elements"] = np.load(f"{path}/data/elements.npy")
    res["hull_comp"] = np.load(f"{path}/data/hull_comp.npy")
    res["hull_e"] = np.load(f"{path}/data/hull_e.npy")
    with open(f"{path}/data/composition_data.json", "r") as f:
        data = json.load(f)
    res["composition_data"] = convert_json_to_ndarray(data)

    res["not_near_ch"] = None
    res["near_ch"] = None
    if threshold is not None:
        thre = float(threshold)
        with open(f"{path}/threshold_{thre}meV/not_near_ch.json", "r") as f:
            data1 = json.load(f)
        with open(f"{path}/threshold_{thre}meV/near_ch.json", "r") as f:
            data2 = json.load(f)
        res["not_near_ch"] = convert_json_to_ndarray(data1)
        res["near_ch"] = convert_json_to_ndarray(data2)

    return res


def convert_json_to_ndarray(data):
    converted = {
        ast.literal_eval(k): {key: np.array(val) for key, val in v.items()}
        for k, v in data.items()
    }
    return converted

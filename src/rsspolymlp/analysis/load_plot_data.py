import ast
import json

import numpy as np


def load_plot_data(path="./phase_analysis", threshold=None):
    res = {}

    res["comp_ch"] = np.load(f"{path}/data/comp_ch.npy")
    res["fe_ch"] = np.load(f"{path}/data/fe_ch.npy")
    with open(f"{path}/data/rss_result_fe.json", "r") as f:
        data = json.load(f)
    res["rss_result_fe"] = convert_json_to_ndarray(data)

    res["not_near_ch"] = None
    res["near_ch"] = None
    if threshold is not None:
        with open(f"{path}/threshold_{threshold}meV/not_near_ch.json", "r") as f:
            data1 = json.load(f)
        with open(f"{path}/threshold_{threshold}meV/near_ch.json", "r") as f:
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

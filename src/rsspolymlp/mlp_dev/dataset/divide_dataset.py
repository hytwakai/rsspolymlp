import argparse
import glob
import math
import os
import random
import shutil

import numpy as np

from pypolymlp.core.interface_vasp import Vasprun
from rsspolymlp.common.atomic_energy import atomic_energy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="Directory path containing vasprun.xml files.",
)
parser.add_argument(
    "--threshold_close_minima",
    type=float,
    default=1.0,
    help="Force threshold for filtering structures classified as close_minima.",
)
args = parser.parse_args()

vasprun_paths = sorted(glob.glob(args.path + "/*"))

vasprun_dict = {"ws_large_force": [], "wo_force": [], "close_minima": [], "normal": []}
for vasprun_path in vasprun_paths:
    try:
        dft_dict = Vasprun(vasprun_path)
    except ValueError:
        continue

    energy = dft_dict.energy
    force = dft_dict.forces
    elements = dft_dict.structure.elements
    for a_t in elements:
        energy = energy - atomic_energy(a_t)

    # dataset containing structures with extremely large forces
    if energy / len(elements) > 10:
        vasprun_dict["wo_force"].append(vasprun_path)
        continue
    elif np.any(np.abs(force) >= 100):
        if energy / len(elements) < -2:
            continue
        else:
            vasprun_dict["wo_force"].append(vasprun_path)
            continue

    # dataset containing structures with some large forces
    if np.any(np.abs(force) >= 10):
        vasprun_dict["ws_large_force"].append(vasprun_path)
        continue

    if np.all(np.abs(force) <= args.threshold_close_minima):
        vasprun_dict["close_minima"].append(vasprun_path)
        continue

    vasprun_dict["normal"].append(vasprun_path)

with open("dataset.yaml", "w") as f:
    pass
output_dir = "dft_dataset_divided"
os.makedirs(output_dir, exist_ok=True)

ratio = 0.1
for data_name, vasprun_list in vasprun_dict.items():
    if len(vasprun_list) == 0:
        continue
    random.shuffle(vasprun_list)
    split_index = math.floor(len(vasprun_list) * ratio)

    train_data = sorted(vasprun_list[split_index:])
    test_data = sorted(vasprun_list[:split_index])

    print(data_name)
    print("  - train_data:", len(train_data))
    print("  - test_data:", len(test_data))

    with open(f"{output_dir}/dataset.yaml", "a") as f:
        print("arguments:", file=f)
        print("  path:", args.path, file=f)
        print("  threshold_close_minima:", args.threshold_close_minima, file=f)
        print("", file=f)
        print(f"{data_name}:", file=f)
        print("  train:", file=f)
        for p in train_data:
            print(f"    - {p}", file=f)
        print("  test:", file=f)
        for p in test_data:
            print(f"    - {p}", file=f)

    os.makedirs(f"{output_dir}/train/{data_name}")
    for p in train_data:
        shutil.copy(p, f"{output_dir}/train/{data_name}")

    if len(test_data) > 0:
        os.makedirs(f"{output_dir}/test/{data_name}")
        for p in test_data:
            shutil.copy(p, f"{output_dir}/test/{data_name}")

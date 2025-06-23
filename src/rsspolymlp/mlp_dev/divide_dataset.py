import argparse
import math
import os
import random
import shutil

import numpy as np

from pypolymlp.core.interface_vasp import parse_vaspruns
from rsspolymlp.common.atomic_energy import atomic_energy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    nargs="+",
    required=True,
    help="Directory paths containing vasprun.xml files.",
)
args = parser.parse_args()

vasprun_paths = args.path

vasprun_dict = {"ws_large_force": [], "wo_force": [], "close_minima": [], "normal": []}
for vasprun_path in vasprun_paths:
    try:
        dft_dict = parse_vaspruns([vasprun_path])
    except ValueError:
        continue
    elements = dft_dict.structures[0].elements
    energy = dft_dict.energies[0]
    for a_t in elements:
        energy = energy - atomic_energy(a_t)
    force = dft_dict.forces

    # dataset containing structures with some large forces
    if np.any(np.abs(force) >= 10):
        vasprun_dict["ws_large_force"].append(vasprun_path)
        continue

    # dataset containing structures with extremely large forces
    if energy / dft_dict.total_n_atoms[0] > 10:
        vasprun_dict["wo_force"].append(vasprun_path)
        continue
    elif np.any(np.abs(force) >= 100):
        if energy / dft_dict.total_n_atoms[0] < -2:
            continue
        else:
            vasprun_dict["wo_force"].append(vasprun_path)
            continue

    if np.all(np.abs(force) <= 1):
        vasprun_dict["close_minima"].append(vasprun_path)
        continue

    vasprun_dict["normal"].append(vasprun_path)

with open("dataset.yaml", "w") as f:
    pass
ratio = 0.1

for data_name, vasprun_list in vasprun_dict.items():
    if len(vasprun_list) == 0:
        continue
    random.shuffle(vasprun_list)
    split_index = math.floor(len(vasprun_list) * ratio)

    train_data = sorted(vasprun_list[split_index:])
    test_data = sorted(vasprun_list[:split_index])

    print("train =", len(train_data))
    print("test =", len(test_data))

    with open("dataset.yaml", "a") as f:
        print(data_name, file=f)
        print("  train:")
        for p in train_data:
            print(f"    - {p}", file=f)
        print("  test:")
        for p in test_data:
            print(f"    - {p}", file=f)

    os.makedirs(f"train/{data_name}")
    for p in train_data:
        shutil.copy(p, f"train/{data_name}")

    if len(test_data) > 0:
        os.makedirs(f"test/{data_name}")
        for p in test_data:
            shutil.copy(p, f"test/{data_name}")

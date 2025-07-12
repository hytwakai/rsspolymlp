import glob
import math
import os
import random
import shutil

import numpy as np

from pypolymlp.core.interface_vasp import Vasprun
from rsspolymlp.common.atomic_energy import atomic_energy


def divide_dataset(vasprun_paths: list[str], threshold_close_minima=1.0):
    vasprun_dict = {"large_force": [], "wo_force": [], "close_minima": [], "normal": []}

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
            vasprun_dict["large_force"].append(vasprun_path)
            continue

        if np.all(np.abs(force) <= threshold_close_minima):
            vasprun_dict["close_minima"].append(vasprun_path)
            continue

        vasprun_dict["normal"].append(vasprun_path)

    return vasprun_dict


def divide_train_test(data_name, vasprun_list, divide_ratio=0.1):
    random.shuffle(vasprun_list)
    split_index = math.floor(len(vasprun_list) * divide_ratio)

    train_data = sorted(vasprun_list[split_index:])
    test_data = sorted(vasprun_list[:split_index])

    os.makedirs(f"train/{data_name}")
    for p in train_data:
        shutil.copy(p, f"train/{data_name}")

    if len(test_data) > 0:
        os.makedirs(f"test/{data_name}")
        for p in test_data:
            shutil.copy(p, f"test/{data_name}")

    return train_data, test_data


if __name__ == "__main__":

    import argparse

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

    target_dir = args.path
    threshold = args.threshold_close_minima

    vasprun_paths = sorted(glob.glob(target_dir + "/*"))

    vasprun_dict = divide_dataset(
        vasprun_paths=vasprun_paths, threshold_close_minima=threshold
    )

    with open("dataset.yaml", "w") as f:
        print("arguments:", file=f)
        print("  path:", target_dir, file=f)
        print("  threshold_close_minima:", threshold, file=f)
        print("", file=f)

    output_dir = "dft_dataset_divided"
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    divide_ratio = 0.1
    for data_name, vasprun_list in vasprun_dict.items():
        if len(vasprun_list) == 0:
            continue

        train_data, test_data = divide_train_test(
            data_name=data_name, vasprun_list=vasprun_list, divide_ratio=divide_ratio
        )
        print(data_name)
        print("  - train_data:", len(train_data))
        print("  - test_data:", len(test_data))

        with open("dataset.yaml", "a") as f:
            print(f"{data_name}:", file=f)
            print("  train:", file=f)
            for p in train_data:
                print(f"    - {p}", file=f)
            print("  test:", file=f)
            for p in test_data:
                print(f"    - {p}", file=f)

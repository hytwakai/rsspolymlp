import math
import os
import random
import shutil

import numpy as np

from pypolymlp.core.interface_vasp import Vasprun
from rsspolymlp.common.atomic_energy import atomic_energy


def divide_dataset(
    vasprun_paths: list[str],
    threshold_vlarge_s: float = 300.0,  # in GPa
    threshold_vlarge_f: float = 100.0,  # in eV/ang
    threshold_large_f: float = 10.0,
    threshold_close_minima: float = 1.0,
):
    """
    Classify VASP calculation results into dataset categories based on force and stress magnitudes.

    Returns:
        A dictionary categorizing paths into:
            - "stress-very-large"
            - "force-very-large"
            - "force-large"
            - "minima-close"
            - "force-normal"
    """
    vasprun_dict = {
        "stress-very-large": [],
        "force-very-large": [],
        "force-large": [],
        "minima-close": [],
        "force-normal": [],
    }

    for vasprun_path in vasprun_paths:
        try:
            dft_dict = Vasprun(vasprun_path)
        except ValueError:
            continue

        energy = dft_dict.energy
        force = dft_dict.forces
        stress = dft_dict.stress
        elements = dft_dict.structure.elements

        for elem in elements:
            energy -= atomic_energy(elem)
        energy_per_atom = energy / len(elements)

        # Structures with extremely large stress components
        if np.any(np.abs(stress) > threshold_vlarge_s * 10):
            if energy_per_atom > -5:
                vasprun_dict["stress-very-large"].append(vasprun_path)
            continue

        # Structures with extremely large force components
        if np.any(np.abs(force) >= threshold_vlarge_f):
            if energy_per_atom > -5:
                vasprun_dict["force-very-large"].append(vasprun_path)
            continue

        # Structures with moderately large forces
        if np.any(np.abs(force) >= threshold_large_f):
            vasprun_dict["force-large"].append(vasprun_path)
            continue

        # Structures with only small forces (close to local minima)
        if np.all(np.abs(force) <= threshold_close_minima):
            vasprun_dict["minima-close"].append(vasprun_path)
            continue

        # Structures with typical (normal) force and stress values
        vasprun_dict["force-normal"].append(vasprun_path)

    return vasprun_dict


def divide_train_test(
    data_name, vasprun_list, divide_ratio=0.1, output_dir="dft_dataset"
):
    random.shuffle(vasprun_list)
    split_index = math.floor(len(vasprun_list) * divide_ratio)

    train_data = sorted(vasprun_list[split_index:])
    test_data = sorted(vasprun_list[:split_index])

    os.makedirs(f"{output_dir}/train/{data_name}")
    for p in train_data:
        shutil.copy(p, f"{output_dir}/train/{data_name}")

    if len(test_data) > 0:
        os.makedirs(f"{output_dir}/test/{data_name}")
        for p in test_data:
            shutil.copy(p, f"{output_dir}/test/{data_name}")

    return train_data, test_data

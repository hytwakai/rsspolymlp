import math
import os
import random
import shutil
from typing import Optional

import numpy as np

from pypolymlp.core.interface_vasp import Vasprun
from pypolymlp.core.units import EVtoGPa
from rsspolymlp.common.atomic_energy import atomic_energy


def divide_dataset(
    vasprun_paths: list[str],
    threshold_e_high: float = 10.0,  # in eV/atom
    threshold_e_low: Optional[float] = None,
    threshold_f_small: float = 3.0,  # in eV/ang
    threshold_f_normal: float = 10.0,
    threshold_f_large: float = 100.0,
    threshold_s_large: float = 200.0,  # in GPa
    threshold_s_small: Optional[float] = None,
    include_stress: bool = False,
):
    """
    Classify VASP calculation results into dataset categories based on
    magnitudes of the energy, force and stress tensor components.

    Returns:
        A dictionary categorizing paths into:
            - "f_small"
            - "f_normal"
            - "f_large"
            - "f_exlarge"
            - "s_large"
            - "f_small-e_high"
            - "f_normal-e_high"
            - "f_large-e_high"
            - "f_exlarge-e_high"
            - "s_large-e_high"
    """
    vasprun_dict = {
        "f_small": [],
        "f_normal": [],
        "f_large": [],
        "f_exlarge": [],
        "s_large": [],
        "f_small-e_high": [],
        "f_normal-e_high": [],
        "f_large-e_high": [],
        "f_exlarge-e_high": [],
        "s_large-e_high": [],
    }

    for vasprun_path in vasprun_paths:
        try:
            dft_dict = Vasprun(vasprun_path)
        except ValueError:
            continue

        energy = dft_dict.energy
        force = dft_dict.forces
        elements = dft_dict.structure.elements
        for elem in elements:
            energy -= atomic_energy(elem)
        energy_per_atom = energy / len(elements)
        cohe_energy = energy_per_atom

        stress = dft_dict.stress
        min_stress = min([stress[0][0], stress[1][1], stress[2][2]])
        max_stress = np.max(np.abs(stress))
        if include_stress:
            vol_per_atom = dft_dict.structure.volume / len(dft_dict.structure.elements)
            pressure = np.mean([(stress / 10).tolist()[i][i] for i in range(3)])
            energy_per_atom += pressure * vol_per_atom / EVtoGPa

        # Filter by energy value
        if energy_per_atom > threshold_e_high or (
            threshold_e_low is not None and energy_per_atom < threshold_e_low
        ):
            e_tag = "-e_high"
        else:
            e_tag = ""

        # Filter by stress tensor components
        if max_stress > threshold_s_large * 10 or (
            threshold_s_small is not None and min_stress < threshold_s_small * 10
        ):
            if cohe_energy > -5:
                vasprun_dict[f"s_large{e_tag}"].append(vasprun_path)
            continue

        # Filter by force components
        if np.all(np.abs(force) <= threshold_f_small):
            vasprun_dict[f"f_small{e_tag}"].append(vasprun_path)
            continue
        if np.all(np.abs(force) <= threshold_f_normal):
            vasprun_dict[f"f_normal{e_tag}"].append(vasprun_path)
            continue
        if np.all(np.abs(force) <= threshold_f_large):
            vasprun_dict[f"f_large{e_tag}"].append(vasprun_path)
            continue
        if cohe_energy > -5:
            vasprun_dict[f"f_exlarge{e_tag}"].append(vasprun_path)

    return vasprun_dict


def divide_train_test(
    data_name, vasprun_list, divide_ratio=0.1, output_dir="dft_dataset"
):
    random.shuffle(vasprun_list)
    split_index = math.floor(len(vasprun_list) * divide_ratio)

    train_data = sorted(vasprun_list[split_index:])
    test_data = sorted(vasprun_list[:split_index])

    try:
        os.makedirs(f"{output_dir}/train/{data_name}")
        for p in train_data:
            shutil.copy(p, f"{output_dir}/train/{data_name}")

        if len(test_data) > 0:
            os.makedirs(f"{output_dir}/test/{data_name}")
            for p in test_data:
                shutil.copy(p, f"{output_dir}/test/{data_name}")
    except FileExistsError:
        print(f"File exists: {os.getcwd()}/{output_dir}, passed")
        pass

    return train_data, test_data

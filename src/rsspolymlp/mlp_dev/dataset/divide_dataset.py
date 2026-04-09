import math
import os
import random
import shutil
from collections import defaultdict
from typing import Optional

import numpy as np

from pypolymlp.core.interface_vasp import Vasprun
from pypolymlp.core.units import EVtoGPa
from rsspolymlp.analysis.phase_analysis import ConvexHullAnalyzer
from rsspolymlp.common.atomic_energy import atomic_energy
from rsspolymlp.common.composition import compute_composition


def parse_vasp_results(elements, vasprun_paths):

    def convert_dict(d):
        result = {}
        for comp_ratio, entries in d.items():
            sorted_entries = sorted(entries, key=lambda x: x["energy"])
            result[comp_ratio] = {
                "energy": np.array([e["energy"] for e in sorted_entries]),
                "force": [e["force"] for e in sorted_entries],
                "stress": [e["stress"] for e in sorted_entries],
                "input_path": np.array([e["input_path"] for e in sorted_entries]),
                "struct_tag": np.array([e["struct_tag"] for e in sorted_entries]),
            }
        return result

    dft_dict = defaultdict(list)
    for vasprun_path in vasprun_paths:
        try:
            vaspobj = Vasprun(vasprun_path)
        except Exception:
            print(vasprun_path, "failed")
            continue

        energy = vaspobj.energy
        force = vaspobj.forces
        stress = vaspobj.stress
        structure = vaspobj.structure
        for element in structure.elements:
            energy -= atomic_energy(element)
        energy /= len(structure.elements)

        if energy < -15:
            print(vasprun_path, "exhibits very low energy:", energy, "eV/atom")
            continue

        pressure = np.mean([(stress / 10).tolist()[i][i] for i in range(3)])  # GPa
        vol_per_atom = structure.volume / len(structure.elements)
        energy += pressure * vol_per_atom / EVtoGPa

        comp_res = compute_composition(structure.elements, elements)
        comp_ratio = tuple(
            np.round(
                np.array(comp_res.comp_ratio) / sum(comp_res.comp_ratio), 10
            ).tolist()
        )

        dft_dict[comp_ratio].append(
            {
                "energy": energy,
                "force": force,
                "stress": stress,
                "input_path": vasprun_path,
                "struct_tag": vasprun_path.split("/")[-1],
            }
        )

    dft_dict_array = convert_dict(dft_dict)

    return dft_dict_array


def divide_dataset(
    elements: list[str],
    vasprun_paths: list[str],
    prototype_paths: list[str],
    threshold_e_high: float = 10.0,  # in eV/atom
    threshold_e_low: Optional[float] = None,
    threshold_f_small: float = 3.0,  # in eV/ang
    threshold_f_normal: float = 10.0,
    threshold_f_large: float = 100.0,
    threshold_s_large: float = 200.0,  # in GPa
    threshold_s_small: Optional[float] = None,
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

    dft_dict_array = parse_vasp_results(elements=elements, vasprun_paths=vasprun_paths)

    ch_analyzer = ConvexHullAnalyzer(elements=elements)
    ch_analyzer.parse_results(input_paths=prototype_paths, parse_vasp=True)
    ch_analyzer.set_endmember_energies()
    ch_analyzer.compute_convex_hull()

    ch_analyzer.composition_data = dft_dict_array
    ch_analyzer.compute_formation_energies(json_output=False)
    ch_analyzer.compute_fe_above_ch()

    for _, data in ch_analyzer.composition_data.items():
        for idx, vasprun_path in enumerate(data["input_path"]):
            fe_above_ch = data["fe_above_ch"][idx]
            force = data["force"][idx]
            stress = data["stress"][idx]

            min_stress = min([stress[0][0], stress[1][1], stress[2][2]])
            max_stress = np.max(np.abs(stress))
            pressure = np.mean([(stress / 10).tolist()[i][i] for i in range(3)])  # GPa

            # Filter by energy value
            if fe_above_ch > threshold_e_high or (
                threshold_e_low is not None
                and pressure > 0
                and fe_above_ch < threshold_e_low
            ):
                e_tag = "-e_high"
            else:
                e_tag = ""

            # Filter by stress tensor components
            if max_stress > threshold_s_large * 10 or (
                threshold_s_small is not None and min_stress < threshold_s_small * 10
            ):
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
        print(f"File exists: {os.getcwd()}/{output_dir}/train/{data_name}, passed")
        pass

    return train_data, test_data

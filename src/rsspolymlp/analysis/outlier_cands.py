import argparse
import ast
import glob
import json
import os
import shutil

import numpy as np

from pypolymlp.core.interface_vasp import Vasprun
from rsspolymlp.utils.ground_state_e import ground_state_energy

EV = 1.602176634e-19  # [J]
EVAngstromToGPa = EV * 1e21


def detect_outlier(energies: np.array):
    """
    Detect outliers and potential outliers in a 1D energy array.

    Returns
    -------
    is_strong_outlier: np.ndarray of bool
        Boolean array marking strong outliers (energy diff > 1.0).
    is_weak_outlier : np.ndarray of bool
        Boolean array marking potential outliers (energy diff > 0.2).
    """
    n = len(energies)
    if n == 1:
        return np.array([False]), np.array([False])

    energy_diffs = np.diff(energies)
    mask = np.abs(energy_diffs) > 1e-6

    group_ids = np.cumsum(mask)
    group_ids = np.concatenate([[0], group_ids])

    unique_groups = np.unique(group_ids)
    group_means = np.array(
        [np.mean(energies[group_ids == gid]) for gid in unique_groups]
    )

    window = int(round(len(energies) * 0.05))
    window = max(window, 10)

    is_strong_group = np.full(group_means.shape, False, dtype=bool)
    is_weak_group = np.full(group_means.shape, False, dtype=bool)
    for i in range(len(group_means) - 1):
        end = min(i + window, len(group_means) - 1)
        diff = abs(group_means[i] - group_means[end])
        if diff > 1.0:
            is_strong_group[i] = True
        elif diff > 0.2:
            is_weak_group[i] = True
        else:
            break

    is_strong_outlier = np.full_like(energies, False, dtype=bool)
    is_weak_outlier = np.full_like(energies, False, dtype=bool)
    for gid, strong, weak in zip(unique_groups, is_strong_group, is_weak_group):
        idx = group_ids == gid
        is_strong_outlier[idx] = strong
        is_weak_outlier[idx] = weak

    return is_strong_outlier, is_weak_outlier


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare_dft",
        action="store_true",
        help="If set, runs detect_true_outlier() to compare with DFT;"
        " otherwise, runs outlier_candidates().",
    )
    parser.add_argument(
        "--result_paths",
        nargs="*",
        type=str,
        default=None,
        help="Path(s) to RSS result log file(s).",
    )
    parser.add_argument(
        "--dft_path",
        type=str,
        default=None,
        help="Path to the directory containing DFT results for outlier structures.",
    )
    args = parser.parse_args()

    if not args.compare_dft:
        outlier_candidates(args.result_paths)
    else:
        detect_actual_outlier(args.dft_path)


def outlier_candidates(result_paths):
    # Prepare output directory: remove existing files if already exists
    os.makedirs("outlier/outlier_candidates", exist_ok=True)
    out_dir = "outlier/outlier_candidates"
    for filename in os.listdir(out_dir):
        if "POSCAR" in filename:
            file_path = os.path.join(out_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Copy weak outlier POSCARs
    outliers_all = []
    for res_path in result_paths:
        with open(res_path) as f:
            loaded_dict = json.load(f)
        rss_results = loaded_dict["rss_results"]

        logname = os.path.basename(res_path).split(".json")[0]
        for res in rss_results:
            if res.get("is_weak_outlier"):
                dest = (
                    f"outlier/outlier_candidates/POSCAR_{logname}_No{res['struct_no']}"
                )
                shutil.copy(res["poscar"], dest)
                _res = res
                _res.pop("structure", None)
                _res["outlier_poscar"] = f"POSCAR_{logname}_No{res['struct_no']}"
                outliers_all.append(_res)

    with open("outlier/outlier_candidates.dat", "w") as f:
        for res in outliers_all:
            print(res, file=f)
    print(f"Detected {len(outliers_all)} potential outliers")


def detect_actual_outlier(dft_path):
    # Load outlier candidates
    outliers_all = []
    with open("outlier/outlier_candidates.dat") as f:
        for line in f:
            outliers_all.append(ast.literal_eval(line.strip()))

    diff_all = []
    for res in outliers_all:
        pressure = res["pressure"]
        poscar_name = res["outlier_poscar"]
        vasprun_paths = glob.glob(f"{dft_path}/{poscar_name}/vasprun*.xml")

        vasprun_get = False
        for vasprun in vasprun_paths:
            try:
                vaspobj = Vasprun(vasprun)
                vasprun_get = True
            except Exception:
                continue
        if not vasprun_get:
            diff_all.append(
                {
                    "diff": "null",
                    "dft_value": "null",
                    "mlp_value": "null",
                    "res": res,
                }
            )
            continue

        energy_dft = vaspobj.energy
        structure = vaspobj.structure
        for element in structure.elements:
            energy_dft -= ground_state_energy(element)
        energy_dft /= len(structure.elements)

        # Subtract pressure term from MLP enthalpy
        mlp_energy = res["energy"]
        mlp_energy -= (
            pressure * structure.volume / (EVAngstromToGPa * len(structure.elements))
        )

        diff = mlp_energy - energy_dft
        diff_all.append(
            {
                "diff": diff,
                "dft_value": energy_dft,
                "mlp_value": mlp_energy,
                "res": res,
            }
        )

    # Write results
    n_true_outlier = 0
    with open("outlier/outlier_detection.yaml", "w") as f:
        print("outliers:", file=f)
        for diff in diff_all:
            poscar = diff["res"]["outlier_poscar"]
            delta = diff["diff"]

            print(f"  - structure: {poscar}", file=f)
            if delta is not None:
                print(f"    energy_diff_meV_per_atom: {delta*1000:.3f}", file=f)
                if delta < -0.1:
                    assessment = "Marked as outlier"
                    n_true_outlier += 1
                else:
                    assessment = "Not an outlier"
                print(f"    assessment: {assessment}", file=f)
            else:
                print("    energy_diff_meV_per_atom: null", file=f)
                print("    assessment: Marked as outlier", file=f)
                n_true_outlier += 1

            print("    details:", file=f)
            for key, val in diff.items():
                if key == "res":
                    continue
                print(f"      {key}: {val}", file=f)

    print(f"Detected {n_true_outlier} actual outliers")

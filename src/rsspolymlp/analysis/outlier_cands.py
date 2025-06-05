import argparse
import ast
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
        Boolean array marking potential outliers (energy diff > 0.1).
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
        if diff > 0.1:
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
    from rsspolymlp.analysis.rss_summarize import load_rss_results

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_paths",
        nargs="*",
        type=str,
        required=True,
        help="Path(s) to RSS result log file(s).",
    )
    args = parser.parse_args()

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
    for res_path in args.result_paths:
        logname = os.path.basename(res_path).split(".log")[0]
        rss_results, _ = load_rss_results(
            res_path, absolute_path=True, get_warning=True
        )

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
    with open("outlier/outlier_candidates.log", "w") as f:
        for res in outliers_all:
            print(res, file=f)


def run_compare_dft():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dft_path",
        type=str,
        required=True,
        help="Path to the directory containing DFT results for outlier structures.",
    )
    args = parser.parse_args()
    dft_path = args.dft_path

    # Load outlier candidates
    outliers_all = []
    with open("outlier/outlier_candidates.log") as f:
        for line in f:
            outliers_all.append(ast.literal_eval(line.strip()))

    diff_all = []
    for res in outliers_all:
        pressure = res["pressure"]
        poscar_name = res["outlier_poscar"]
        vasprun_path = f"{dft_path}/{poscar_name}/vasprun.xml"

        try:
            vaspobj = Vasprun(vasprun_path)
        except Exception:
            diff_all.append(
                {
                    "diff": None,
                    "dft_value": None,
                    "mlp_value": None,
                    "res": res,
                }
            )
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
    with open("outlier/outlier_detection.log", "w") as f:
        for diff in diff_all:
            poscar = diff["res"]["outlier_poscar"]
            delta = diff["diff"]

            if delta is not None:
                print(f"Structure: {poscar}", file=f)
                print(f" - Energy difference (MLP - DFT): {delta:.3f} eV/atom", file=f)
                if delta < -0.1:
                    print(" - Assessment: Marked as outlier", file=f)
                else:
                    print(" - Assessment: Not an outlier", file=f)
                print(f" - Details: {diff}\n", file=f)
            else:
                print(f"Structure: {poscar}", file=f)
                print(f" - Energy difference (MLP - DFT): {delta}", file=f)
                print(" - Assessment: Marked as outlier", file=f)
                print(f" - Details: {diff}\n", file=f)


if __name__ == "__main__":
    run()

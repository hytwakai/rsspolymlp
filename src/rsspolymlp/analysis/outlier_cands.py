import argparse
import ast
import glob
import json
import os
import re
import shutil

import numpy as np
from sklearn.cluster import KMeans

from pypolymlp.core.interface_vasp import Vasprun
from rsspolymlp.utils.ground_state_e import ground_state_energy

EV = 1.602176634e-19  # [J]
EVAngstromToGPa = EV * 1e21


def detect_outlier(energies: np.array, distances: np.array):
    """Detect outliers and potential outliers in a 1D energy array"""
    n = len(energies)
    if n == 1:
        return np.array([False]), np.array([False])

    energy_diffs = np.diff(energies)
    mask = np.abs(energy_diffs) > 1e-6

    group_ids = np.cumsum(mask)
    group_ids = np.concatenate([[0], group_ids])

    unique_groups = np.unique(group_ids)
    cent_e = np.array([np.mean(energies[group_ids == gid]) for gid in unique_groups])
    cent_dist = np.array(
        [np.mean(distances[group_ids == gid]) for gid in unique_groups]
    )

    outliers, not_outliers, is_outlier_group = _detect_outlier_kmeans(
        cent_e, cent_dist, len(cent_e)
    )

    is_outlier = np.full_like(energies, False, dtype=bool)
    for gid, is_out in zip(unique_groups, is_outlier_group):
        idx = group_ids == gid
        is_outlier[idx] = is_out

    return is_outlier, [not_outliers[0], outliers]


def _detect_outlier_kmeans(cent_e, cent_dist, num_energy):
    for prop in [0.5, 0.2, 0.1, 0.01]:
        window = int(round(num_energy * prop))
        window = max(window, 5)

        end = min(window, len(cent_e) - 1)
        data = cent_dist[0 : end + 1]

        if prop == 0.5:
            dist_mean = np.mean(data)
        valid_data_idx = np.where(data > dist_mean * 0.5)[0]
        invalid_data_idx = np.where(data <= dist_mean * 0.5)[0]
        valid_data = data[valid_data_idx]
        if len(valid_data) == 0:
            continue

        kmeans = KMeans(n_clusters=2, random_state=0).fit(valid_data.reshape(-1, 1))
        labels = kmeans.labels_

        cluster_means = [np.mean(valid_data[labels == i]) for i in range(2)]
        outlier_label = np.argmin(cluster_means)

        is_outlier_valid_data = np.full(valid_data.shape, False, dtype=bool)
        outlier_indices = np.where(labels == outlier_label)[0]
        is_outlier_valid_data[outlier_indices] = True
        # print(is_outlier_valid_data)
        is_outlier_valid_data[np.argmax(~is_outlier_valid_data) + 1 :] = False

        outliers_valdat = valid_data[is_outlier_valid_data]
        not_outliers_valdat = valid_data[~is_outlier_valid_data]
        # print(outliers_valdat)
        # print(not_outliers_valdat)

        is_outlier = np.full(cent_dist.shape, False, dtype=bool)
        if len(invalid_data_idx) > 0:
            is_outlier[invalid_data_idx] = True

        if len(outliers_valdat) > 0:
            outlier_mean = np.mean(outliers_valdat)
            # print(outlier_mean)
            not_outlier_mean = np.mean(not_outliers_valdat)
            # print(not_outlier_mean)
            if outlier_mean / not_outlier_mean < 0.8:
                is_outlier[valid_data_idx[is_outlier_valid_data]] = True
                is_outlier[np.argmax(~is_outlier) + 1 :] = False
                outliers = cent_dist[is_outlier]
                not_outliers = cent_dist[~is_outlier]
                break

        is_outlier[np.argmax(~is_outlier) + 1 :] = False
        outliers = cent_dist[is_outlier]
        not_outliers = cent_dist[~is_outlier]

    return outliers, not_outliers, is_outlier


def get_outlier_results(dir_path):
    dist_min_e = []
    with open(f"{dir_path}/outlier/dist_minE_struct.dat") as f:
        for line in f:
            dist_min_e.append(float(line.split()[0]))
    dist_min_e = np.array(dist_min_e)
    print("dist_min_e")
    # print(np.sort(dist_min_e))
    print("min, max =", np.min(dist_min_e), np.max(dist_min_e))
    if os.path.isfile(f"{dir_path}/outlier/dist_outlier.dat"):
        with open(f"{dir_path}/outlier/dist_outlier.dat") as f:
            content = f.read()
        dist_outlier = re.findall(r"\d+\.\d+", content)
        dist_outlier = np.array([float(n) for n in dist_outlier])
        print("dist_outlier")
        print(np.sort(dist_outlier))
        # print("min, max =", np.min(dist_outlier), np.max(dist_outlier))


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
        "--dft_dir",
        type=str,
        default=None,
        help="Path to the directory containing DFT results for outlier structures.",
    )
    args = parser.parse_args()

    if not args.compare_dft:
        dir_path = os.path.dirname(args.result_paths[0])
        os.makedirs(f"{dir_path}/../outlier/outlier_candidates", exist_ok=True)
        os.chdir(f"{dir_path}/../")
        outlier_candidates(args.result_paths)
    else:
        base_dir = os.path.basename(args.dft_dir)
        os.makedirs(f"{base_dir}/../outlier/outlier_candidates", exist_ok=True)
        os.chdir(f"{base_dir}/../")
        detect_actual_outlier(args.dft_dir)


def outlier_candidates(result_paths):
    # Prepare output directory: remove existing files if already exists
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
            if res.get("is_outlier"):
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
            if not delta == "null":
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

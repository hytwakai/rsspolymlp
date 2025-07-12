import ast
import glob
import json
import os
import re
import shutil

import numpy as np
from sklearn.cluster import KMeans

from pypolymlp.core.interface_vasp import Vasprun
from pypolymlp.core.units import EVtoGPa
from rsspolymlp.common.atomic_energy import atomic_energy


def detect_ghost_minima(energies: np.array, distances: np.array):
    """Detect ghost minima and potential ghost minima in a 1D energy array"""
    n = len(energies)
    if n == 1:
        return np.array([False]), [distances[0], []]

    energy_diffs = np.diff(energies)
    mask = np.abs(energy_diffs) > 1e-6

    group_ids = np.cumsum(mask)
    group_ids = np.concatenate([[0], group_ids])

    unique_groups = np.unique(group_ids)
    cent_e = np.array([np.mean(energies[group_ids == gid]) for gid in unique_groups])
    cent_dist = np.array(
        [np.mean(distances[group_ids == gid]) for gid in unique_groups]
    )

    ghost_minima, not_ghost_minima, is_ghost_minima_group = _detect_ghost_minima_kmeans(
        cent_e, cent_dist, len(cent_e)
    )

    is_ghost_minima = np.full_like(energies, False, dtype=bool)
    for gid, is_out in zip(unique_groups, is_ghost_minima_group):
        idx = group_ids == gid
        is_ghost_minima[idx] = is_out

    return is_ghost_minima, [not_ghost_minima[0], ghost_minima]


def _detect_ghost_minima_kmeans(cent_e, cent_dist, num_energy):
    ghost_minima = []
    not_ghost_minima = cent_dist
    is_ghost_minima = np.full(cent_dist.shape, False, dtype=bool)

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
        if len(valid_data) < 5:
            continue

        kmeans = KMeans(n_clusters=2, random_state=0).fit(valid_data.reshape(-1, 1))
        labels = kmeans.labels_
        cluster_means = [np.mean(valid_data[labels == i]) for i in range(2)]

        is_ghost_minima_valid_data = np.full(valid_data.shape, False, dtype=bool)
        ghmin_indices = np.where(labels == np.argmin(cluster_means))[0]
        is_ghost_minima_valid_data[ghmin_indices] = True
        is_ghost_minima_valid_data[np.argmax(~is_ghost_minima_valid_data) + 1 :] = False

        ghost_minima_valdat = valid_data[is_ghost_minima_valid_data]
        not_ghost_minima_valdat = valid_data[~is_ghost_minima_valid_data]

        is_ghost_minima = np.full(cent_dist.shape, False, dtype=bool)
        if len(invalid_data_idx) > 0:
            is_ghost_minima[invalid_data_idx] = True

        if len(ghost_minima_valdat) > 0:
            ghmin_mean = np.mean(ghost_minima_valdat)
            not_ghmin_mean = np.mean(not_ghost_minima_valdat)

            if ghmin_mean / not_ghmin_mean < 0.85:
                is_ghost_minima[valid_data_idx[is_ghost_minima_valid_data]] = True
                is_ghost_minima[np.argmax(~is_ghost_minima) + 1 :] = False
                ghost_minima = cent_dist[is_ghost_minima]
                not_ghost_minima = cent_dist[~is_ghost_minima]
                break

        is_ghost_minima[np.argmax(~is_ghost_minima) + 1 :] = False
        ghost_minima = cent_dist[is_ghost_minima]
        not_ghost_minima = cent_dist[~is_ghost_minima]

    return ghost_minima, not_ghost_minima, is_ghost_minima


def get_ghost_minima_dists(dir_path):
    dist_min_e = []
    with open(f"{dir_path}/ghost_minima/dist_minE_struct.dat") as f:
        for line in f:
            dist_min_e.append(float(line.split()[0]))
    dist_min_e = np.array(dist_min_e)
    print("dist_min_e")
    # print(np.sort(dist_min_e))
    print("min, max =", np.min(dist_min_e), np.max(dist_min_e))
    if os.path.isfile(f"{dir_path}/ghost_minima/dist_ghost_minima.dat"):
        with open(f"{dir_path}/ghost_minima/dist_ghost_minima.dat") as f:
            content = f.read()
        dist_ghost_minima = re.findall(r"\d+\.\d+", content)
        dist_ghost_minima = np.array([float(n) for n in dist_ghost_minima])
        print("dist_ghost_minima")
        print(np.sort(dist_ghost_minima))
        # print("min, max =", np.min(dist_ghost_minima), np.max(dist_ghost_minima))


def ghost_minima_candidates(result_paths):
    # Prepare output directory: remove existing files if already exists
    out_dir = "ghost_minima/ghost_minima_candidates"
    for filename in os.listdir(out_dir):
        if "POSCAR" in filename:
            file_path = os.path.join(out_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Copy weak ghost_minima POSCARs
    ghost_minima_all = []
    for res_path in result_paths:
        with open(res_path) as f:
            loaded_dict = json.load(f)
        rss_results = loaded_dict["rss_results"]

        logname = os.path.basename(res_path).split(".json")[0]
        for res in rss_results:
            if res.get("is_ghost_minima"):
                dest = f"ghost_minima/ghost_minima_candidates/POSCAR_{logname}_No{res['struct_no']}"
                shutil.copy(res["poscar"], dest)
                _res = res
                _res.pop("structure", None)
                _res["ghost_minima_poscar"] = f"POSCAR_{logname}_No{res['struct_no']}"
                ghost_minima_all.append(_res)

    with open("ghost_minima/ghost_minima_candidates.dat", "w") as f:
        for res in ghost_minima_all:
            print(res, file=f)
    print(f"Detected {len(ghost_minima_all)} potential ghost_minima")


def detect_actual_ghost_minima(dft_path):
    # Load ghost_minima candidates
    ghost_minima_all = []
    with open("ghost_minima/ghost_minima_candidates.dat") as f:
        for line in f:
            ghost_minima_all.append(ast.literal_eval(line.strip()))

    diff_all = []
    for res in ghost_minima_all:
        pressure = res["pressure"]
        poscar_name = res["ghost_minima_poscar"]
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
                    "energy_diff": "null",
                    "stress_diff": "null",
                    "dft_energy": "null",
                    "mlp_energy": "null",
                    "dft_stress": "null",
                    "res": res,
                }
            )
            continue

        energy_dft = vaspobj.energy
        structure = vaspobj.structure
        for element in structure.elements:
            energy_dft -= atomic_energy(element)
        energy_dft /= len(structure.elements)
        vol_per_atom = structure.volume / len(structure.elements)

        # Subtract pressure term from MLP enthalpy
        mlp_energy = res["energy"]
        mlp_energy -= pressure * vol_per_atom / EVtoGPa

        stress_dft = [(vaspobj.stress / 10).tolist()[i][i] for i in range(3)]
        press_diff = [pressure - stress_dft[i] for i in range(3)]
        stress_diff = np.max(np.abs(press_diff)) * vol_per_atom / EVtoGPa

        e_diff = mlp_energy - energy_dft
        diff_all.append(
            {
                "energy_diff": e_diff,
                "stress_diff": stress_diff,
                "dft_energy": energy_dft,
                "mlp_energy": mlp_energy,
                "dft_stress": stress_dft,
                "res": res,
            }
        )

    # Write results
    n_true_ghost_minima = 0
    n_not_ghost_minima = 0
    with open("ghost_minima/ghost_minima_detection.yaml", "w") as f:
        print("ghost_minima:", file=f)
        for diff in diff_all:
            poscar = diff["res"]["ghost_minima_poscar"]
            delta_e = diff["energy_diff"]
            delta_s = diff["stress_diff"]

            print(f"  - structure: {poscar}", file=f)
            if not delta_e == "null":
                print(f"    energy_diff_meV_per_atom: {delta_e*1000:.3f}", file=f)
                print(f"    stress_diff_meV_per_atom: {delta_s*1000:.3f}", file=f)
                e_threshold = -0.1  # unit: eV/atom
                s_threshold = 0.5
                if delta_e < e_threshold or delta_s > s_threshold:
                    assessment = "Marked as ghost minimum"
                    n_true_ghost_minima += 1
                else:
                    assessment = "Not a ghost minimum"
                    n_not_ghost_minima += 1
                print(f"    assessment: {assessment}", file=f)
            else:
                print("    energy_diff_meV_per_atom: null", file=f)
                print("    assessment: Marked as ghost minimum", file=f)
                n_true_ghost_minima += 1

            print("    details:", file=f)
            for key, val in diff.items():
                if key == "res":
                    continue
                print(f"      {key}: {val}", file=f)

    print(f"{n_true_ghost_minima} structures are actual ghost minima")
    print(f"{n_not_ghost_minima} structures are not ghost minima.")

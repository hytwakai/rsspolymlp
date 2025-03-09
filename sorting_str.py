import glob
import os
import re
from time import time

import numpy as np
from go_tools.initial_str import generate_initial_structure

from pypolymlp.core.interface_vasp import Poscar


def read_file(poscar_name):
    _res = {
        "potential": None,
        "spg": None,
        "res_f": None,
        "res_s": None,
        "time": None,
        "energy": None,
        "iter": 0,
        "fval": 0,
        "gval": 0,
        "dup_count": 1,
    }
    _res["poscar"] = poscar_name
    with open(f"log/{_res['poscar']}.log") as f:
        s_lines = [i.strip() for i in f]
        for i in range(len(s_lines)):
            if "Selected potential:" in s_lines[i]:
                _res["potential"] = str(s_lines[i].split()[-1])
            if "Space group set" in s_lines[i]:
                _res["spg"] = eval(s_lines[i + 1])
            if "Iterations" in s_lines[i]:
                _res["iter"] += int(s_lines[i].split()[-1])
            if "Function evaluations" in s_lines[i]:
                _res["fval"] += int(s_lines[i].split()[-1])
            if "Gradient evaluations" in s_lines[i]:
                _res["gval"] += int(s_lines[i].split()[-1])
            if "Maximum absolute value in Residuals (force)" in s_lines[i]:
                _res["res_f"] = float(s_lines[i].split()[-1])
            if "Maximum absolute value in Residuals (stress)" in s_lines[i]:
                _res["res_s"] = float(s_lines[i].split()[-1])
            if "Computational time" in s_lines[i]:
                _res["time"] = float(s_lines[i].split()[-1])
            if "Final function value (eV/atom):" in s_lines[i]:
                _res["energy"] = float(s_lines[i].split()[-1])
            if "Maximum number of relaxation iterations has been exceeded" in s_lines[i]:
                return _res, "iteration"
            if "Geometry optimization failed: Huge" in s_lines[i]:
                if abs(_res["energy"]) < 10**-3:
                    return _res, "energy_zero"
                else:
                    return _res, "energy_low"

    return _res, True


def angle_between(v1, v2):
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))


def axis_to_abc(a, b, c):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    norm_c = np.linalg.norm(c)
    alpha = angle_between(a, b)
    beta = angle_between(b, c)
    gamma = angle_between(c, a)
    return np.array([norm_a, norm_b, norm_c, alpha, beta, gamma]).tolist()


nondup_str = []
iter_str = []
fval_str = []
gval_str = []
f_conv = 0
s_conv = 0
energy_low_err = 0
energy_zero_err = 0
iteration_err = 0
else_err = 0
energy_low_err_poscar = []
energy_zero_err_poscar = []
not_converged_poscar_f = []
not_converged_poscar_s = []
iteration_err_poscar = []
else_err_poscar = []
time_all = 0

time_ini = time()

poscar_path = glob.glob("initial_str/*")
poscar_path = sorted(poscar_path, key=lambda x: int(x.split("_")[-1]))
for pos in poscar_path:
    poscar_name = pos.split("/")[-1]
    _res, judge = read_file(poscar_name)
    if judge == "iteration":
        iteration_err += 1
        iteration_err_poscar.append(_res["poscar"])
    elif judge == "energy_low":
        energy_low_err += 1
        energy_low_err_poscar.append(_res["poscar"])
    elif judge == "energy_zero":
        energy_zero_err += 1
        energy_zero_err_poscar.append(_res["poscar"])
    if judge is not True:
        continue
    if any(_res[key] is None for key in ["time", "spg", "energy", "res_f", "res_s"]):
        else_err += 1
        else_err_poscar.append(_res["poscar"])
        continue

    if _res["res_f"] > 10**-4:
        f_conv += 1
        not_converged_poscar_f.append(_res["poscar"])
        continue
    if _res["res_s"] > 10**-3:
        s_conv += 1
        not_converged_poscar_s.append(_res["poscar"])
        continue

    if os.path.isfile(f"optimized_str/{_res['poscar']}"):
        polymlp_st = Poscar(f"optimized_str/{_res['poscar']}").structure
        _res["elements"] = polymlp_st.elements
        _res["volume"] = np.linalg.det(polymlp_st.axis) / len(_res["elements"])
        a, b, c = np.array(polymlp_st.axis)
        _res["axis"] = axis_to_abc(a, b, c)
        _res["positions"] = polymlp_st.positions.T.tolist()
        _res["distance"] = generate_initial_structure().nearest_neighbor_atomic_distance(
            polymlp_st.axis, polymlp_st.positions
        )
    else:
        else_err += 1
        else_err_poscar.append(_res["poscar"])
        continue

    app_ = True
    change = False
    for idx, entry in enumerate(nondup_str):
        ene, spg_set = entry["energy"], entry["spg"]
        if abs(ene - _res["energy"]) < 1e-8:
            if any(spg in _res["spg"] for spg in spg_set):
                app_ = False
                spg_count1 = sum(int(re.search(r"\((\d+)\)", s).group(1)) for s in spg_set)
                spg_count2 = sum(int(re.search(r"\((\d+)\)", s).group(1)) for s in _res["spg"])
                if spg_count2 > spg_count1:
                    change = True
                break

    if iter_str == []:
        iter_str.append(_res["iter"])
        fval_str.append(_res["fval"])
        gval_str.append(_res["gval"])
    else:
        iter_str[-1] += _res["iter"]
        fval_str[-1] += _res["fval"]
        gval_str[-1] += _res["gval"]

    if not app_:
        nondup_str[idx]["dup_count"] += 1
        if change:
            for key in ["energy", "spg", "axis", "positions", "elements", "volume", "distance"]:
                nondup_str[idx][key] = _res[key]
    else:
        nondup_str.append(_res)
        iter_str.append(iter_str[-1])
        fval_str.append(fval_str[-1])
        gval_str.append(gval_str[-1])
        _res["iter"] = iter_str[-1]
        _res["fval"] = fval_str[-1]
        _res["gval"] = gval_str[-1]

    time_all += _res["time"]

time_fin = time() - time_ini

nondup_str = sorted(nondup_str, key=lambda x: x["energy"])
energy_min = -10
for _str in nondup_str:
    if energy_min < _str["energy"]:
        energy_min = _str["energy"]
        break

error_count = sum([energy_low_err, energy_zero_err, f_conv, s_conv, iteration_err, else_err])
with open("sorted_result.log", "w") as f:
    print("---- General results ----", file=f)
    print("Sorting time (sec.):           ", round(time_fin, 3), file=f)
    print("Selected potential:            ", nondup_str[0]["potential"], file=f)
    print("Number of initial structures:  ", len(poscar_path), file=f)
    print("Number of optimized structures:", len(poscar_path) - error_count, file=f)
    print("Total computation time (sec.): ", round(time_all, 1), file=f)
    print("", file=f)
    print("---- Error counts ----", file=f)
    print("Total error counts:", error_count, file=f)
    print(" - Low energy:        ", energy_low_err, file=f)
    print(" - Zero energy:       ", energy_zero_err, file=f)
    print(" - Force convergence: ", f_conv, file=f)
    print(" - Stress convergence:", s_conv, file=f)
    print(" - Max iteration:     ", iteration_err, file=f)
    print(" - Other reason:      ", else_err, file=f)
    print("", file=f)
    print("---- Nonduplicate structures ----", file=f)
    for idx, _str in enumerate(nondup_str):
        e_diff = round((_str["energy"] - energy_min) * 1000, 2)
        print(f"No. {idx+1}", file=f)
        print(f"{_str['poscar']} ({e_diff} meV/atom, {_str['dup_count']} duplicates)", file=f)
        print(" - Enthalpy:   ", _str["energy"], file=f)
        print(" - Axis:       ", _str["axis"], file=f)
        print(" - Postions:   ", _str["positions"], file=f)
        print(" - Elements:   ", _str["elements"], file=f)
        print(" - Space group:", _str["spg"], file=f)
        print(
            " - Other Info.:",
            int(len(_str["elements"])),
            "atom",
            "/ distance",
            round(_str["distance"], 3),
            "(Ang.) / volume",
            round(_str["volume"], 2),
            "(A^3/atom) / iteration",
            _str["iter"],
            file=f,
        )

    print("", file=f)
    print("---- Total evaluation counts ----", file=f)
    print("Iteration:           ", iter_str[-1], file=f)
    print("Function evaluations:", fval_str[-1], file=f)
    print("Gradient evaluations:", gval_str[-1], file=f)
    print("", file=f)
    print("---- Evaluation count per structure ----", file=f)
    print("Iteration (list):           ", iter_str, file=f)
    print("Function evaluations (list):", fval_str, file=f)
    print("Gradient evaluations (list):", gval_str, file=f)
    print("", file=f)

    print("---- Poscar name (failed) ----", file=f)
    print("Low energy:        ", energy_low_err_poscar, file=f)
    print("Zero energy:       ", energy_zero_err_poscar, file=f)
    print("Force convergence: ", not_converged_poscar_f, file=f)
    print("Stress convergence:", not_converged_poscar_s, file=f)
    print("Max iteration:     ", iteration_err_poscar, file=f)
    print("Other reason:      ", else_err_poscar, file=f)

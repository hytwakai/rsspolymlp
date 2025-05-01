import copy
import glob
import time

import numpy as np

from myutils import pymat_util
from rsspolymlp.struct_matcher.irrep_position import IrrepPos
from rsspolymlp.utils.spglib_utils import SymCell

pymat = pymat_util.MyPymat()

poscar_num = 1000
compare_pymatgen = False

poscar_all = glob.glob("./poscars2/*")
poscar_all = poscar_all[:poscar_num]

print("--- Comparing irrep atomic position ---")
sym_st = []
start = time.time()
for p_name in poscar_all:
    symutil = SymCell(poscar_name=p_name, symprec=1e-3)
    st = symutil.primitive_cell()
    _res = {}
    _res["poscar"] = p_name
    _res["structure"] = st
    sym_st.append(_res)
el_time1 = round(time.time() - start, 3)
print(" convert to primitive cell:", (el_time1) * 1000)
print(f"  (elapsed time per structure: {(el_time1) * 1000 / poscar_num})")

irrep_pos = IrrepPos(symprec=1e-5)
start = time.time()
for i in range(len(sym_st)):
    axis = sym_st[i]["structure"].axis
    pos = sym_st[i]["structure"].positions.T
    rep_pos, recommend_order = irrep_pos.irrep_positions(axis, pos)
    sym_st[i]["rep_pos"] = rep_pos
    sym_st[i]["recommend_order"] = recommend_order
el_time2 = round(time.time() - start, 3)
print(" get irrep atomic positions:", (el_time2) * 1000)
print(f"  (elapsed time per structure: {(el_time2) * 1000 / poscar_num})")

start = time.time()
nondup_st = []
count1 = 0
for st in sym_st:
    app = True
    for st_ref in nondup_st:
        if len(st_ref["rep_pos"]) == len(st["rep_pos"]):
            count1 += 1
            if len(st["rep_pos"]) != 0:
                diffs = st["rep_pos"] - st_ref["rep_pos"]
                residual_pos = np.max(abs(diffs))
            else:
                residual_pos = 0
            residual_axis = st["structure"].axis - st_ref["structure"].axis
            residual_axis = np.max(np.sum(residual_axis**2, axis=1))
            if residual_pos < 0.01 and residual_axis < 0.01:
                app = False
                break
    if app:
        nondup_st.append(st)
el_time3 = round(time.time() - start, 3)
print(" eliminate duplicates:", (el_time3) * 1000)
print(" evaluation count:", count1)
print(f"  (elapsed time per evaluation: {round((el_time3) * 1000 / count1, 3)})")
pos_irpos = []
for st in nondup_st:
    pos_irpos.append(st["poscar"])
print(" number of nonduplicate structures:", len(nondup_st))

print("--- Comparing irrep atomic position (in recommended symprec) ---")
start = time.time()
for i in range(len(nondup_st)):
    rep_pos = []
    for symprec in nondup_st[i]["recommend_order"]:
        axis = nondup_st[i]["structure"].axis
        pos = nondup_st[i]["structure"].positions.T
        irrep_pos = IrrepPos(symprec=symprec)
        _rep_pos, recommend_order = irrep_pos.irrep_positions(axis, pos)
        rep_pos.append(_rep_pos)
    if not rep_pos == []:
        nondup_st[i]["rep_pos"] = np.stack(rep_pos, axis=0)
    else:
        nondup_st[i]["rep_pos"] = [[], []]
_sym_st = copy.copy(nondup_st)
el_time4 = round(time.time() - start, 3)
print(" get irrep atomic positions::", (el_time4) * 1000)
print(f"  (elapsed time per structure: {round((el_time4) * 1000 / len(nondup_st), 2)})")

start = time.time()
count2 = 0
nondup_st = []
for st in _sym_st:
    app = True
    for st_ref in nondup_st:
        if len(st_ref["rep_pos"][0]) == len(st["rep_pos"][0]):
            count2 += 1
            if len(st["rep_pos"]) > 0:
                diffs = st_ref["rep_pos"][:, None, :] - st["rep_pos"][None, :, :]
                diffs_flat = diffs.reshape(-1, diffs.shape[2])
                residual_pos = np.min(np.max(np.abs(diffs_flat), axis=1))
            else:
                residual_pos = 0
            residual_axis = st["structure"].axis - st_ref["structure"].axis
            residual_axis = np.max(np.sum(residual_axis**2, axis=1))
            if residual_pos < 0.01 and residual_axis < 0.01:
                app = False
                break
    if app:
        nondup_st.append(st)
el_time5 = round(time.time() - start, 3)
print(" eliminate duplicates:", (el_time5) * 1000)
print(" evaluation count:", count2)
print(f"  (elapsed time per evaluation: {round((el_time5) * 1000 / count2, 3)})")
pos_irpos = []
for st in nondup_st:
    pos_irpos.append(st["poscar"])
print(" number of nonduplicate structures:", len(nondup_st))

if compare_pymatgen:
    print("--- Pymatgen.StructureMatcher result ---")
    sym_st_past = sym_st
    sym_st = []
    start = time.time()
    for st_in in sym_st_past:
        st = {}
        st["pymat"] = pymat.parameter_to_pymat_st(
            st_in["structure"].axis,
            st_in["structure"].positions.T,
            st_in["structure"].elements,
        )
        st["poscar"] = st_in["poscar"]
        st["duplicate"] = [st_in["poscar"]]
        sym_st.append(st)
    el_time6 = round(time.time() - start, 3)
    print(" generate Pymatgen.Structure:", (el_time6) * 1000)

    start = time.time()
    nondup_st_pymat = []
    count3 = 0
    for st in sym_st:
        app = True
        for i in range(len(nondup_st_pymat)):
            st_ref = nondup_st_pymat[i]
            if len(st_ref["pymat"].species) == len(st["pymat"].species):
                count3 += 1
                if pymat.match_str(
                    st["pymat"],
                    st_ref["pymat"],
                    primitive_cell=False,
                    ltol=0.01,
                    stol=0.01,
                    angle_tol=0.5,
                ):
                    app = False
                    nondup_st_pymat[i]["duplicate"].append(st["poscar"])
                    break
        if app:
            nondup_st_pymat.append(st)
    el_time7 = round(time.time() - start, 3)
    print(" eliminate duplicates:", (el_time7) * 1000)
    print(" evaluation count:", count3)
    print(f"  (elapsed time per evaluation: {round((el_time7) * 1000 / count3, 3)})")
    pos_pymat = []
    for st in nondup_st_pymat:
        pos_pymat.append(st["poscar"])
    print(" number of nonduplicate structures:", len(nondup_st_pymat))
    print("--- Comparing nonduplicate structures ---")
    print("set(irrep_pos) - set(pymatgen):")
    print(set(pos_irpos) - set(pos_pymat))
    print("set(pymatgen) - set(irrep_pos):")
    print(set(pos_pymat) - set(pos_irpos))

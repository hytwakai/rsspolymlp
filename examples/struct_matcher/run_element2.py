import glob
import os
import tarfile
import time

import numpy as np

from rsspolymlp.analysis.struct_matcher.struct_match import (
    generate_irrep_struct,
    generate_primitive_cells,
    struct_match,
)
from rsspolymlp.analysis.struct_matcher.utils import get_recommend_symprecs
from rsspolymlp.utils import pymatgen_utils

pymat = pymatgen_utils.PymatUtil()

poscar_num = 2000
compare_pymatgen = True

if not os.path.exists("./poscar_element3"):
    os.makedirs("./poscar_element3")
    with tarfile.open("./poscar_element.tar.gz", "r:gz") as tar:
        tar.extractall(path="./poscar_element2", filter="data")

poscar_all = sorted(glob.glob("./poscar_element3/*"))
poscar_all = poscar_all[:poscar_num]

print("--- Comparing irrep atomic position ---")
sym_st = []
start = time.time()
for p_name in poscar_all:
    st, spg = generate_primitive_cells(
        poscar_name=p_name,
        symprec_set=[1e-5, 1e-4, 1e-3, 1e-2],
    )
    _res = {}
    _res["poscar"] = p_name
    _res["structure"] = st
    _res["spg_number"] = spg
    sym_st.append(_res)
el_time1 = round(time.time() - start, 3)
print(" convert to primitive cell:", (el_time1) * 1000)
print(f"  (elapsed time per structure: {(el_time1) * 1000 / poscar_num})")

start = time.time()
for i in range(len(sym_st)):
    sym_st[i]["irrep_st"] = []
    for h, st in enumerate(sym_st[i]["structure"]):
        recommend_symprecs = get_recommend_symprecs(st)
        symprec_list = [1e-5]
        symprec_list.extend(recommend_symprecs)
        irrep_st = generate_irrep_struct(
            st,
            sym_st[i]["spg_number"][h],
            symprec_irreps=symprec_list,
        )
        sym_st[i]["irrep_st"].append(irrep_st)
    # print(sym_st[i]["poscar"])
el_time2 = round(time.time() - start, 3)
print(" get irrep atomic positions:", (el_time2) * 1000)
print(f"  (elapsed time per structure: {(el_time2) * 1000 / poscar_num})")

start = time.time()
nondup_st = []
count1 = 0
for st in sym_st:
    app = True
    for st_ref in nondup_st:
        count1 += 1
        st_similarity = struct_match(st_ref["irrep_st"], st["irrep_st"])
        if st_similarity:
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

if compare_pymatgen:
    print("--- Pymatgen.StructureMatcher result ---")
    sym_st_past = sym_st
    sym_st = []
    start = time.time()
    for st_in in sym_st_past:
        st = {}
        st["pymat"] = pymat.parameter_to_pymat_st(
            st_in["structure"][-1].axis,
            st_in["structure"][-1].positions.T,
            st_in["structure"][-1].elements,
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
                    ltol=0.1,
                    stol=0.01,
                    angle_tol=1,
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

    with open(f"myresult/result_element{poscar_num}.log", "w") as f:
        print("generate sym_st:", (el_time1) * 1000, file=f)
        print("get irrep_pos:", (el_time2) * 1000, file=f)
        print("sorting sym_st:", (el_time3) * 1000, file=f)
        print(count1, file=f)
        print("generate sym_st (pymatgen):", (el_time6) * 1000, file=f)
        print("sorting sym_st (pymatgen):", (el_time7) * 1000, file=f)
        print(count3, file=f)
        print(len(nondup_st), file=f)
        print(len(nondup_st_pymat), file=f)
        print(set(pos_irpos) - set(pos_pymat), file=f)
        print(set(pos_pymat) - set(pos_irpos), file=f)
        for st in nondup_st_pymat:
            print(st["duplicate"], file=f)

import copy
import glob
import os
import tarfile
import time

from rsspolymlp.utils import pymatgen_utils
from rsspolymlp.struct_matcher.struct_match import get_irrep_positions, struct_match
from rsspolymlp.utils.spglib_utils import SymCell

pymat = pymatgen_utils.PymatUtil()

poscar_num = 1000
compare_pymatgen = False

if not os.path.exists("./poscar_element2"):
    os.makedirs("./poscar_element2")
    with tarfile.open("./poscar_element.tar.gz", "r:gz") as tar:
        tar.extractall(path="./poscar_element2", filter="data")

poscar_all = glob.glob("./poscar_element2/*")
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

start = time.time()
for i in range(len(sym_st)):
    irrep_st = get_irrep_positions(struct=sym_st[i]["structure"], symprec_irreps=[1e-5])
    sym_st[i]["irrep_st"] = irrep_st
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

print("--- Comparing irrep atomic position (in recommended symprec) ---")
start = time.time()
for i in range(len(nondup_st)):
    irrep_st = get_irrep_positions(
        struct=nondup_st[i]["structure"],
        symprec_irreps=nondup_st[i]["irrep_st"].recommend_symprecs,
    )
    nondup_st[i]["irrep_st"] = irrep_st
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
        count2 += 1
        st_similarity = struct_match(st_ref["irrep_st"], st["irrep_st"])
        if st_similarity:
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

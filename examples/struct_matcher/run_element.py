import numpy as np

from rsspolymlp.analysis.struct_matcher.struct_match import struct_match
from rsspolymlp.analysis.unique_struct import generate_unique_struct
from rsspolymlp.utils import pymatgen_utils
from rsspolymlp.utils.property import PropUtil

pymat = pymatgen_utils.PymatUtil()
all_test_mode = ["example", "invert", "swap", "symprec"]
all_test_mode = ["invert"]
all_test_mode = ["symprec_2"]
all_test_mode = ["test"]
all_test_mode = ["poscar_3"]
all_test_mode = ["poscar_else"]
final_res = []
pymatgen_res = []

for test_mode in all_test_mode:
    if test_mode == "example":
        pos1 = "./poscar_element/example_1"
        pos2 = "./poscar_element/example_2"
        symprec = 1e-5
    elif test_mode == "invert":
        pos1 = "./poscar_element/invert_1"
        pos2 = "./poscar_element/invert_2"
        symprec = 1e-5
    elif test_mode == "swap":
        pos1 = "./poscar_element/swap_1"
        pos2 = "./poscar_element/swap_2"
        symprec = 1e-5
    elif test_mode == "symprec":
        pos1 = "./poscar_element/symprec_1e-2_1"
        pos2 = "./poscar_element/symprec_1e-2_2"
        symprec = 1e-2
    elif test_mode == "symprec_2":
        pos1 = "./poscar_element/symprec_1"
        pos2 = "./poscar_element/symprec_2"
        symprec_set = [1e-5, 1e-4, 1e-3, 1e-2]
    elif test_mode == "test":
        pos1 = "./poscar_element/test_1"
        pos2 = "./poscar_element/test_2"
        symprec_set = [1e-5, 1e-4, 1e-3, 1e-2]
    elif test_mode == "poscar":
        pos1 = "./poscar_element3/POSCAR_45"
        pos2 = "./poscar_element3/POSCAR_443"
        symprec_set = [1e-5, 1e-4, 1e-3, 1e-2]
        symprec_set = [1e-5]
    elif test_mode == "poscar_2":
        pos1 = "./poscar_element3/POSCAR_1897"
        pos2 = "./poscar_element3/POSCAR_130"
        symprec_set = [1e-5, 1e-4, 1e-3, 1e-2]
        symprec_set = [1e-5]
    elif test_mode == "poscar_3":
        pos1 = "./poscar_element3/POSCAR_1042"
        pos2 = "./poscar_element3/POSCAR_1023"
        symprec_set = [1e-5, 1e-4, 1e-3, 1e-2]
    elif test_mode == "poscar_else":
        pos1 = "/home/wakai/data/csp_mlp/Bi-Ca/csp_try3_rsspolymlp/0.0GPa/12atom/0_12/opt_struct/POSCAR_46"
        pos2 = "/home/wakai/data/csp_mlp/Bi-Ca/csp_try3_rsspolymlp/0.0GPa/12atom/0_12/opt_struct/POSCAR_673"
        symprec_set = [1e-5, 1e-4, 1e-3, 1e-2]
    else:
        raise ValueError(f"Unknown test_mode: {test_mode}")

    """symutil = SymCell(poscar_name=pos1, symprec=1e-3)
    symutil2 = SymCell(poscar_name=pos2, symprec=1e-3)
    st1 = symutil.refine_cell(standardize_cell=True)
    st2 = symutil2.refine_cell(standardize_cell=True)"""

    print("st1")
    unique_struct1 = generate_unique_struct(
        pos1,
        symprec_set=symprec_set,
    )
    print("st2")
    unique_struct2 = generate_unique_struct(pos2, symprec_set=symprec_set)
    st1 = unique_struct1.original_structure
    st2 = unique_struct2.original_structure

    judge = struct_match(
        unique_struct1.irrep_struct_set, unique_struct2.irrep_struct_set
    )

    print(f"----- test_mode: {test_mode} -----")
    print("--- Stucture 1 ---")
    print(" - Axis:")
    print(PropUtil(st1.axis.T, st1.positions.T).axis_to_abc)
    print(" - Positions:")
    print(np.round(st1.positions, 5))
    print(" - Elements:")
    print(st1.elements)
    for irrep_st in unique_struct1.irrep_struct_set:
        print("spg_number", irrep_st.spg_number)
        print(np.round(irrep_st.positions[0].reshape(3, -1), 3))
        print(np.round(irrep_st.positions[1].reshape(3, -1), 3))
        print(np.round(irrep_st.positions[2].reshape(3, -1), 3))
    print(unique_struct1.recommend_symprecs)
    print("--- Structure 2 ---")
    print(" - Axis:")
    print(PropUtil(st2.axis.T, st2.positions.T).axis_to_abc)
    print(" - Positions:")
    print(np.round(st2.positions, 5))
    print(" - Elements:")
    print(st2.elements)
    for irrep_st in unique_struct2.irrep_struct_set:
        print("spg_number", irrep_st.spg_number)
        print(np.round(irrep_st.positions[0].reshape(3, -1), 3))
        print(np.round(irrep_st.positions[1].reshape(3, -1), 3))
        print(np.round(irrep_st.positions[2].reshape(3, -1), 3))
    print("positions difference")
    try:
        print(
            np.round(
                unique_struct1.irrep_struct_set[0].positions[2].reshape(3, -1)
                - unique_struct2.irrep_struct_set[0].positions[2].reshape(3, -1),
                3,
            )
        )
    except:
        pass
    print(unique_struct2.recommend_symprecs)
    print(f" - These strctures are similar ?: {judge}")
    final_res.append(judge)

    pymat_st1 = pymat.parameter_to_pymat_st(st1.axis, st1.positions.T, st1.elements)
    pymat_st2 = pymat.parameter_to_pymat_st(st2.axis, st2.positions.T, st2.elements)
    judge = pymat.match_str(
        pymat_st1,
        pymat_st2,
        ltol=0.1,
        stol=0.02,
        angle_tol=1,
    )
    print("--- Pymatgen.StructureMatcher result ---")
    print(judge)
    pymatgen_res.append(bool(judge))

print("")
print("Final results")
print("--- Test modes ---")
print(all_test_mode)
print("--- Comparing irrep position ---")
print(final_res)
print("--- Pymatgen.StructureMatcher ---")
print(pymatgen_res)

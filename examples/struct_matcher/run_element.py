import numpy as np

from pypolymlp.utils.spglib_utils import SymCell
from rsspolymlp.analysis.struct_matcher.irrep_position import IrrepPosition
from rsspolymlp.analysis.struct_matcher.struct_match import (
    generate_primitive_cell,
    get_recommend_symprecs,
)
from rsspolymlp.utils import pymatgen_utils
from rsspolymlp.utils.property import PropUtil

pymat = pymatgen_utils.PymatUtil()
all_test_mode = ["example", "invert", "swap", "symprec"]
all_test_mode = ["invert"]
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
    else:
        raise ValueError(f"Unknown test_mode: {test_mode}")

    st1, spg1 = generate_primitive_cell(poscar_name=pos1)
    st2, spg2 = generate_primitive_cell(poscar_name=pos2)
    print(spg1)
    print(spg2)
    symutil = SymCell(poscar_name=pos1, symprec=1e-3)
    symutil2 = SymCell(poscar_name=pos2, symprec=1e-3)
    st1 = symutil.refine_cell(standardize_cell=True)
    st2 = symutil2.refine_cell(standardize_cell=True)

    irrep_pos = IrrepPosition(symprec=symprec)

    rep_pos1, sorted_elements1 = irrep_pos.irrep_positions(
        st1.axis, st1.positions.T, st1.elements, spg1
    )
    order1 = get_recommend_symprecs(st1)
    rep_pos2, sorted_elements2 = irrep_pos.irrep_positions(
        st2.axis, st2.positions.T, st2.elements, spg2
    )
    order2 = get_recommend_symprecs(st2)

    print(f"----- test_mode: {test_mode} -----")
    print("--- Stucture 1 ---")
    print(" - Axis:")
    print(PropUtil(st1.axis.T, st1.positions.T).axis_to_abc)
    print(" - Positions:")
    print(np.round(st1.positions, 5))
    print(" - Elements:")
    print(st1.elements)
    print(" - Irrep positions:")
    print(rep_pos1.reshape(3, -1))
    print(" - Recommended symprec:", order1)
    print("--- Structure 2 ---")
    print(" - Axis:")
    print(PropUtil(st2.axis.T, st2.positions.T).axis_to_abc)
    print(" - Positions:")
    print(np.round(st2.positions, 5))
    print(" - Elements:")
    print(st2.elements)
    print(" - Irrep positions:")
    print(rep_pos2.reshape(3, -1))
    print(" - Recommended symprec:", order2)
    print("--- Structural similarity ---")
    print(" - Difference of axis:")
    residual_axis = st1.axis - st2.axis
    residual_axis = np.sum(residual_axis**2, axis=1)
    print(residual_axis)
    print(" - Difference of irrep positions:")
    print(abs(rep_pos1 - rep_pos2))
    residual_pos = np.max(abs(rep_pos1 - rep_pos2))
    print(" - Maximum absolute value:", residual_pos)
    if residual_pos < 0.01 and np.max(residual_axis) < 0.01:
        print(" - These strctures are similar ?: Yes")
        final_res.append(True)
    else:
        print(" - These strctures are similar ?: No")
        final_res.append(False)

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

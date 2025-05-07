import time

import numpy as np

from pypolymlp.utils.spglib_utils import SymCell
from rsspolymlp.struct_matcher.irrep_position import IrrepPos
from rsspolymlp.utils.property import PropUtil

poscar_name = "./poscar_element2/POSCAR_112"
symutil = SymCell(poscar_name=poscar_name, symprec=1e-3)
st1 = symutil.refine_cell(standardize_cell=True)

irrep_pos = IrrepPos(symprec=1e-5)
start = time.time()
rep_pos1, sorted_elements1, order1 = irrep_pos.irrep_positions(
    st1.axis, st1.positions.T, st1.elements
)
el_time1 = round(time.time() - start, 5)
print("get irrep atomic positions:", (el_time1) * 1000)

print("--- Stucture 1 ---")
print(" - Axis:")
print(PropUtil(st1.axis.T, st1.positions).axis_to_abc)
print(" - Positions:")
print(np.round(st1.positions, 5))
print(" - Elements:")
print(st1.elements)
print(" - Irrep positions:")
print(rep_pos1.reshape(3, -1))
print(" - Recommended symprec:", order1)

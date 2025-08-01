import os

import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.strgen import StructureGenerator
from pypolymlp.utils.structure_utils import supercell_diagonal
from pypolymlp.utils.vasp_utils import write_poscar_file
from rsspolymlp.common.property import PropUtil


def gen_mlp_data(
    poscar,
    per_volume=1.0,
    disp_max=30,
    disp_grid=1,
    natom_lb=30,
    natom_ub=150,
    str_name=-1,
):
    os.makedirs("poscar", exist_ok=True)

    try:
        polymlp_st = Poscar(poscar).structure
    except IndexError:
        print(poscar, "failed")
        return

    objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
    least_distance = objprop.least_distance

    strgen = StructureGenerator(polymlp_st, natom_lb=natom_lb, natom_ub=natom_ub)
    if np.array(strgen._size).tolist() == [1, 1, 1]:
        n_atoms = int(strgen._supercell.n_atoms[0])
        if n_atoms * 8 <= natom_ub:
            strgen._size = np.array([2, 2, 2])
            strgen._supercell = supercell_diagonal(strgen.unitcell, strgen._size)
            strgen._supercell.axis_inv = np.linalg.inv(strgen._supercell.axis)

    with open("struct_size.yaml", "a") as f:
        print("- name:          ", poscar, file=f)
        print("  supercell_size:", np.array(strgen._size).tolist(), file=f)
        print("  n_atoms:       ", int(strgen._supercell.n_atoms[0]), file=f)

    disp_list = np.arange(disp_grid, disp_max + 0.0001, disp_grid)
    for disp_ratio in disp_list:
        disp = least_distance * disp_ratio / 100
        str_rand = strgen.random_single_structure(disp, vol_ratio=per_volume)

        _str_name = poscar.split("/")[str_name]
        poscar_path = f"poscar/{_str_name}_d{disp_ratio}_v{per_volume}"

        if not os.path.isfile(poscar_path):
            write_poscar_file(str_rand, poscar_path)
        else:
            print(f"{poscar_path} already exists.")

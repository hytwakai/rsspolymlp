import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.strgen import StructureGenerator
from pypolymlp.utils.vasp_utils import write_poscar_file
from rsspolymlp.common.property import PropUtil


def gen_mlp_data(
    poscar,
    per_volume=1.0,
    disp_max=40,
    disp_grid=2,
    natom_lb=30,
    natom_ub=150,
    str_name=-1,
):
    try:
        polymlp_st = Poscar(poscar).structure
    except IndexError:
        print(poscar, "failed")
        return

    objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
    least_distance = objprop.least_distance

    strgen = StructureGenerator(polymlp_st, natom_lb=natom_lb, natom_ub=natom_ub)
    with open("struct_size.yaml", "a") as f:
        print("- name:          ", poscar, file=f)
        print("  supercell_size:", np.array(strgen._size).tolist(), file=f)
        print("  n_atoms:       ", int(strgen._supercell.n_atoms[0]), file=f)

    per_volume = per_volume
    disp_list = np.arange(disp_grid, disp_max + 0.0001, disp_grid)
    for disp_ratio in disp_list:
        disp = least_distance * disp_ratio / 100
        str_rand = strgen.random_single_structure(disp, vol_ratio=per_volume)
        _str_name = poscar.split("/")[str_name]
        write_poscar_file(str_rand, f"poscar/{_str_name}_d{disp_ratio}_v{per_volume}")


if __name__ == "__main__":

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--poscars",
        type=str,
        nargs="+",
        required=True,
        help="Input POSCAR file(s) for structure generation",
    )
    parser.add_argument(
        "--per_volume",
        type=float,
        default=1.0,
        help="Volume scaling factor for generated structures",
    )
    parser.add_argument(
        "--disp_max",
        type=float,
        default=40,
        help="Maximum displacement ratio for structure generation",
    )
    parser.add_argument(
        "--disp_grid",
        type=float,
        default=2,
        help="Displacement ratio interval (step size)",
    )
    parser.add_argument(
        "--natom_lb",
        type=int,
        default=30,
        help="Minimum number of atoms in generated structure",
    )
    parser.add_argument(
        "--natom_ub",
        type=int,
        default=150,
        help="Maximum number of atoms in generated structure",
    )
    parser.add_argument(
        "--str_name",
        type=int,
        default=-1,
        help="Index for extracting structure name from POSCAR path",
    )
    args = parser.parse_args()

    os.makedirs("poscar", exist_ok=True)
    with open("struct_size.yaml", "w"):
        pass

    for poscar in args.poscars:
        gen_mlp_data(
            poscar=poscar,
            per_volume=args.per_volume,
            disp_max=args.disp_max,
            disp_grid=args.disp_grid,
            natom_lb=args.natom_lb,
            natom_ub=args.natom_ub,
            str_name=args.str_name,
        )

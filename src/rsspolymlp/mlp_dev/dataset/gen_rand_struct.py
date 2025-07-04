import argparse
import os
from contextlib import redirect_stdout

import numpy as np

from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.core.strgen import StructureGenerator
from pypolymlp.utils.vasp_utils import write_poscar_file
from rsspolymlp.common.property import PropUtil

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
parser.add_argument(
    "--dir_name",
    type=int,
    default=0,
    help="Index for extracting directory name from POSCAR path (0 to disable)",
)
args = parser.parse_args()

os.makedirs("random_struct/poscar", exist_ok=True)
os.chdir("random_struct")
with open("struct_size.yaml", "w"):
    pass

for poscar in args.poscars:
    try:
        polymlp_st = Poscar(poscar).structure
    except IndexError:
        print(poscar, "failed")
        continue
    objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
    n_atoms = len(polymlp_st.elements)
    axis_abc = objprop.abc
    least_distance = objprop.least_distance

    strgen = StructureGenerator(
        polymlp_st, natom_lb=args.natom_lb, natom_ub=args.natom_ub
    )
    with open("struct_size.yaml", "a") as f, redirect_stdout(f):
        print("- name:          ", poscar, file=f)
        print("  supercell_size:", np.array(strgen._size).tolist(), flush=True)
        print("  n_atoms:       ", int(strgen._supercell.n_atoms[0]), flush=True)

    per_volume = args.per_volume
    disp_list = np.arange(args.disp_grid, args.disp_max + 0.0001, args.disp_grid)
    for disp in disp_list:
        str_rand = strgen.random_single_structure(disp, vol_ratio=per_volume)
        _str_name = poscar.split("/")[args.str_name]
        if not args.dir_name == 0:
            _dir_name = f"/{poscar.split('/')[args.dir_name]}"
            os.makedirs(f"poscar{_dir_name}", exist_ok=True)
        else:
            _dir_name = ""
        write_poscar_file(
            str_rand, f"poscar{_dir_name}/{_str_name}_d{disp}_v{per_volume}"
        )

import re
from typing import Optional

from pypolymlp.core.data_format import PolymlpStructure

from rsspolymlp.common.interface_vasp import Poscar
from rsspolymlp.utils.spglib_utils import SymCell


def generate_primitive_cells(
    poscar_name: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    refine_cell: bool = False,
) -> tuple[list[PolymlpStructure], list[int]]:
    """
    Generate primitive cells of a given structure under different symmetry tolerances.

    Parameters
    ----------
    poscar_name : str, optional
        Path to a POSCAR file.
    polymlp_st : PolymlpStructure, optional
        PolymlpStructure object.
    symprec_set : list of float
        List of symmetry tolerances to use for identifying space group and primitive cell.

    Returns
    -------
    primitive_st_set : list of PolymlpStructure
        List of primitive cells determined from the given structure under each tolerance.
    spg_number_set : list of int
        Corresponding list of space group numbers for each primitive structure.
    """

    if poscar_name is not None and polymlp_st is None:
        polymlp_st = Poscar(poscar_name).structure
    elif polymlp_st is None:
        return [], []

    primitive_st_set = []
    spg_number_set = []
    for symprec in symprec_set:
        symutil = SymCell(st=polymlp_st, symprec=symprec)
        spg_str = symutil.get_spacegroup()
        spg_number = int(re.search(r"\((\d+)\)", spg_str).group(1))
        if spg_number in spg_number_set:
            continue
        else:
            try:
                if not refine_cell:
                    primitive_st = symutil.primitive_cell()
                else:
                    primitive_st = symutil.refine_cell()
            except TypeError:
                continue
            primitive_st_set.append(primitive_st)
            spg_number_set.append(spg_number)

    return primitive_st_set, spg_number_set

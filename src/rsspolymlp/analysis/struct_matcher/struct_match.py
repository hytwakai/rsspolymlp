import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from pypolymlp.utils.vasp_utils import write_poscar_file
from rsspolymlp.analysis.struct_matcher.irrep_position import IrrepPosition
from rsspolymlp.common.composition import compute_composition
from rsspolymlp.utils.spglib_utils import SymCell


@dataclass
class IrrepStructure:
    axis: np.ndarray
    positions: np.ndarray
    elements: np.ndarray
    element_count: Counter[str]
    spg_number: int


def struct_match(
    st_1_set: list[IrrepStructure],
    st_2_set: list[IrrepStructure],
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
) -> bool:

    struct_match = False
    for st_1 in st_1_set:
        for st_2 in st_2_set:
            if (
                struct_match is True
                or st_1.spg_number != st_2.spg_number
                or st_1.element_count != st_2.element_count
            ):
                continue

            axis_diff = st_1.axis - st_2.axis
            max_axis_diff = np.max(np.sum(axis_diff**2, axis=1))
            if max_axis_diff >= axis_tol:
                continue

            deltas = st_1.positions[:, None, :] - st_2.positions[None, :, :]
            deltas_flat = deltas.reshape(-1, deltas.shape[2])
            max_pos_error = np.min(np.max(np.abs(deltas_flat), axis=1))
            if max_pos_error < pos_tol:
                struct_match = True

    return struct_match


def generate_primitive_cells(
    poscar_name: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    symprec_set: list[float] = [1e-5],
) -> PolymlpStructure:

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
                primitive_st = symutil.primitive_cell()
            except TypeError:
                continue
            primitive_st_set.append(primitive_st)
            spg_number_set.append(spg_number)

    return primitive_st_set, spg_number_set


def generate_irrep_struct(
    primitive_st: PolymlpStructure,
    spg_number: int,
    symprec_irreps: list = [1e-5],
) -> IrrepStructure:

    irrep_positions = []
    for symprec_irrep in symprec_irreps:
        if isinstance(symprec_irrep, float):
            symprec_irrep = [symprec_irrep] * 3
        irrep_pos = IrrepPosition(symprec=symprec_irrep)
        _axis = primitive_st.axis.T
        _pos = primitive_st.positions.T
        _elements = primitive_st.elements
        rep_pos, sorted_elements = irrep_pos.irrep_positions(
            _axis, _pos, _elements, spg_number
        )
        irrep_positions.append(rep_pos)

    return IrrepStructure(
        axis=_axis,
        positions=np.stack(irrep_positions, axis=0),
        elements=sorted_elements,
        element_count=Counter(sorted_elements),
        spg_number=spg_number,
    )


def write_poscar_irrep_struct(irrep_st: IrrepStructure, file_name: str = "POSCAR"):
    axis = irrep_st.axis
    positions = irrep_st.positions[-1].reshape(3, -1)
    elements = irrep_st.elements
    comp_res = compute_composition(elements)
    polymlp_st = PolymlpStructure(
        axis.T,
        positions,
        comp_res.atom_counts,
        elements,
        comp_res.types,
    )
    write_poscar_file(polymlp_st, filename=file_name)

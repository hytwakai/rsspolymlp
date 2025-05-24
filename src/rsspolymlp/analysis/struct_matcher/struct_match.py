import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.utils.vasp_utils import write_poscar_file
from rsspolymlp.analysis.struct_matcher.irrep_position import IrrepPos
from rsspolymlp.analysis.struct_matcher.utils import IrrepUtil
from rsspolymlp.common.comp_ratio import compute_composition
from rsspolymlp.utils.spglib_utils import SymCell


@dataclass
class IrrepStructure:
    axis: np.ndarray
    positions: np.ndarray
    elements: np.ndarray
    element_count: Counter[str]


def struct_match(
    st_1: IrrepStructure,
    st_2: IrrepStructure,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
) -> bool:

    if st_1.element_count != st_2.element_count:
        return False

    axis_diff = st_1.axis - st_2.axis
    max_axis_diff = np.max(np.sum(axis_diff**2, axis=1))
    if max_axis_diff >= axis_tol:
        return False

    deltas = st_1.positions[:, None, :] - st_2.positions[None, :, :]
    deltas_flat = deltas.reshape(-1, deltas.shape[2])
    max_pos_error = np.min(np.max(np.abs(deltas_flat), axis=1))
    if max_pos_error >= pos_tol:
        return False

    return True


def generate_primitive_cell(
    poscar_name: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    elements: Optional[np.ndarray] = None,
    symprec: float = 1e-5,
) -> PolymlpStructure:

    if poscar_name is None and polymlp_st is None:
        comp_res = compute_composition(elements)
        polymlp_st = PolymlpStructure(
            axis,
            positions,
            comp_res.atom_counts,
            elements,
            comp_res.types,
        )

    if poscar_name is not None:
        symutil = SymCell(poscar_name=poscar_name, symprec=symprec)
    elif polymlp_st is not None:
        symutil = SymCell(st=polymlp_st, symprec=symprec)
    try:
        primitive_st = symutil.primitive_cell()
    except TypeError:
        return None, None
    
    spg_str = symutil.get_spacegroup()
    spg_number = int(re.search(r"\((\d+)\)", spg_str).group(1))

    return primitive_st, spg_number


def generate_irrep_struct(
    primitive_st: PolymlpStructure,
    spg_number: int,
    symprec_irreps: list = [1e-5],
) -> IrrepStructure:

    irrep_positions = []
    for symprec_irrep in symprec_irreps:
        irrep_pos = IrrepPos(symprec=symprec_irrep)
        _axis = primitive_st.axis
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
    )


def get_recommend_symprecs(
    primitive_st: PolymlpStructure,
    symprec_irrep: float = 1e-5,
):
    _pos = primitive_st.positions.T
    _elements = primitive_st.elements
    irrep_util = IrrepUtil(_pos, _elements, symprec=symprec_irrep)
    recommend_symprecs = irrep_util.recommended_symprec()

    return recommend_symprecs


def get_distance_cluster(
    polymlp_st: PolymlpStructure,
    symprec_irrep: float = 1e-5,
):
    _pos = polymlp_st.positions.T
    _elements = polymlp_st.elements
    irrep_util = IrrepUtil(_pos, _elements, symprec=symprec_irrep)
    distance_cluster = irrep_util.inter_cluster_diffs()

    return distance_cluster


def write_poscar_irrep_struct(irrep_st: IrrepStructure, file_name: str = "POSCAR"):
    axis = irrep_st.axis
    positions = irrep_st.positions[-1].reshape(3, -1)
    print(positions)
    elements = irrep_st.elements
    print(elements)
    comp_res = compute_composition(elements)
    polymlp_st = PolymlpStructure(
        axis.T,
        positions,
        comp_res.atom_counts,
        elements,
        comp_res.types,
    )
    write_poscar_file(polymlp_st, filename=file_name)

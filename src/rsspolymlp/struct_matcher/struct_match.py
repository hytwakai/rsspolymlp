from collections import Counter
from dataclasses import dataclass

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from rsspolymlp.struct_matcher.irrep_position import IrrepPos
from rsspolymlp.utils.spglib_utils import SymCell


@dataclass
class IrrepStruct:
    axis: np.ndarray
    positions: np.ndarray
    elements: np.ndarray
    element_count: Counter[str]
    recommend_symprecs: list[float]


def struct_match(
    st_1: IrrepStruct,
    st_2: IrrepStruct,
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


def get_irrep_positions(
    poscar_name: str = None,
    struct: PolymlpStructure = None,
    symprec_primitive: float = 1e-3,
    symprec_irreps: list = [1e-5],
) -> IrrepStruct:

    if poscar_name is not None:
        symutil = SymCell(poscar_name=poscar_name, symprec=symprec_primitive)
        struct = symutil.primitive_cell()

    irrep_positions = []
    for symprec_irrep in symprec_irreps:
        irrep_pos = IrrepPos(symprec=symprec_irrep)
        _axis = struct.axis
        _pos = struct.positions.T
        _elements = struct.elements
        rep_pos, sorted_elements, recommend_symprecs = irrep_pos.irrep_positions(
            _axis, _pos, _elements
        )
        irrep_positions.append(rep_pos)

    return IrrepStruct(
        axis=_axis,
        positions=np.stack(irrep_positions, axis=0),
        elements=sorted_elements,
        element_count=Counter(sorted_elements),
        recommend_symprecs=recommend_symprecs,
    )


def get_recommend_symprecs(
    poscar_name: str = None,
    struct: PolymlpStructure = None,
    symprec_primitive: float = 1e-3,
    symprec_irrep: float = 1e-5,
):
    if poscar_name is not None:
        symutil = SymCell(poscar_name=poscar_name, symprec=symprec_primitive)
        struct = symutil.primitive_cell()

    irrep_pos = IrrepPos(symprec=symprec_irrep, get_recommend_symprecs=True)
    _axis = struct.axis
    _pos = struct.positions.T
    _elements = struct.elements
    _, _, recommend_symprecs = irrep_pos.irrep_positions(_axis, _pos, _elements)

    return struct, recommend_symprecs


def get_distance_cluster(
    struct: PolymlpStructure,
    symprec_irrep: float = 1e-5,
):
    irrep_pos = IrrepPos(symprec=symprec_irrep, get_recommend_symprecs=True)
    _axis = struct.axis
    _pos = struct.positions.T
    _elements = struct.elements
    _, _, _ = irrep_pos.irrep_positions(_axis, _pos, _elements)

    return irrep_pos.distance_cluster

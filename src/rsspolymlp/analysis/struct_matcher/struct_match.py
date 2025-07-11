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
    """
    Determine whether two sets of IrrepStructure objects are structurally
    equivalent.

    This function compares all pairs of irreducible representations from the
    two input sets and checks if any pair matches within the specified lattice
    and position tolerances.
    Structures are compared only if they share the same space group number
    and identical element counts.

    Parameters
    ----------
    st_1_set : list of IrrepStructure
        First set of symmetry-reduced structures (e.g., from structure A).
    st_2_set : list of IrrepStructure
        Second set of symmetry-reduced structures (e.g., from structure B).
    axis_tol : float, default=0.01
        Tolerance for lattice vector differences, computed using the squared
        L2 norm along each axis.
    pos_tol : float, default=0.01
        Tolerance for atomic position differences. Computed as the minimum of
        the maximum absolute deviation among all pairwise differences.

    Returns
    -------
    bool
        True if a matching pair of structures is found under the given
        tolerances, False otherwise.
    """

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
            max_axis_diff = np.max(np.sqrt(np.sum(axis_diff**2, axis=1)))
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
) -> tuple[list[PolymlpStructure], list[int]]:
    """
    Generate primitive cells of a given structure under different symmetry tolerances.

    Parameters
    ----------
    poscar_name : str, optional
        Path to a POSCAR file.
    polymlp_st : PolymlpStructure, optional
        PolymlpStructure object.
    symprec_set : list of float, default=[1e-5]
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
    """
    Generate an IrrepStructure by computing irreducible atomic positions
    for a primitive structure under different symmetry tolerances.

    Parameters
    ----------
    primitive_st : PolymlpStructure
        Primitive structure.
    spg_number : int
        Space group number corresponding to the given primitive structure.
    symprec_irreps : list of float or list of 3-float lists, default=[1e-5]
        List of symmetry tolerances used to calculate irreducible representations.

    Returns
    -------
    IrrepStructure
        Object containing the standardized lattice, stacked irreducible positions,
        element list, element counts, and the space group number.
    """

    irrep_positions = []
    for symprec_irrep in symprec_irreps:
        if isinstance(symprec_irrep, float):
            symprec_irrep = [symprec_irrep] * 3

        _axis = primitive_st.axis.T
        _pos = primitive_st.positions.T
        _elements = primitive_st.elements

        volume = np.linalg.det(primitive_st.axis)
        standardized_axis = _axis / (volume ** (1 / 3))

        irrep_pos = IrrepPosition(symprec=symprec_irrep)
        rep_pos, sorted_elements = irrep_pos.irrep_positions(
            _axis, _pos, _elements, spg_number
        )
        irrep_positions.append(rep_pos)

    return IrrepStructure(
        axis=standardized_axis,
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

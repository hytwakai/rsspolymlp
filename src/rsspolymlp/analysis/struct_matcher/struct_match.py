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
    symprec_irreps: list


def struct_match(
    st_1_set: list[IrrepStructure],
    st_2_set: list[IrrepStructure],
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    spg_match: bool = True,
    verbose: bool = False,
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
    axis_d_min = None
    pos_d_min = None
    for st_1 in st_1_set:
        for st_2 in st_2_set:
            if struct_match or st_1.element_count != st_2.element_count:
                continue
            if spg_match and st_1.spg_number != st_2.spg_number:
                continue

            axis_d = (
                st_1.axis[:, None, :] - st_2.axis[None, :, :]
            )  # (N_symp1, N_symp2, 6)
            axis_d_flat = axis_d.reshape(-1, axis_d.shape[2])  # (N_symp1*N_symp2, 6)
            l2_norm = np.linalg.norm(axis_d_flat, axis=1)
            match_axis = l2_norm < axis_tol
            if not np.any(match_axis):
                continue

            pos_d = st_1.positions[:, None, :] - st_2.positions[None, :, :]
            pos_d_flat = pos_d.reshape(-1, pos_d.shape[2])
            max_abs = np.max(np.abs(pos_d_flat), axis=1)
            min_idx = np.argmin(max_abs[match_axis])

            axis_l2_norm = l2_norm[match_axis][min_idx]
            pos_max_abs = max_abs[match_axis][min_idx]
            i, j = divmod(np.where(match_axis)[0][min_idx], st_2.positions.shape[0])
            if pos_d_min is None or pos_d_min[0] > pos_max_abs:
                axis_d_min = [
                    axis_l2_norm,
                    [st_1.axis[i], st_2.axis[j], axis_d[i, j]],
                ]
                pos_d_min = [
                    pos_max_abs,
                    [
                        st_1.symprec_irreps[i],
                        st_1.positions[i],
                        st_2.symprec_irreps[j],
                        st_2.positions[j],
                        pos_d[i, j],
                    ],
                ]
            if pos_max_abs < pos_tol:
                struct_match = True

    if verbose:

        def log_axis_positions(symprec, axis, positions):
            print(" - symprec:", np.round(symprec, 5).tolist())
            print("   - Metric_tensor:", np.round(axis, 4).tolist())
            print("   - Positions:")
            for axis_tag, p in zip(
                ["a", "b", "c"], np.round(positions.reshape(3, -1), 3).tolist()
            ):
                formatted = ",".join(f"{val:6.3f}" for val in p)
                print(f"     - {axis_tag}: [{formatted}]")

        print("Structures:")
        for i, st_set in enumerate([st_1_set, st_2_set]):
            print(f" - Struct_No: {i+1}")
            for st in st_set:
                print(" - spg_number:", st.spg_number)
                for h, pos in enumerate(st.positions):
                    log_axis_positions(st.symprec_irreps[h], st.axis[h], pos)
        if axis_d_min is not None:
            print("")
            print("difference_log:")
            print(" - axis_l2_norm:", np.round(axis_d_min[0], 3))
            print(" - pos_max_abs", np.round(pos_d_min[0], 3))
            print(" - Structure_1:")
            log_axis_positions(pos_d_min[1][0], axis_d_min[1][0], pos_d_min[1][1])
            print(" - Structure_2:")
            log_axis_positions(pos_d_min[1][2], axis_d_min[1][1], pos_d_min[1][3])
            print(" - diffs:")
            log_axis_positions([], axis_d_min[1][2], pos_d_min[1][4])
        print("")
        print("Match:", struct_match)

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

    metric_tensors = []
    irrep_positions = []
    for symprec_irrep in symprec_irreps:
        if isinstance(symprec_irrep, float):
            symprec_irrep = [symprec_irrep] * 3

        _axis = primitive_st.axis.T
        _pos = primitive_st.positions.T
        _elements = primitive_st.elements

        irrep_pos = IrrepPosition(symprec=symprec_irrep)
        metric_tensor, rep_pos, sorted_elements = irrep_pos.irrep_positions(
            _axis, _pos, _elements, spg_number
        )
        metric_tensors.append(metric_tensor)
        irrep_positions.append(rep_pos)

    return IrrepStructure(
        axis=np.stack(metric_tensors, axis=0),
        positions=np.stack(irrep_positions, axis=0),
        elements=sorted_elements,
        element_count=Counter(sorted_elements),
        spg_number=spg_number,
        symprec_irreps=symprec_irreps,
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

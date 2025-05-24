import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from rsspolymlp.analysis.struct_matcher.struct_match import (
    IrrepStructure,
    generate_irrep_struct,
    generate_primitive_cell,
    get_recommend_symprecs,
    struct_match,
)
from rsspolymlp.common.property import PropUtil


@dataclass
class UniqueStructure:
    energy: float
    spg_list: list[str]
    irrep_struct: IrrepStructure
    original_axis: np.ndarray
    original_positions: np.ndarray
    original_elements: np.ndarray
    axis_abc: np.ndarray
    n_atoms: int
    volume: float
    least_distance: float
    input_poscar: Optional[str]
    dup_count: int = 1


def generate_unique_struct(
    energy: float,
    spg_list: list[str],
    poscar_name: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    elements: Optional[np.ndarray] = None,
    symprec: float = 1e-2,
) -> UniqueStructure:
    """
    Generate a UniqueStructure object from various structure inputs.

    Parameters
    ----------
    energy : float
        Enthalpy or energy value of the structure.
    spg_list : list of str
        List of space group labels.
    poscar_name : str, optional
        Path to POSCAR file.
    polymlp_st : PolymlpStructure, optional
        Already parsed structure object.
    axis : np.ndarray, optional
        3x3 lattice vectors (used if no POSCAR is provided).
    positions : np.ndarray, optional
        Fractional atomic positions (N x 3).
    elements : np.ndarray, optional
        Element symbols (N).

    Returns
    -------
    UniqueStructure
        A standardized structure object for uniqueness evaluation.
    """
    if poscar_name is None and polymlp_st is None:
        _axis = axis
        _positions = positions
        _elements = elements
        primitive_st, spg_number = generate_primitive_cell(
            axis=_axis,
            positions=_positions,
            elements=_elements,
            symprec=symprec,
        )
    else:
        if polymlp_st is None:
            polymlp_st = Poscar(poscar_name).structure
        _axis = polymlp_st.axis.T
        _positions = polymlp_st.positions.T
        _elements = polymlp_st.elements
        primitive_st, spg_number = generate_primitive_cell(
            polymlp_st=polymlp_st, symprec=symprec
        )
    if primitive_st is None:
        return None

    objprop = PropUtil(_axis, _positions)

    recommend_symprecs = get_recommend_symprecs(primitive_st, symprec_irrep=1e-5)
    symprec_list = [1e-5] + recommend_symprecs
    irrep_struct = generate_irrep_struct(
        primitive_st, spg_number, symprec_irreps=symprec_list
    )

    return UniqueStructure(
        energy=energy,
        spg_list=spg_list,
        irrep_struct=irrep_struct,
        original_axis=_axis,
        original_positions=_positions,
        original_elements=_elements,
        axis_abc=objprop.axis_to_abc,
        n_atoms=int(len(_elements)),
        volume=objprop.volume,
        least_distance=objprop.least_distance,
        input_poscar=poscar_name,
    )


class UniqueStructureAnalyzer:

    def __init__(self):
        self.unique_str = []  # List to store unique structures
        self.unique_str_prop = []  # List to store unique structure properties

    def identify_duplicate_struct(
        self,
        unique_struct: UniqueStructure,
        other_properties: Optional[dict] = None,
        use_energy_spg_check: bool = False,
        energy_diff: float = 1e-8,
    ):
        """
        Identify and manage duplicate structures based on one or both of the following criteria:

        1. Energy + space group similarity (optional):
        If `use_energy_spg_check=True`, a structure is considered a duplicate if its energy
        is within `energy_diff` of an existing structure, and it shares at least one space group.
        Note: This method does not distinguish between chiral structures, as enantiomorphs
        can exist with identical energy and space group.

        2. Irreducible structural representation:
        A structure is considered a duplicate if it matches an existing structure based on
        irreducible position equivalence.

        Parameters
        ----------
        unique_struct : UniqueStructure
            The structure to be compared and registered if unique.
        other_properties : dict, optional
            Additional metadata associated with the structure.
        use_energy_spg_check : bool
            Whether to enable duplicate detection using energy and space group similarity.
        energy_diff : float
            Energy tolerance used in energy-based duplicate detection.

        Returns
        -------
        is_unique : bool
            True if the structure is unique.
        is_change_struct : bool
            True if the existing structure was replaced due to higher symmetry.
        """

        is_unique = True
        is_change_struct = False
        _energy = unique_struct.energy
        _spg_list = unique_struct.spg_list
        _irrep_struct = unique_struct.irrep_struct
        if other_properties is None:
            other_properties = {}

        for idx, ndstr in enumerate(self.unique_str):
            if use_energy_spg_check:
                if abs(ndstr.energy - _energy) < energy_diff and any(
                    spg in _spg_list for spg in ndstr.spg_list
                ):
                    is_unique = False
                    if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                        ndstr.spg_list
                    ):
                        is_change_struct = True
                    break
            if struct_match(ndstr.irrep_struct, _irrep_struct):
                is_unique = False
                if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                    ndstr.spg_list
                ):
                    is_change_struct = True
                break

        if not is_unique:
            # Update duplicate count and replace with better data if necessary
            if is_change_struct:
                unique_struct.dup_count = self.unique_str[idx].dup_count
                self.unique_str[idx] = unique_struct
                self.unique_str_prop[idx] = other_properties
            self.unique_str[idx].dup_count += 1
        else:
            self.unique_str.append(unique_struct)
            self.unique_str_prop.append(other_properties)

        return is_unique, is_change_struct

    def _extract_spg_count(self, spg_list):
        """Extract and sum space group counts from a list of space group strings."""
        return sum(
            int(re.search(r"\((\d+)\)", s).group(1))
            for s in spg_list
            if re.search(r"\((\d+)\)", s)
        )

    def _initialize_unique_structs(
        self, unique_structs, unique_str_prop: Optional[list[dict]] = None
    ):
        """Initialize unique structures and their associated properties."""
        self.unique_str = unique_structs
        if unique_str_prop is None:
            self.unique_str_prop = [{} for _ in unique_structs]
        else:
            self.unique_str_prop = unique_str_prop

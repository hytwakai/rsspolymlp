import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from rsspolymlp.analysis.struct_matcher.struct_match import (
    IrrepStructure,
    get_irrep_positions,
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
    input_poscar: str
    dup_count: int = 1


def generate_unique_struct(
    energy: float,
    spg_list: list[str],
    poscar_name: str,
    original_polymlp_st: Optional[PolymlpStructure] = None,
):
    if original_polymlp_st is None:
        original_polymlp_st = Poscar(poscar_name).structure
    _st = original_polymlp_st
    objprop = PropUtil(_st.axis.T, _st.positions.T)

    struct, recommend_symprecs = get_recommend_symprecs(
        poscar_name=poscar_name, symprec_irrep=1e-5
    )
    symprec_list = [1e-5]
    symprec_list.extend(recommend_symprecs)
    irrep_struct = get_irrep_positions(struct=struct, symprec_irreps=symprec_list)

    return UniqueStructure(
        energy=energy,
        spg_list=spg_list,
        irrep_struct=irrep_struct,
        original_axis=_st.axis.T,
        original_positions=_st.positions.T,
        original_elements=_st.elements,
        axis_abc=objprop.axis_to_abc,
        n_atoms=int(len(_st.elements)),
        volume=objprop.volume,
        least_distance=objprop.least_distance,
        input_poscar=poscar_name,
    )


class UniqueStructureAnalyzer:

    def __init__(self):
        """Initialize data structures for sorting structures."""
        self.unique_str = []  # List to store unique structures
        self.unique_str_prop = []  # List to store unique structure properties

    def identify_duplicate_struct(
        self,
        unique_struct: UniqueStructure,
        other_properties: Optional[dict] = None,
        energy_diff=1e-8,
    ):
        """
        Identify and manage duplicate structures based on energy, space group,
        and irreducible strcture representation.

        A structure is considered a duplicate if it satisfies either of
        the following conditions, evaluated in order:

        1. Its energy is within 1e-8 (default) of an existing structure,
        and it shares at least one space group with that structure.
        2. It matches an existing structure based on irreducible representation.

        If a duplicate is found:
        - The duplicate count is incremented.
        - The structure is replaced if it has higher space group symmetry.

        If no duplicate is found, the structure is added as a new unique entry.
        """

        is_unique = True
        is_change_struct = False
        _energy = unique_struct.energy
        _spg_list = unique_struct.spg_list
        _irrep_struct = unique_struct.irrep_struct
        if other_properties is None:
            other_properties = {}

        for idx, ndstr in enumerate(self.unique_str):
            if abs(ndstr.energy - _energy) < energy_diff and any(
                spg in _spg_list for spg in ndstr.spg_list
            ):
                is_unique = False
                if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                    ndstr.spg_list
                ):
                    is_change_struct = True
                break
            elif struct_match(ndstr.irrep_struct, _irrep_struct):
                is_unique = False
                if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                    ndstr.spg_list
                ):
                    is_change_struct = True
                break

        if not is_unique:
            # Update duplicate count and replace with better data if necessary
            if is_change_struct:
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

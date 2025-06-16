import re
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.interface_vasp import Poscar
from rsspolymlp.analysis.struct_matcher.struct_match import (
    IrrepStructure,
    generate_irrep_struct,
    generate_primitive_cells,
    struct_match,
)
from rsspolymlp.analysis.struct_matcher.utils import get_recommend_symprecs
from rsspolymlp.common.composition import compute_composition
from rsspolymlp.common.property import PropUtil


@dataclass
class UniqueStructure:
    irrep_struct_set: list[IrrepStructure]
    recommend_symprecs: list[float]
    original_structure: PolymlpStructure
    axis_abc: np.ndarray
    n_atoms: int
    volume: float
    least_distance: float
    energy: Optional[float]
    spg_list: Optional[list[str]]
    input_poscar: Optional[str]
    dup_count: int = 1


def generate_unique_struct(
    poscar_name: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    elements: Optional[np.ndarray] = None,
    energy: Optional[float] = None,
    spg_list: Optional[list[str]] = None,
    symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
) -> UniqueStructure:
    """
    Generate a UniqueStructure object from various structure inputs.

    Parameters
    ----------
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
    energy : float
        Enthalpy or energy value of the structure.
    spg_list : list of str
        List of space group labels.

    Returns
    -------
    UniqueStructure
        A standardized structure object for uniqueness evaluation.
    """
    if poscar_name is None and polymlp_st is None:
        comp_res = compute_composition(elements)
        polymlp_st = PolymlpStructure(
            axis,
            positions,
            comp_res.atom_counts,
            elements,
            comp_res.types,
        )
    else:
        if polymlp_st is None:
            polymlp_st = Poscar(poscar_name).structure

    primitive_st_set, spg_number_set = generate_primitive_cells(
        polymlp_st=polymlp_st,
        symprec_set=symprec_set,
    )
    if primitive_st_set == []:
        return None

    irrep_struct_set = []
    for i, primitive_st in enumerate(primitive_st_set):
        recommend_symprecs = get_recommend_symprecs(primitive_st)
        irrep_struct = generate_irrep_struct(
            primitive_st,
            spg_number_set[i],
            symprec_irreps=[1e-5] + recommend_symprecs,
        )
        irrep_struct_set.append(irrep_struct)

    objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)

    return UniqueStructure(
        irrep_struct_set=irrep_struct_set,
        recommend_symprecs=recommend_symprecs,
        original_structure=polymlp_st,
        axis_abc=objprop.abc,
        n_atoms=int(len(polymlp_st.elements)),
        volume=objprop.volume,
        least_distance=objprop.least_distance,
        energy=energy,
        spg_list=spg_list,
        input_poscar=poscar_name,
    )


def generate_unique_structs(
    rss_results, use_joblib=True, num_process=-1, backend="loky"
) -> list[UniqueStructure]:
    if use_joblib:
        unique_structs = joblib.Parallel(n_jobs=num_process, backend=backend)(
            joblib.delayed(generate_unique_struct)(
                res["poscar"],
                res["structure"],
                energy=res["energy"],
                spg_list=res["spg_list"],
            )
            for res in rss_results
        )
    else:
        unique_structs = []
        for res in rss_results:
            unique_structs.append(
                generate_unique_struct(
                    res["poscar"],
                    res["structure"],
                    energy=res["energy"],
                    spg_list=res["spg_list"],
                )
            )
    unique_structs = [s for s in unique_structs if s is not None]
    return unique_structs


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
        _irrep_struct_set = unique_struct.irrep_struct_set
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
            if struct_match(ndstr.irrep_struct_set, _irrep_struct_set):
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

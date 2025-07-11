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
from rsspolymlp.common.convert_dict import polymlp_struct_to_dict
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
    pressure: Optional[float]
    input_poscar: Optional[str]
    struct_no: Optional[int]
    dup_count: int = 1


class UniqueStructureAnalyzer:

    def __init__(self):
        self.unique_str = []  # List to store unique structures
        self.unique_str_prop = []  # List to store unique structure properties
        self.unique_str_keep = []
        self.unique_str_prop_keep = []

    def identify_duplicate_struct(
        self,
        unique_struct: UniqueStructure,
        other_properties: Optional[dict] = None,
        use_energy_spg_check: bool = False,
        energy_diff: float = 1e-8,
        axis_tol: float = 0.01,
        pos_tol: float = 0.01,
        keep_unique: bool = False,
    ):
        """
        Identify and manage duplicate structures based on one or both of the following criteria:

        1. Energy + space group similarity (optional):
            If `use_energy_spg_check=True`, a structure is considered a duplicate if its energy
            is within energy_diff of an existing structure, and it shares at least one space group.
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

        for idx, _ndstr in enumerate(self.unique_str):
            targets = self.unique_str_keep[idx] if keep_unique else [_ndstr]

            for ndstr in targets:
                if use_energy_spg_check:
                    same_energy = abs(ndstr.energy - _energy) < energy_diff
                    shared_spg = any(spg in _spg_list for spg in ndstr.spg_list)

                    if same_energy and shared_spg:
                        is_unique = False
                        if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                            ndstr.spg_list
                        ):
                            is_change_struct = True
                        break

                if struct_match(
                    ndstr.irrep_struct_set,
                    _irrep_struct_set,
                    axis_tol=axis_tol,
                    pos_tol=pos_tol,
                ):
                    is_unique = False
                    if self._extract_spg_count(_spg_list) > self._extract_spg_count(
                        ndstr.spg_list
                    ):
                        is_change_struct = True
                    break

            if not is_unique:
                break

        if not is_unique:
            if is_change_struct:
                # Update duplicate count and replace with better data if necessary
                unique_struct.dup_count = self.unique_str[idx].dup_count
                self.unique_str[idx] = unique_struct
                self.unique_str_prop[idx] = other_properties
            self.unique_str[idx].dup_count += 1
            if keep_unique:
                self.unique_str_keep[idx].append(unique_struct)
                self.unique_str_prop_keep[idx].append(other_properties)
        else:
            self.unique_str.append(unique_struct)
            self.unique_str_prop.append(other_properties)
            if keep_unique:
                self.unique_str_keep.append([unique_struct])
                self.unique_str_prop_keep.append([other_properties])

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


def generate_unique_struct(
    poscar_name: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    elements: Optional[np.ndarray] = None,
    energy: Optional[float] = None,
    spg_list: Optional[list[str]] = None,
    pressure: Optional[float] = None,
    struct_no: Optional[int] = None,
    symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
) -> UniqueStructure:
    """
    Generate a UniqueStructure object.

    Parameters
    ----------
    poscar_name : str, optional
        Path to a POSCAR file.
    polymlp_st : PolymlpStructure, optional
        PolymlpStructure object. If not provided, this tries to construct it
        from `poscar_name`, or from `axis`, `positions`, and `elements`.
    axis : np.ndarray, shape (3, 3), optional
        Lattice vectors of the structure.
    positions : np.ndarray, shape (N, 3), optional
        Fractional atomic positions.
    elements : np.ndarray or list of str, shape (N,), optional
        Element symbols for each atom.
    energy : float, optional
        Total energy or enthalpy of the structure.
    spg_list : list of str, optional
        List of space group symbols or labels.
    pressure : float, optional
        Pressure term (in GPa)
    struct_no: int, optional
        Structure identifier (e.g., structure number)
    symprec_set : list of float, default=[1e-5, 1e-4, 1e-3, 1e-2]
        Symmetry tolerances used to determine distinct primitive cells.

    Returns
    -------
    UniqueStructure
        A structure object for uniqueness evaluation.
    """
    if poscar_name is None and polymlp_st is None:
        comp_res = compute_composition(elements)
        polymlp_st = PolymlpStructure(
            axis.T,
            positions.T,
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
        pressure=pressure,
        input_poscar=poscar_name,
        struct_no=struct_no,
    )


def generate_unique_structs(
    rss_results, use_joblib=True, num_process=-1, backend="loky"
) -> list[UniqueStructure]:
    """
    Generate a list of UniqueStructure objects from the given RSS results.

    Parameters
    ----------
    rss_results : list of dict
        A list of dictionaries, where each dictionary contains a single structure information.
        Each dictionary must include the following keys:
            - "poscar": structure in POSCAR format (str or Poscar object)
            - "structure": PolymlpStructure object
            - "energy": total energy (float)
            - "spg_list": list of space group symbols identified under multiple tolerances
            - "pressure" (optional): pressure term (in GPa)
            - "struct_no" (optional): structure identifier (e.g., structure number)
    use_joblib : bool, default=True
        Whether to use joblib.Parallel for parallel processing.
    num_process : int, default=-1
        The number of parallel jobs. -1 means using all available processors.
    backend : str, default="loky"
        Backend used by joblib.

    Returns
    -------
    unique_structs : list of UniqueStructure
        A list of UniqueStructure objects.
    """
    if use_joblib:
        unique_structs = joblib.Parallel(n_jobs=num_process, backend=backend)(
            joblib.delayed(generate_unique_struct)(
                poscar_name=res["poscar"],
                polymlp_st=res["structure"],
                energy=res["energy"],
                spg_list=res["spg_list"],
                pressure=res.get("pressure", None),
                struct_no=res.get("struct_no", None),
            )
            for res in rss_results
        )
    else:
        unique_structs = []
        for res in rss_results:
            unique_structs.append(
                generate_unique_struct(
                    poscar_name=res["poscar"],
                    polymlp_st=res["structure"],
                    energy=res["energy"],
                    spg_list=res["spg_list"],
                    pressure=res.get("pressure", None),
                    struct_no=res.get("struct_no", None),
                )
            )
    unique_structs = [s for s in unique_structs if s is not None]
    return unique_structs


def log_unique_structures(
    file_name,
    unique_structs,
    is_ghost_minima,
    pressure=None,
    unique_struct_iters=None,
):
    struct_num_max = max(
        (_s.struct_no for _s in unique_structs if _s.struct_no is not None), default=0
    )
    for _s in unique_structs:
        if _s.struct_no is None:
            struct_num_max += 1
            _s.struct_no = struct_num_max

    for i in range(len(unique_structs)):
        if not is_ghost_minima[i]:
            energy_min = unique_structs[i].energy
            break

    rss_results = []
    with open(file_name, "a") as f:
        print("unique_structures:", file=f)
        for idx, _str in enumerate(unique_structs):
            e_diff = round((_str.energy - energy_min) * 1000, 2)
            print(f"  - struct_No: {_str.struct_no}", file=f)
            print(f"    poscar_name: {_str.input_poscar}", file=f)
            print(f"    energy_diff_meV_per_atom: {e_diff}", file=f)
            print(f"    duplicates: {_str.dup_count}", file=f)
            print(f"    enthalpy: {_str.energy}", file=f)
            print(f"    axis: {_str.axis_abc}", file=f)
            print(
                f"    positions: {_str.original_structure.positions.T.tolist()}", file=f
            )
            print(f"    elements: {_str.original_structure.elements}", file=f)
            print(f"    space_group: {_str.spg_list}", file=f)

            info = [
                f"{_str.n_atoms} atom",
                f"distance {round(_str.least_distance, 3)} (Ang.)",
                f"volume {round(_str.volume, 2)} (A^3/atom)",
            ]
            if unique_struct_iters is not None:
                info.append(f"iteration {unique_struct_iters[idx]}")
            print(f"    other_info: {' / '.join(info)}", file=f)

            if is_ghost_minima[idx]:
                print("    ghost_minima_flag: true", file=f)

            _res = {}
            _res["poscar"] = _str.input_poscar
            polymlp_st = _str.original_structure
            polymlp_st_dict = polymlp_struct_to_dict(polymlp_st)
            _res["structure"] = polymlp_st_dict
            _res["energy"] = _str.energy
            _res["pressure"] = pressure
            _res["spg_list"] = _str.spg_list
            _res["struct_no"] = _str.struct_no
            _res["is_ghost_minima"] = bool(is_ghost_minima[idx])
            rss_results.append(_res)

    comp_res = compute_composition(unique_structs[0].original_structure.elements)

    rss_result_all = {
        "elements": comp_res.unique_elements.tolist(),
        "comp_ratio": comp_res.comp_ratio,
        "pressure": pressure,
        "rss_results": rss_results,
    }

    return rss_result_all


def log_all_unique_structures(
    file_name,
    unique_structs,
):
    rss_results = []
    with open(file_name, "a") as f:
        print("unique_structures:", file=f)
        for idx1, _str in enumerate(unique_structs):
            print(f"  - struct_No: {idx1+1}", file=f)
            print("    structures:", file=f)
            for idx2, _str in enumerate(unique_structs[idx1]):
                print(f"    - sub_struct_No: '{idx1+1}_{idx2+1}'", file=f)
                print(f"      poscar_name: {_str.input_poscar}", file=f)
                print(f"      pressure: {_str.pressure}", file=f)
                print(f"      enthalpy: {_str.energy}", file=f)
                print(f"      axis: {_str.axis_abc}", file=f)
                print(
                    f"      positions: {_str.original_structure.positions.T.tolist()}",
                    file=f,
                )
                print(f"      elements: {_str.original_structure.elements}", file=f)
                print(f"      space_group: {_str.spg_list}", file=f)

                info = [
                    f"{_str.n_atoms} atom",
                    f"distance {round(_str.least_distance, 3)} (Ang.)",
                    f"volume {round(_str.volume, 2)} (A^3/atom)",
                ]
                print(f"      other_info: {' / '.join(info)}", file=f)

                _res = {}
                _res["poscar"] = _str.input_poscar
                polymlp_st = _str.original_structure
                polymlp_st_dict = polymlp_struct_to_dict(polymlp_st)
                _res["structure"] = polymlp_st_dict
                _res["energy"] = _str.energy
                _res["pressure"] = None
                _res["spg_list"] = _str.spg_list
                _res["struct_no"] = f"{idx1 + 1}_{idx2+1}"
                _res["is_ghost_minima"] = False
                rss_results.append(_res)

    comp_res = compute_composition(unique_structs[0][0].original_structure.elements)

    rss_result_all = {
        "elements": comp_res.unique_elements.tolist(),
        "comp_ratio": comp_res.comp_ratio,
        "pressure": None,
        "rss_results": rss_results,
    }

    return rss_result_all

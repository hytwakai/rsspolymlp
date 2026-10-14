from collections import Counter
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
from pypolymlp.core.data_format import PolymlpStructure

from rsspolymlp.analysis.struct_matcher.position_allrep import PositionRepGenerator
from rsspolymlp.analysis.struct_matcher.utils import generate_primitive_cells
from rsspolymlp.common.composition import compute_composition
from rsspolymlp.common.interface_vasp import Poscar
from rsspolymlp.common.property import PropUtil


@dataclass
class AllStructRep:
    axis: np.ndarray
    positions: np.ndarray
    elements: np.ndarray
    element_count: Counter[str]
    spg_number: int


@dataclass
class AllStructReps:
    allrep_list: list[AllStructRep]
    original_polymlp_st: PolymlpStructure
    spg_list: list[str]
    primitive_symprecs: list[float]
    standardize_axis: bool
    original_axis: bool
    refine_cell: bool
    struct_path: Optional[str]
    struct_tag: Optional[int]
    dupstr_paths: set[str]


def generate_allrep(
    polymlp_st: PolymlpStructure,
    spg_number: int,
    standardize_axis: bool = False,
    original_axis: bool = False,
) -> AllStructRep:
    """
    Parameters
    ----------
    polymlp_st : PolymlpStructure
    spg_number : int
        Space group number corresponding to the given structure.
    symprec : float or list of 3-float lists.
        List of symmetry tolerances used to calculate representations.
    """
    _axis = polymlp_st.axis.T
    _pos = polymlp_st.positions.T
    _elements = polymlp_st.elements

    generator = PositionRepGenerator(
        standardize_axis=standardize_axis,
        original_axis=original_axis,
    )
    metric_tensor, all_positions, sorted_elements = (
        generator.get_all_structure_representations(_axis, _pos, _elements, spg_number)
    )
    return AllStructRep(
        axis=metric_tensor,
        positions=all_positions,
        elements=sorted_elements,
        element_count=Counter(sorted_elements),
        spg_number=spg_number,
    )


def generate_allreps(
    poscar_path: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    elements: Optional[np.ndarray] = None,
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    standardize_axis: bool = False,
    original_axis: bool = False,
    refine_cell: bool = False,
    struct_path: Optional[str] = None,
    struct_tag: Optional[str] = None,
    dupstr_paths: Optional[set[str]] = None,
) -> AllStructReps:
    if struct_path is None and poscar_path is None:
        raise ValueError("Set struct_path or poscar_path")
    elif struct_path is None:
        struct_path = poscar_path

    if poscar_path is None and polymlp_st is None:
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
            polymlp_st = Poscar(poscar_path).structure
    objprop = PropUtil(polymlp_st.axis.T, polymlp_st.positions.T)
    spg_list = objprop.analyze_space_group(polymlp_st.elements)

    primitive_st_set, spg_number_set = generate_primitive_cells(
        polymlp_st=polymlp_st,
        symprec_set=primitive_symprecs,
        refine_cell=refine_cell,
    )
    if primitive_st_set == []:
        return None

    allrep_list = []
    for i, primitive_st in enumerate(primitive_st_set):
        allrep = generate_allrep(
            primitive_st,
            spg_number_set[i],
            standardize_axis=standardize_axis,
            original_axis=original_axis,
        )
        allrep_list.append(allrep)

    if dupstr_paths is None:
        dupstr_paths = set([struct_path])
    return AllStructReps(
        allrep_list=allrep_list,
        original_polymlp_st=polymlp_st,
        spg_list=spg_list,
        primitive_symprecs=primitive_symprecs,
        standardize_axis=standardize_axis,
        original_axis=original_axis,
        refine_cell=refine_cell,
        struct_path=struct_path,
        struct_tag=struct_tag,
        dupstr_paths=dupstr_paths,
    )


def generate_allreps_parallel(
    struct_lists: list[dict],
    num_process: int = -1,
    backend: str = "loky",
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    standardize_axis: bool = False,
    original_axis: bool = False,
    refine_cell: bool = False,
) -> list[AllStructReps]:
    """
    Generate a list of ReducedStructReps objects from the given RSS results.

    Parameters
    ----------
    struct_lists : list of dict
        A list of dictionaries, where each dictionary contains a single structure information.
        Each dictionary must include "struct_path" keys:
            - "struct_path": path of POSCAR format file
        Optional keys:
            - "structure": PolymlpStructure object
            - "struct_tag" (optional): structure identifier (e.g., structure number)
    num_process : int, default=-1
        The number of parallel jobs. -1 means using all available processors.
    backend : str, default="loky"
        Backend used by joblib.
    primitive_symprecs : list of float, default=[1e-5, 1e-4, 1e-3, 1e-2]
        Symmetry tolerances used to determine distinct primitive cells.
    """
    if num_process == 1:
        redreps_list = []
        for res in struct_lists:
            redreps_list.append(
                generate_allreps(
                    poscar_path=res["struct_path"],
                    polymlp_st=res.get("structure", None),
                    primitive_symprecs=primitive_symprecs,
                    standardize_axis=standardize_axis,
                    original_axis=original_axis,
                    refine_cell=refine_cell,
                    struct_tag=res.get("struct_tag", None),
                    dupstr_paths=res.get("dupstr_paths", None),
                )
            )
    else:
        redreps_list = joblib.Parallel(n_jobs=num_process, backend=backend)(
            joblib.delayed(generate_allreps)(
                poscar_path=res["struct_path"],
                polymlp_st=res.get("structure", None),
                primitive_symprecs=primitive_symprecs,
                standardize_axis=standardize_axis,
                original_axis=original_axis,
                refine_cell=refine_cell,
                struct_path=res["struct_path"],
                struct_tag=res.get("struct_tag", None),
                dupstr_paths=res.get("dupstr_paths", None),
            )
            for res in struct_lists
        )

    redreps_list = [s for s in redreps_list if s is not None]
    return redreps_list

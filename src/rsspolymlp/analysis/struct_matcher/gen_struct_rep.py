import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np

from pypolymlp.core.data_format import PolymlpStructure
from rsspolymlp.analysis.struct_matcher.position_redrep import PositionRepReducer
from rsspolymlp.common.composition import compute_composition
from rsspolymlp.common.interface_vasp import Poscar
from rsspolymlp.common.property import PropUtil
from rsspolymlp.utils.spglib_utils import SymCell


@dataclass
class StructRep:
    reduced_reps: dict
    all_reps: dict
    elements: np.ndarray
    element_count: Counter[str]
    spg_number: int


@dataclass
class StructReps:
    struct_rep_list: list[StructRep]
    original_polymlp_st: PolymlpStructure
    spg_list: list[str]
    primitive_symprecs: list[float]
    redrep_symprecs: list[float]
    standardize_axis: bool
    original_axis: bool
    cartesian_coords: bool
    refine_cell: bool
    struct_path: Optional[str]
    struct_tag: Optional[int]
    dupstr_paths: set[str]


def generate_struct_rep(
    st: PolymlpStructure,
    spg_number: int,
    symprec_set: list = [1e-4, 1e-2, 1e-1],
    standardize_axis: bool = False,
    original_axis: bool = False,
    cartesian_coords: bool = True,
) -> StructRep:
    """
    Generate an StructRep by computing irreducible atomic positions
    for a refined structure under different symmetry tolerances.

    Parameters
    ----------
    st : PolymlpStructure
    spg_number : int
        Space group number corresponding to the given structure.
    symprec_set : list of float or list of 3-float lists
        List of symmetry tolerances used to calculate irreducible representations.
    """
    reduced_reps = {"metric_tensors": [], "positions": [], "symprec": []}
    all_reps = {"metric_tensors": [], "positions": [], "symprec": []}
    for symprec in symprec_set:
        if isinstance(symprec, float):
            input_symprec = [symprec] * 3
        reducer = PositionRepReducer(
            symprec=input_symprec,
            standardize_axis=standardize_axis,
            original_axis=original_axis,
            cartesian_coords=cartesian_coords,
        )
        metric_tensor, red_posrep, all_posrep, sorted_elements = (
            reducer.get_reduced_structure_representation(
                st.axis.T, st.positions.T, st.elements, spg_number
            )
        )

        unique_metric = True
        for i, mt_ref in enumerate(reduced_reps["metric_tensors"]):
            diffs = np.abs(mt_ref - metric_tensor)
            if np.all(diffs < 1e-4):
                unique_metric = False
                break
        unique_pos = True
        if not unique_metric and len(red_posrep) > 0:
            for i, ps_ref in enumerate(reduced_reps["positions"]):
                diffs = np.abs(ps_ref - red_posrep)
                if np.all(diffs < 1e-4):
                    unique_pos = False
                    break
        if unique_metric:
            all_reps["metric_tensors"].append(metric_tensor)
            all_reps["positions"].append(all_posrep)
            all_reps["symprec"].append(symprec)
        if unique_metric or unique_pos:
            reduced_reps["metric_tensors"].append(metric_tensor)
            reduced_reps["positions"].append(red_posrep)
            reduced_reps["symprec"].append(symprec)

    return StructRep(
        reduced_reps=reduced_reps,
        all_reps=all_reps,
        elements=sorted_elements,
        element_count=Counter(sorted_elements),
        spg_number=spg_number,
    )


def generate_struct_reps(
    poscar_path: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    axis: Optional[np.ndarray] = None,
    positions: Optional[np.ndarray] = None,
    elements: Optional[np.ndarray] = None,
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    redrep_symprecs: list[float] = [1e-4, 1e-2, 1e-1],
    standardize_axis: bool = False,
    original_axis: bool = False,
    cartesian_coords: bool = True,
    refine_cell: bool = False,
    struct_path: Optional[str] = None,
    struct_tag: Optional[str] = None,
    dupstr_paths: Optional[set[str]] = None,
) -> StructReps:
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

    struct_rep_list = []
    for i, primitive_st in enumerate(primitive_st_set):
        redrep_symprecs = sorted(
            redrep_symprecs,
            key=lambda x: x if isinstance(x, (int, float)) else sum(x) / len(x),
        )
        struct_rep = generate_struct_rep(
            primitive_st,
            spg_number_set[i],
            symprec_set=redrep_symprecs,
            standardize_axis=standardize_axis,
            original_axis=original_axis,
            cartesian_coords=cartesian_coords,
        )
        struct_rep_list.append(struct_rep)

    if dupstr_paths is None:
        dupstr_paths = set([struct_path])
    return StructReps(
        struct_rep_list=struct_rep_list,
        original_polymlp_st=polymlp_st,
        spg_list=spg_list,
        primitive_symprecs=primitive_symprecs,
        redrep_symprecs=redrep_symprecs,
        standardize_axis=standardize_axis,
        original_axis=original_axis,
        cartesian_coords=cartesian_coords,
        refine_cell=refine_cell,
        struct_path=struct_path,
        struct_tag=struct_tag,
        dupstr_paths=dupstr_paths,
    )


def generate_primitive_cells(
    poscar_name: Optional[str] = None,
    polymlp_st: Optional[PolymlpStructure] = None,
    symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    refine_cell: bool = False,
) -> tuple[list[PolymlpStructure], list[int]]:
    """
    Generate primitive cells of a given structure under different symmetry tolerances.

    Parameters
    ----------
    poscar_name : str, optional
        Path to a POSCAR file.
    polymlp_st : PolymlpStructure, optional
        PolymlpStructure object.
    symprec_set : list of float
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
                if not refine_cell:
                    primitive_st = symutil.primitive_cell()
                else:
                    primitive_st = symutil.refine_cell()
            except TypeError:
                continue
            primitive_st_set.append(primitive_st)
            spg_number_set.append(spg_number)
    return primitive_st_set, spg_number_set


def generate_struct_reps_parallel(
    struct_lists: list[dict],
    num_process: int = -1,
    backend: str = "loky",
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    redrep_symprecs: list[float] = [1e-4, 1e-2, 1e-1],
    standardize_axis: bool = False,
    original_axis: bool = False,
    cartesian_coords: bool = True,
    refine_cell: bool = False,
) -> list[StructReps]:
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
                generate_struct_reps(
                    poscar_path=res["struct_path"],
                    polymlp_st=res.get("structure", None),
                    primitive_symprecs=primitive_symprecs,
                    redrep_symprecs=redrep_symprecs,
                    standardize_axis=standardize_axis,
                    original_axis=original_axis,
                    cartesian_coords=cartesian_coords,
                    refine_cell=refine_cell,
                    struct_tag=res.get("struct_tag", None),
                    dupstr_paths=res.get("dupstr_paths", None),
                )
            )
    else:
        redreps_list = joblib.Parallel(n_jobs=num_process, backend=backend)(
            joblib.delayed(generate_struct_reps)(
                poscar_path=res["struct_path"],
                polymlp_st=res.get("structure", None),
                primitive_symprecs=primitive_symprecs,
                redrep_symprecs=redrep_symprecs,
                standardize_axis=standardize_axis,
                original_axis=original_axis,
                cartesian_coords=cartesian_coords,
                refine_cell=refine_cell,
                struct_path=res["struct_path"],
                struct_tag=res.get("struct_tag", None),
                dupstr_paths=res.get("dupstr_paths", None),
            )
            for res in struct_lists
        )

    redreps_list = [s for s in redreps_list if s is not None]
    return redreps_list

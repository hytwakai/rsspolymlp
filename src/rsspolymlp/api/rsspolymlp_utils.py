from typing import Literal, Optional

from pypolymlp.utils.spglib_utils import SymCell
from rsspolymlp.analysis.rss_summarize import (
    log_all_unique_structures,
    log_unique_structures,
)
from rsspolymlp.analysis.struct_matcher.gen_struct_rep import (
    generate_struct_reps,
    generate_struct_reps_parallel,
)
from rsspolymlp.analysis.struct_matcher.identify_uniqstr import (
    UniqueStructIdentifier,
)
from rsspolymlp.analysis.struct_matcher.struct_match import (
    struct_match,
    struct_match_allrep,
)
from rsspolymlp.rss.optimization_mlp import OptimizationMLP


def struct_matcher(
    poscar_paths: Optional[list[str]] = None,
    struct_lists: Optional[list[dict]] = None,
    axis_tol: float = 0.05,
    pos_tol: float = 0.03,
    assign_method: Literal["redrep", "greedy", "HK"] = "redrep",
    num_process: int = -1,
    backend: str = "loky",
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    reduced_symprecs: list[float] = None,
    refine_cell: bool = False,
    standardize_axis: bool = False,
    frac_coords: bool = False,
    keep_unique: bool = False,
    output_file: str = "unique_struct.yaml",
    log_all: bool = False,
    verbose: bool = False,
):
    if poscar_paths is None and struct_lists is None:
        raise ValueError("Set poscar_paths or struct_lists")

    if reduced_symprecs is None:
        if not frac_coords:
            reduced_symprecs = [1e-4, 1e-2, 1e-1]
        else:
            reduced_symprecs = [1e-5, 1e-3, 1e-2]

    if struct_lists is None:
        struct_lists = [{} for _ in poscar_paths]
    if poscar_paths is not None:
        for struct, poscar_path in zip(struct_lists, poscar_paths):
            struct["poscar_path"] = poscar_path
            struct["struct_path"] = poscar_path

    struct_reps_list = generate_struct_reps_parallel(
        struct_lists,
        num_process=num_process,
        backend=backend,
        primitive_symprecs=primitive_symprecs,
        redrep_symprecs=reduced_symprecs,
        standardize_axis=standardize_axis,
        refine_cell=refine_cell,
        cartesian_coords=not frac_coords,
        verbose=verbose,
    )

    analyzer = UniqueStructIdentifier()
    if not assign_method == "redrep":
        pos_tol /= 10
    for struct_reps in struct_reps_list:
        analyzer.identify_duplicate_struct(
            struct_reps=struct_reps,
            keep_unique=keep_unique,
            axis_tol=axis_tol,
            pos_tol=pos_tol,
            assign_method=assign_method,
        )

    if log_all:
        unique_structs = analyzer.unique_str_keep
        unique_structs_sorted = sorted(
            unique_structs, key=lambda x: len(x), reverse=True
        )
    else:
        unique_structs = analyzer.unique_str
        unique_structs_sorted = sorted(
            unique_structs,
            key=lambda x: len(x.original_polymlp_st.positions.T),
            reverse=False,
        )

    with open(output_file, "w"):
        pass
    if log_all:
        log_all_unique_structures(
            output_file,
            unique_structs_sorted,
        )
    else:
        log_unique_structures(
            output_file,
            unique_structs_sorted,
        )


def struct_compare(
    poscar_paths,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    assign_method: Literal["redrep", "greedy", "HK"] = "redrep",
    refine_symprec: float = 1e-3,
    reduced_symprecs: list[float] = None,
    standardize_axis: bool = False,
    original_axis: bool = False,
    frac_coords: bool = False,
    refine_cell: bool = False,
    verbose: bool = True,
):
    sym = SymCell(poscar_name=poscar_paths[0], symprec=refine_symprec)
    polymlp_st1 = sym.refine_cell()
    sym = SymCell(poscar_name=poscar_paths[1], symprec=refine_symprec)
    polymlp_st2 = sym.refine_cell()

    if verbose:
        print("Structure 1:")
        print("- Axis:")
        print(polymlp_st1.axis.T)
        print("- Positions:")
        print(polymlp_st1.positions)
        print("Structure 2:")
        print("- Axis:")
        print(polymlp_st2.axis.T)
        print("- Positions:")
        print(polymlp_st2.positions)

    if reduced_symprecs is None:
        if not frac_coords:
            reduced_symprecs = [1e-4, 1e-2, 1e-1]
        else:
            reduced_symprecs = [1e-5, 1e-3, 1e-2]
    struct_reps_1 = generate_struct_reps(
        polymlp_st=polymlp_st1,
        refine_cell=refine_cell,
        redrep_symprecs=reduced_symprecs,
        standardize_axis=standardize_axis,
        original_axis=original_axis,
        cartesian_coords=not frac_coords,
        struct_path="st1",
    )
    struct_reps_2 = generate_struct_reps(
        polymlp_st=polymlp_st2,
        refine_cell=refine_cell,
        redrep_symprecs=reduced_symprecs,
        standardize_axis=standardize_axis,
        original_axis=original_axis,
        cartesian_coords=not frac_coords,
        struct_path="st2",
    )

    if assign_method == "redrep":
        judge = struct_match(
            struct_reps_1,
            struct_reps_2,
            axis_tol=axis_tol,
            pos_tol=pos_tol,
            verbose=verbose,
            spg_match=False,
        )
    else:
        judge = struct_match_allrep(
            struct_reps_1,
            struct_reps_2,
            axis_tol=axis_tol,
            pos_tol=pos_tol,
            assign_method=assign_method,
            verbose=verbose,
        )
    if verbose:
        print("Match:", judge)
    return judge


def geometry_opt(
    poscar_paths,
    pot="polymlp.yaml",
    pressure=0.0,
    with_symmetry=False,
    solver_method="CG",
    gtol=1e-6,
    c_maxiter=100,
):
    rssobj = OptimizationMLP(
        pot=pot,
        pressure=pressure,
        with_symmetry=with_symmetry,
        solver_method=solver_method,
        gtol=gtol,
        c_maxiter=c_maxiter,
        n_opt_str=10**6,
        not_stop_rss=True,
    )

    for poscar in poscar_paths:
        rssobj.run_optimization(poscar)

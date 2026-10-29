from typing import Literal, Optional

from rsspolymlp.analysis.rss_summarize import (
    log_all_unique_structures,
    log_unique_structures,
)
from rsspolymlp.analysis.struct_matcher.gen_struct_rep import StructRepGenerator
from rsspolymlp.analysis.struct_matcher.identify_uniqstr import (
    UniqueStructIdentifier,
)
from rsspolymlp.analysis.struct_matcher.struct_match import (
    struct_match,
    struct_match_allrep,
)
from rsspolymlp.rss.optimization_mlp import OptimizationMLP
from rsspolymlp.utils.spglib_utils import SymCell


def struct_matcher(
    poscar_paths: Optional[list[str]] = None,
    struct_list: Optional[list[dict]] = None,
    struct_reps_list: Optional[list[dict]] = None,
    axis_tol: float = 0.05,
    pos_tol: float = 0.03,
    assign_method: Literal["redrep", "greedy", "HK", "pymatgen"] = "redrep",
    num_process: int = -1,
    backend: str = "loky",
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    reduced_symprecs: list[float] = None,
    refine_cell: bool = False,
    normalize_axis: bool = False,
    output_file: str = "unique_struct.yaml",
    print_duplicates: bool = False,
    log_all: bool = False,
    verbose: bool = False,
):
    if poscar_paths is None and struct_list is None and struct_reps_list is None:
        raise ValueError("Set poscar_paths, struct_list, or struct_reps_list")

    time_log = []
    time_refine = iter_refine = time_redrep = iter_redrep = 0
    if struct_reps_list is None:
        if reduced_symprecs is None:
            if not normalize_axis:
                reduced_symprecs = [1e-4, 1e-2, 1e-1]
            else:
                reduced_symprecs = [1e-5, 1e-3, 1e-2]

        if struct_list is None:
            struct_list = [{} for _ in poscar_paths]
        if poscar_paths is not None:
            for struct, poscar_path in zip(struct_list, poscar_paths):
                struct["poscar_path"] = poscar_path
                struct["struct_path"] = poscar_path

        generator = StructRepGenerator(
            primitive_symprecs=primitive_symprecs,
            redrep_symprecs=reduced_symprecs,
            normalize_axis=normalize_axis,
            refine_cell=refine_cell,
            verbose=verbose,
        )
        struct_reps_list = generator.generate_struct_reps_parallel(
            struct_list,
            num_process=num_process,
            backend=backend,
        )
        time_refine = generator.time_refine
        iter_refine = generator.iter_refine
        time_redrep = generator.time_redrep
        iter_redrep = generator.iter_redrep
        time_log.append(("Refine", iter_refine, time_refine))
        time_log.append(("Reduce", iter_redrep, time_redrep))

    analyzer = UniqueStructIdentifier(
        keep_unique=log_all,
        axis_tol=axis_tol,
        pos_tol=pos_tol,
        assign_method=assign_method,
        verbose=verbose,
    )
    for struct_reps in struct_reps_list:
        analyzer.identify_duplicate_struct(
            struct_reps=struct_reps,
        )
    time_match = analyzer.time_match
    iter_match = analyzer.iter_match
    time_log.append(("Match", iter_match, time_match))

    print("code: rsspolymlp_utils.struct_matcher")
    print("  assign_method:", assign_method)
    print("  time:")
    print("    Process   Iterations  Avg. time (ms)  Total time (s)")
    for name, iterations, total_time in time_log:
        avg_time = total_time * 1000 / iterations
        print(
            f"    {name:<7} {iterations:>12} " f"{avg_time:>15.2f} {total_time:>15.2f}"
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

    with open(output_file, "w") as f:
        print("timing:", file=f)
        if iter_refine > 0:
            print("  refine:", file=f)
            print(
                f"    average_time_ms: {time_refine * 1000 / iter_refine:.2f}", file=f
            )
            print(f"    total_time_s: {time_refine:.2f}", file=f)
            print(f"    iterations: {iter_refine}", file=f)
            print("  reduce:", file=f)
            print(
                f"    average_time_ms: {time_redrep * 1000 / iter_redrep:.2f}", file=f
            )
            print(f"    total_time_s: {time_redrep:.2f}", file=f)
            print(f"    iterations: {iter_redrep}", file=f)
        print("  match:", file=f)
        print(f"    average_time_ms: {time_match * 1000 / iter_match:.2f}", file=f)
        print(f"    total_time_s: {time_match:.2f}", file=f)
        print(f"    iterations: {iter_match}", file=f)

    if log_all:
        log_all_unique_structures(
            output_file,
            unique_structs_sorted,
        )
    else:
        log_unique_structures(
            output_file,
            unique_structs_sorted,
            print_duplicates=print_duplicates,
        )


def struct_compare(
    poscar_paths,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    assign_method: Literal["redrep", "greedy", "HK", "pymatgen"] = "redrep",
    refine_symprec: float = 1e-3,
    reduced_symprecs: list[float] = None,
    normalize_axis: bool = False,
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
        if not normalize_axis:
            reduced_symprecs = [1e-4, 1e-2, 1e-1]
        else:
            reduced_symprecs = [1e-5, 1e-3, 1e-2]
    generator = StructRepGenerator(
        redrep_symprecs=reduced_symprecs,
        normalize_axis=normalize_axis,
        refine_cell=refine_cell,
    )
    struct_reps_1, _ = generator.generate_struct_reps(
        polymlp_st=polymlp_st1,
        struct_path="st1",
    )
    struct_reps_2, _ = generator.generate_struct_reps(
        polymlp_st=polymlp_st2,
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

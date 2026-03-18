from pypolymlp.utils.spglib_utils import SymCell
from rsspolymlp.analysis.struct_matcher.struct_match import (
    generate_primitive_cells,
    generate_reduced_struct,
    struct_match,
)
from rsspolymlp.analysis.unique_struct import (
    UniqueStructureAnalyzer,
    generate_unique_structs,
    log_unique_structures,
)
from rsspolymlp.rss.optimization_mlp import OptimizationMLP


def struct_matcher(
    poscar_paths,
    num_process: int = -1,
    backend: str = "loky",
    primitive_symprecs: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    output_file: str = "unique_struct.yaml",
):
    rss_results = []
    for poscar_path in poscar_paths:
        rss_results.append({"struct_path": poscar_path})

    unique_structs = generate_unique_structs(
        rss_results,
        num_process=num_process,
        backend=backend,
        symprec_set1=primitive_symprecs,
    )

    analyzer = UniqueStructureAnalyzer()
    for unique_struct in unique_structs:
        analyzer.identify_duplicate_struct(
            unique_struct=unique_struct,
            axis_tol=0.05,
            pos_tol=0.03,
        )

    unique_structs = analyzer.unique_str
    unique_structs_sorted = sorted(
        unique_structs,
        key=lambda x: len(x.original_structure.positions.T),
        reverse=False,
    )

    with open(output_file, "w"):
        pass
    log_unique_structures(
        output_file,
        unique_structs_sorted,
    )


def struct_compare(
    poscar_paths,
    refine_symprec: float = 1e-3,
    reduced_symprecs: list[float] = None,
    axis_tol: float = 0.01,
    pos_tol: float = 0.01,
    standardize_axis: bool = False,
    original_axis: bool = False,
    frac_coords: bool = False,
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
        print(polymlp_st1.positions.T)
        print("Structure 2:")
        print("- Axis:")
        print(polymlp_st2.axis.T)
        print("- Positions:")
        print(polymlp_st2.positions.T)

    if reduced_symprecs is None:
        if not frac_coords:
            reduced_symprecs = [1e-4, 1e-2, 1e-1]
        else:
            reduced_symprecs = [1e-5, 1e-3, 1e-2]

    primitive_st_set, spg_number_set = generate_primitive_cells(
        polymlp_st=polymlp_st1,
    )
    reduced_struct_set1 = []
    for i, primitive_st in enumerate(primitive_st_set):
        reduced_struct = generate_reduced_struct(
            primitive_st,
            spg_number_set[i],
            symprec_set=reduced_symprecs,
            standardize_axis=standardize_axis,
            original_axis=original_axis,
            cartesian_coords=not frac_coords,
        )
        reduced_struct_set1.append(reduced_struct)

    primitive_st_set, spg_number_set = generate_primitive_cells(
        polymlp_st=polymlp_st2,
    )
    reduced_struct_set2 = []
    for i, primitive_st in enumerate(primitive_st_set):
        reduced_struct = generate_reduced_struct(
            primitive_st,
            spg_number_set[i],
            symprec_set=reduced_symprecs,
            standardize_axis=standardize_axis,
            original_axis=original_axis,
            cartesian_coords=not frac_coords,
        )
        reduced_struct_set2.append(reduced_struct)

    judge = struct_match(
        reduced_struct_set1,
        reduced_struct_set2,
        axis_tol=axis_tol,
        pos_tol=pos_tol,
        verbose=verbose,
        spg_match=False,
    )
    return judge


def geometry_opt(
    poscar_paths,
    pot="polymlp.yaml",
    pressure=0.0,
    with_symmetry=False,
    solver_method="CG",
    c_maxiter=100,
):
    rssobj = OptimizationMLP(
        pot=pot,
        pressure=pressure,
        with_symmetry=with_symmetry,
        solver_method=solver_method,
        c_maxiter=c_maxiter,
        n_opt_str=10**6,
        not_stop_rss=True,
    )

    for poscar in poscar_paths:
        rssobj.run_optimization(poscar)

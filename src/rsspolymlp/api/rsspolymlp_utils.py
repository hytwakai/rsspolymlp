from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor

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
    symprec_set: list[float] = [1e-5, 1e-4, 1e-3, 1e-2],
    output_file: str = "unique_struct.yaml",
):
    rss_results = []
    for poscar_path in poscar_paths:
        rss_results.append({"poscar": poscar_path})

    unique_structs = generate_unique_structs(
        rss_results,
        num_process=num_process,
        backend=backend,
        symprec_set1=symprec_set,
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


def geometry_opt(
    poscar_paths,
    pot="polymlp.yaml",
    pressure=0.0,
    with_symmetry=False,
    solver_method="CG",
    c_maxiter=100,
    num_process=-1,
    backend="loky",
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

    if num_process == 1:
        for poscar in poscar_paths:
            rssobj.run_optimization(poscar)
    else:
        # Perform parallel optimization with joblib
        Parallel(n_jobs=num_process, backend=backend, verbose=100)(
            delayed(rssobj.run_optimization)(poscar) for poscar in poscar_paths
        )
        executor = get_reusable_executor(max_workers=num_process)
        executor.shutdown(wait=True)

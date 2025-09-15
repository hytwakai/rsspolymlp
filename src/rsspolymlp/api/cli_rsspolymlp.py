import argparse

from rsspolymlp.api.rsspolymlp import (
    rss_ghost_minima_cands,
    rss_ghost_minima_validate,
    rss_init_struct,
    rss_phase_analysis,
    rss_polymlp,
    rss_run_parallel,
    rss_run_single,
    rss_summarize,
    rss_uniq_struct,
)


def run():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init_struct",
        action="store_true",
        help="Mode: Initial random structure generation",
    )
    parser.add_argument(
        "--rss_parallel",
        action="store_true",
        help="Mode: RSS using the polynomial MLP in parallel",
    )
    parser.add_argument(
        "--rss_single",
        action="store_true",
        help="Mode: RSS using the polynomial MLP in single core",
    )
    parser.add_argument(
        "--uniq_struct",
        action="store_true",
        help="Mode: Unique structure identification",
    )
    parser.add_argument(
        "--rss_full",
        action="store_true",
        help="Run the full RSS workflow including:\n"
        "  (1) Initial random structure generation\n"
        "  (2) RSS using the polynomial MLP (parallel)\n"
        "  (3) Unique structure identification",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Mode: Unique structure identification across atom numbers `n` or pressure `p`",
    )
    parser.add_argument(
        "--ghost_minima",
        action="store_true",
        help="Mode: Ghost minimum structure identification",
    )
    parser.add_argument(
        "--phase_analysis",
        action="store_true",
        help="Mode: Ghost minimum structure identification",
    )

    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="List of element symbols",
    )

    # --init_struct mode
    parser.add_argument(
        "--atom_counts",
        type=int,
        nargs="+",
        default=None,
        help="Number of atoms for each element",
    )
    parser.add_argument(
        "--n_init_str",
        type=int,
        default=5000,
        help="Number of randomly generated initial structures",
    )
    parser.add_argument(
        "--max_volume",
        type=float,
        default=100.0,
        help="Maximum volume of initial structure (A^3/atom)",
    )
    parser.add_argument(
        "--min_volume",
        type=float,
        default=0.0,
        help="Minimum volume of initial structure (A^3/atom)",
    )
    parser.add_argument(
        "--least_distance",
        type=float,
        default=0.0,
        help="Minimum interatomic distance in initial structure (angstrom)",
    )

    # --rss_parallel mode
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=["polymlp.yaml"],
        help="Potential file for polynomial MLP",
    )
    parser.add_argument(
        "--n_opt_str",
        type=int,
        default=1000,
        help="Maximum number of optimized structures obtained from RSS",
    )
    parser.add_argument(
        "--max_init_str",
        type=int,
        default=None,
        help="Maximum number of randomly generated initial structures",
    )
    parser.add_argument(
        "--pressure", type=float, default=0.0, help="Pressure term (in GPa)"
    )
    parser.add_argument(
        "--solver_method", type=str, default="CG", help="Type of solver"
    )
    parser.add_argument(
        "--c_maxiter",
        type=int,
        default=100,
        help="Maximum number of iterations when c1 and c2 values are changed",
    )
    parser.add_argument(
        "--not_stop_rss",
        action="store_true",
        help="If enabled, the search continues until all structures are processed.",
    )

    # Options for enabling parallel execution (--rss_parallel, --uniq_struct, --summarize mode)
    parser.add_argument(
        "--parallel_method",
        type=str,
        choices=["joblib", "srun"],
        default="joblib",
        help="Select parallelization method: 'joblib' or 'srun'.",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=-1,
        help="Number of processes to use with joblib. Use -1 to use all available CPU cores.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["loky", "threading", "multiprocessing"],
        default="loky",
        help="Backend for joblib parallelization",
    )
    parser.add_argument(
        "--not_use_joblib",
        action="store_true",
        help="Disable parallel processing using joblib.",
    )

    # --uniq_struct mode
    parser.add_argument(
        "--num_str",
        type=int,
        default=-1,
        help="Number of optimized structures to analyze (-1 means all)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Cutoff radius used in the MLP model (optional)",
    )

    # Target paths for rsspolymlp postprocessing
    parser.add_argument(
        "--paths",
        nargs="*",
        type=str,
        help=(
            "Specify target directories or log files depending on the selected mode:\n"
            "  --summarize       : JSON files or directories containing VASP calculation results\n"
            "  --ghost_minima    : JSON files summarizing RSS results (e.g., Al1Cu1.json)\n"
            "  --phase_analysis  : JSON files or directories containing VASP calculation results\n"
        ),
    )

    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Energy threshold values (in meV/atom) for outputting POSCAR files (--summarize)."
        " Threshold values for energy above the convex hull in meV/atom (--phase_analysis)",
    )

    # --summarize mode
    parser.add_argument(
        "--symprec_set",
        nargs="*",
        type=float,
        default=[1e-5, 1e-4, 1e-3, 1e-2],
        help="List of symmetry tolerances used to identify distinct primitive cells.",
    )
    parser.add_argument(
        "--output_poscar",
        action="store_true",
        help="If set, POSCAR files will be output",
    )
    parser.add_argument(
        "--parse_vasp",
        action="store_true",
        help="If set, parse VASP output directories instead of RSS directories",
    )
    parser.add_argument(
        "--summarize_p",
        action="store_true",
        help="",
    )
    parser.add_argument(
        "--update_result",
        action="store_true",
        help="",
    )

    # --ghost_minima mode
    parser.add_argument(
        "--compare_dft",
        action="store_true",
        help="If set, runs detect_true_ghost_minima() to compare with DFT;"
        " otherwise, runs ghost_minima_candidates().",
    )

    # --phase_analysis mode
    parser.add_argument(
        "--ghost_minima_file",
        type=str,
        default=None,
        help="Path to a file listing the names of ghost_minima structures to exclude",
    )

    args = parser.parse_args()

    if args.init_struct:
        rss_init_struct(
            elements=args.elements,
            atom_counts=args.atom_counts,
            n_init_str=args.n_init_str,
            least_distance=args.least_distance,
            min_volume=args.min_volume,
            max_volume=args.max_volume,
        )

    if args.rss_parallel:
        rss_run_parallel(
            pot=args.pot,
            pressure=args.pressure,
            solver_method=args.solver_method,
            c_maxiter=args.c_maxiter,
            n_opt_str=args.n_opt_str,
            not_stop_rss=args.not_stop_rss,
            parallel_method=args.parallel_method,
            num_process=args.num_process,
            backend=args.backend,
        )

    if args.rss_single:
        rss_run_single(
            pot=args.pot,
            pressure=args.pressure,
            solver_method=args.solver_method,
            c_maxiter=args.c_maxiter,
            n_opt_str=args.n_opt_str,
            not_stop_rss=args.not_stop_rss,
        )

    if args.uniq_struct:
        rss_uniq_struct(
            num_str=args.num_str,
            cutoff=args.cutoff,
            use_joblib=not args.not_use_joblib,
            num_process=args.num_process,
            backend=args.backend,
        )

    if args.rss_full:
        rss_polymlp(
            elements=args.elements,
            atom_counts=args.atom_counts,
            pot=args.pot,
            pressure=args.pressure,
            c_maxiter=args.c_maxiter,
            n_opt_str=args.n_opt_str,
            max_init_str=args.max_init_str,
            min_volume=args.min_volume,
            max_volume=args.max_volume,
            least_distance=args.least_distance,
            solver_method=args.solver_method,
            not_stop_rss=args.not_stop_rss,
            num_process=args.num_process,
            backend=args.backend,
        )

    if args.summarize:
        rss_summarize(
            elements=args.elements,
            result_paths=args.paths,
            use_joblib=not args.not_use_joblib,
            num_process=args.num_process,
            backend=args.backend,
            symprec_set=args.symprec_set,
            output_poscar=args.output_poscar,
            thresholds=args.thresholds,
            parse_vasp=args.parse_vasp,
            summarize_p=args.summarize_p,
            update_result=args.update_result,
        )

    if args.ghost_minima:
        if args.compare_dft:
            rss_ghost_minima_validate(dft_dir=args.paths)
        else:
            rss_ghost_minima_cands(result_paths=args.paths)

    if args.phase_analysis:
        rss_phase_analysis(
            elements=args.elements,
            input_paths=args.paths,
            ghost_minima_file=args.ghost_minima_file,
            parse_vasp=args.parse_vasp,
            thresholds=args.thresholds,
        )

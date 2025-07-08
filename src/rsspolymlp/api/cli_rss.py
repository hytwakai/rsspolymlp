import argparse

from rsspolymlp.api.api_rss import (
    rss_init_struct,
    rss_parallel,
    rss_single_srun,
    rss_uniq_struct,
)
from rsspolymlp.api.api_rss_postprocess import (
    rss_ghost_minima_cands,
    rss_ghost_minima_validate,
    rss_phase_analysis,
    rss_summarize,
)
from rsspolymlp.api.parse_arguments import ParseArgument


def run_rss_init_struct():
    parser = argparse.ArgumentParser()
    ParseArgument.add_initial_structure_arguments(parser)
    args = parser.parse_args()

    rss_init_struct(
        elements=args.elements,
        atom_counts=args.atom_counts,
        num_init_str=args.num_init_str,
        least_distance=args.least_distance,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
    )


def run_rss_parallel():
    parser = argparse.ArgumentParser()
    ParseArgument.add_parallelization_arguments(parser)
    ParseArgument.add_optimization_arguments(parser)
    args = parser.parse_args()

    rss_parallel(
        pot=args.pot,
        pressure=args.pressure,
        solver_method=args.solver_method,
        maxiter=args.maxiter,
        num_opt_str=args.num_opt_str,
        not_stop_rss=args.not_stop_rss,
        parallel_method=args.parallel_method,
        num_process=args.num_process,
        backend=args.backend,
    )


def run_rss_single_srun():
    parser = argparse.ArgumentParser()
    ParseArgument.add_optimization_arguments(parser)
    args = parser.parse_args()

    rss_single_srun(
        pot=args.pot,
        pressure=args.pressure,
        solver_method=args.solver_method,
        maxiter=args.maxiter,
        num_opt_str=args.num_opt_str,
        not_stop_rss=args.not_stop_rss,
    )


def run_rss_uniq_struct():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_str",
        type=int,
        default=-1,
        help="Number of optimized structures to analyze (-1 means all)",
    )
    parser.add_argument(
        "--not_use_joblib",
        action="store_true",
        help="Disable parallel processing using joblib.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Cutoff radius used in the MLP model (optional)",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help="Pressure settings (in GPa) (optional)",
    )
    ParseArgument.add_parallelization_arguments(parser)
    args = parser.parse_args()

    rss_uniq_struct(
        num_str=args.num_str,
        cutoff=args.cutoff,
        pressure=args.pressure,
        use_joblib=not args.not_use_joblib,
        num_process=args.num_process,
        backend=args.backend,
    )


def run_rss_summarize():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        nargs="*",
        type=str,
        default=None,
        help="List of target element symbols",
    )
    parser.add_argument(
        "--result_paths",
        nargs="*",
        type=str,
        required=True,
        help="Path(s) to directories where RSS was performed",
    )
    parser.add_argument(
        "--not_use_joblib",
        action="store_true",
        help="Disable parallel processing using joblib.",
    )
    parser.add_argument(
        "--output_poscar",
        action="store_true",
        help="If set, POSCAR files will be output",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Energy threshold (in meV/atom) for outputting POSCAR files",
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
    ParseArgument.add_parallelization_arguments(parser)
    args = parser.parse_args()

    rss_summarize(
        elements=args.elements,
        result_paths=args.result_paths,
        use_joblib=not args.not_use_joblib,
        num_process=args.num_process,
        backend=args.backend,
        output_poscar=args.output_poscar,
        threshold=args.threshold,
        parse_vasp=args.parse_vasp,
        summarize_p=args.summarize_p,
    )


def run_rss_ghost_minima():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare_dft",
        action="store_true",
        help="If set, runs detect_true_ghost_minima() to compare with DFT;"
        " otherwise, runs ghost_minima_candidates().",
    )
    parser.add_argument(
        "--result_paths",
        nargs="*",
        type=str,
        default=None,
        help="Path(s) to RSS result log file(s).",
    )
    parser.add_argument(
        "--dft_dir",
        type=str,
        default=None,
        help="Path to the directory containing DFT results for ghost_minima structures.",
    )
    args = parser.parse_args()

    if args.compare_dft:
        rss_ghost_minima_validate(args.dft_dir)
    else:
        rss_ghost_minima_cands(args.result_paths)


def run_rss_phase_analysis():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        nargs="*",
        type=str,
        required=True,
        help="phase_analysisemical elements, e.g., La Bi",
    )
    parser.add_argument(
        "--result_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to RSS result log files, or to directories "
        "containing VASP geometry optimization results (used when --parse_vasp is enabled)",
    )
    parser.add_argument(
        "--ghost_minima_file",
        type=str,
        default=None,
        help="Path to a file listing the names of ghost_minima structures to exclude",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Threshold values for energy above the convex hull in meV/atom",
    )
    parser.add_argument(
        "--parse_vasp",
        action="store_true",
        help="If set, parse VASP output directories instead of RSS log files",
    )
    args = parser.parse_args()

    rss_phase_analysis(
        elements=args.elements,
        result_paths=args.result_paths,
        ghost_minima_file=args.ghost_minima_file,
        parse_vasp=args.parse_vasp,
        thresholds=args.thresholds,
    )

import argparse

from rsspolymlp.api.rsspolymlp_utils import geometry_opt, struct_matcher


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--struct_matcher",
        action="store_true",
        help="Mode: struct_matcher",
    )
    parser.add_argument(
        "--geometry_opt",
        action="store_true",
        help="Mode: geometry_optimizations",
    )

    # --struct_matcher mode
    parser.add_argument(
        "--poscar",
        type=str,
        nargs="+",
        default=None,
        help="Paths of target POSCAR files.",
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
        "--symprec_set",
        nargs="*",
        type=float,
        default=[1e-5, 1e-4, 1e-3, 1e-2],
        help="List of symmetry tolerances used to identify distinct primitive cells.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="unique_struct.yaml",
        help="Output file name (default: unique_struct.yaml).",
    )

    # --geometry_opt mode
    parser.add_argument(
        "--pot",
        nargs="*",
        type=str,
        default=["polymlp.yaml"],
        help="Potential file for polynomial MLP",
    )
    parser.add_argument(
        "--pressure", type=float, default=0.0, help="Pressure term (in GPa)"
    )
    parser.add_argument(
        "--symmetry",
        action="store_true",
        help="If enabled, the optimization is comducted with using symmetry constraints.",
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

    args = parser.parse_args()

    if args.struct_matcher:
        struct_matcher(
            poscar_paths=args.poscar,
            num_process=args.num_process,
            backend=args.backend,
            symprec_set=args.symprec_set,
            output_file=args.output_file,
        )

    if args.geometry_opt:
        geometry_opt(
            poscar_paths=args.poscar,
            pot=args.pot,
            pressure=args.pressure,
            with_symmetry=args.symmetry,
            solver_method=args.solver_method,
            c_maxiter=args.c_maxiter,
            num_process=args.num_process,
            backend=args.backend,
        )

import argparse

from rsspolymlp.api.rsspolymlp_utils import geometry_opt, struct_compare, struct_matcher


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--struct_matcher",
        action="store_true",
        help="Mode: struct_matcher",
    )
    parser.add_argument(
        "--struct_compare",
        action="store_true",
        help="Mode: struct_compare",
    )
    parser.add_argument(
        "--geometry_opt",
        action="store_true",
        help="Mode: geometry_optimizations",
    )

    # --struct_matcher mode
    parser.add_argument(
        "--poscars",
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
        "--primitive_symprecs",
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

    # --struct_compare mode
    parser.add_argument(
        "--refine_symprec",
        type=float,
        default=1e-3,
        help="Symmetry tolerance used to identify refine structure.",
    )
    parser.add_argument(
        "--reduced_symprecs",
        nargs="*",
        type=float,
        default=None,
        help="List of symmetry tolerances used to identify reduced structure representation.",
    )
    parser.add_argument(
        "--axis_tol",
        type=float,
        default=1e-2,
        help="",
    )
    parser.add_argument(
        "--pos_tol",
        type=float,
        default=1e-2,
        help="",
    )
    parser.add_argument("--standardize_axis", action="store_true", help="")
    parser.add_argument("--original_axis", action="store_true", help="")
    parser.add_argument("--frac_coords", action="store_true", help="")
    parser.add_argument("--not_verbose", action="store_true", help="")

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
            poscar_paths=args.poscars,
            num_process=args.num_process,
            backend=args.backend,
            primitive_symprecs=args.primitive_symprecs,
            output_file=args.output_file,
        )

    if args.struct_compare:
        struct_compare(
            poscar_paths=args.poscars,
            refine_symprec=args.refine_symprec,
            reduced_symprecs=args.reduced_symprecs,
            axis_tol=args.axis_tol,
            pos_tol=args.pos_tol,
            standardize_axis=args.standardize_axis,
            original_axis=args.original_axis,
            frac_coords=args.frac_coords,
            verbose=not args.not_verbose,
        )

    if args.geometry_opt:
        geometry_opt(
            poscar_paths=args.poscars,
            pot=args.pot,
            pressure=args.pressure,
            with_symmetry=args.symmetry,
            solver_method=args.solver_method,
            c_maxiter=args.c_maxiter,
        )

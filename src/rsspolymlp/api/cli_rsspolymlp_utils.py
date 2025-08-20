import argparse

from rsspolymlp.api.rsspolymlp_utils import struct_matcher


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--struct_matcher",
        action="store_true",
        help="Mode: struct_matcher",
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
        "--not_use_joblib",
        action="store_true",
        help="Disable parallel processing using joblib.",
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

    args = parser.parse_args()

    if args.struct_matcher:
        struct_matcher(
            poscar_paths=args.poscar,
            use_joblib=not args.not_use_joblib,
            num_process=args.num_process,
            backend=args.backend,
            symprec_set=args.symprec_set,
            output_file=args.output_file,
        )

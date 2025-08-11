import argparse

from rsspolymlp.api.rsspolymlp_plot import pareto_opt_mlp, plot_binary, plot_rss_error


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Mode: RSS results in binary system",
    )
    parser.add_argument(
        "--pareto_opt",
        action="store_true",
        help="Mode: Preto-optimal MLPs",
    )
    parser.add_argument(
        "--rss_error",
        action="store_true",
        help="Mode: RSS error",
    )

    # --binary mode
    parser.add_argument("--threshold", type=float, default=None)

    # --pareto_opt mode
    parser.add_argument(
        "--paths",
        type=str,
        nargs="+",
        default=None,
        help="Directory paths containing polymlp_error.yaml and polymlp_cost.yaml.",
    )
    parser.add_argument(
        "--error_path",
        type=str,
        default="polymlp_error.yaml",
        help="File name for storing MLP prediction errors.",
    )
    parser.add_argument(
        "--rmse_path",
        type=str,
        default="test/minima-close",
        help="A part of the path name of the dataset used to compute the energy RMSE "
        "for identifying Pareto-optimal MLPs.",
    )
    parser.add_argument(
        "--include_force",
        action="store_true",
        help="Filtered Pareto-optimal MLPs are shown as closed squares in a plot.",
    )
    parser.add_argument(
        "--rmse_max",
        type=float,
        default=30,
        help="Y axis maximum in the Pareto-optimal MLP plot",
    )

    # --rss_error mode
    parser.add_argument(
        "--mlp_jsons",
        type=str,
        nargs="+",
        default=None,
        help="JSON file paths.",
    )
    parser.add_argument(
        "--dft_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Directories containing VASP calculation results.",
    )
    parser.add_argument(
        "--stress_tensor",
        action="store_true",
        help="Including stress tensor.",
    )
    parser.add_argument(
        "--e_low",
        type=float,
        default=-5,
        help="Minimum energy value.",
    )
    parser.add_argument(
        "--e_high",
        type=float,
        default=10,
        help="Maximum energy value.",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="rss_error.png",
        help="Output file name.",
    )

    args = parser.parse_args()

    if args.binary:
        plot_binary(threshold=args.threshold)

    if args.pareto_opt:
        pareto_opt_mlp(
            mlp_paths=args.paths,
            error_path=args.error_path,
            rmse_path=args.rmse_path,
            include_force=args.include_force,
            rmse_max=args.rmse_max,
        )

    if args.rss_error:
        plot_rss_error(
            mlp_jsons=args.mlp_jsons,
            dft_dirs=args.dft_dirs,
            stress_tensor=args.stress_tensor,
            e_low=args.e_low,
            e_high=args.e_high,
            file_name=args.file_name,
        )

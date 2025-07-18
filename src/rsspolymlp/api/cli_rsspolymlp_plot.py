import argparse

from rsspolymlp.api.rsspolymlp_plot import pareto_opt_mlp, plot_binary


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

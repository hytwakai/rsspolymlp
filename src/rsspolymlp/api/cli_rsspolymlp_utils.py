import argparse

from rsspolymlp.api.rsspolymlp_utils import estimate_cost, pareto_opt_mlp, polymlp_dev


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlp_dev",
        action="store_true",
        help="Mode: Polymomial MLP development",
    )
    parser.add_argument(
        "--calc_cost",
        action="store_true",
        help="Mode: Estimation of polymomial MLP cost",
    )
    parser.add_argument(
        "--pareto_opt",
        action="store_true",
        help="Mode: Pareto-optimal MLP detection",
    )

    # --mlp_dev mode
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Directory path containing polymlp*.in files.",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        required=True,
        help="List of chemical element symbols.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        required=True,
        help="List of paths containing training datasets.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        nargs="+",
        required=True,
        help="List of paths containing test datasets.",
    )
    parser.add_argument(
        "--w_large_force",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some large forces.",
    )
    parser.add_argument(
        "--w_wo_force",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some very large forces.",
    )
    parser.add_argument(
        "--include_wo_force",
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "--alpha_param",
        type=int,
        nargs=3,
        default=[-4, 3, 8],
        help="Three integers specifying the reg_alpha_params values to replace (default: -4 3 8).",
    )

    # Target paths containing polynomial MLP infomation
    parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Specify target directories based on the mode:\n"
            "  --calc_cost  : Contains polymlp.yaml or polymlp.in\n"
            "  --pareto_opt : Contains polymlp_error.yaml and polymlp_cost.yaml\n"
        ),
    )

    # --calc_cost mode
    parser.add_argument(
        "--param_input",
        action="store_true",
        help="",
    )

    # --pareto_opt mode
    parser.add_argument(
        "--error_path",
        type=str,
        default="polymlp_error.yaml",
        help="File name for storing MLP prediction errors.",
    )
    parser.add_argument(
        "--rmse_path",
        type=str,
        default="test/close_minima",
        help="A part of the path name of the dataset used to compute the energy RMSE "
        "for identifying Pareto-optimal MLPs.",
    )

    args = parser.parse_args()

    if args.mlp_dev:
        polymlp_dev(
            input_path=args.input_path,
            elements=args.elements,
            train_fata=args.train_data,
            test_data=args.test_data,
            w_large_force=args.w_large_force,
            w_wo_force=args.w_wo_force,
            include_wo_force=args.include_wo_force,
            alpha_param=args.alpha_param,
        )

    if args.calc_cost:
        estimate_cost(
            mlp_paths=args.paths,
            param_input=args.param_input,
        )

    if args.pareto_opt:
        pareto_opt_mlp(
            mlp_paths=args.paths,
            error_path=args.error_path,
            rmse_path=args.rmse_path,
        )

import argparse

from rsspolymlp.api.rsspolymlp_devkit import (
    compress_vasprun,
    divide_dft_dataset,
    estimate_cost,
    mlp_dataset,
    pareto_opt_mlp,
    polymlp_dev,
)


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
    parser.add_argument(
        "--gen_data",
        action="store_true",
        help="Mode: MLP dataset generation",
    )
    parser.add_argument(
        "--compress_data",
        action="store_true",
        help="Mode: Compress vasprun.xml files and check convergence",
    )
    parser.add_argument(
        "--divide_data",
        action="store_true",
        help="Mode: DFT dataset division",
    )

    # --mlp_dev mode
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Directory path containing polymlp*.in files.",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="List of chemical element symbols.",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        default=None,
        help="List of paths containing training datasets.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        nargs="+",
        default=None,
        help="List of paths containing test datasets.",
    )
    parser.add_argument(
        "--w_large_f",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some large forces.",
    )
    parser.add_argument(
        "--w_vlarge_f",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some very large forces.",
    )
    parser.add_argument(
        "--w_vlarge_s",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some very large stress tensor.",
    )
    parser.add_argument(
        "--include_vlarge_f",
        action="store_true",
        help="Include force entries in the force-very-large dataset.",
    )
    parser.add_argument(
        "--include_vlarge_s",
        action="store_true",
        help="Include stress tensor entries in the stress-very-large dataset.",
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
        "--paths",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Specify target directories based on the mode:\n"
            "  --calc_cost : Contains polymlp.yaml or polymlp.in\n"
            "  --pareto_opt : Contains polymlp_error.yaml and polymlp_cost.yaml\n"
            "  --compress_data : Contains a vasprun.xml file\n"
            "  --divide_data : Contains vasprun.xml files\n"
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
        default="test/minima-close",
        help="A part of the path name of the dataset used to compute the energy RMSE "
        "for identifying Pareto-optimal MLPs.",
    )

    # --gen_data mode
    parser.add_argument(
        "--poscars",
        type=str,
        nargs="+",
        default=None,
        help="Input POSCAR file(s) for structure generation",
    )
    parser.add_argument(
        "--per_volume",
        type=float,
        default=1.0,
        help="Volume scaling factor for generated structures",
    )
    parser.add_argument(
        "--disp_max",
        type=float,
        default=30,
        help="Maximum displacement ratio for structure generation",
    )
    parser.add_argument(
        "--disp_grid",
        type=float,
        default=1,
        help="Displacement ratio interval (step size)",
    )
    parser.add_argument(
        "--natom_lb",
        type=int,
        default=30,
        help="Minimum number of atoms in generated structure",
    )
    parser.add_argument(
        "--natom_ub",
        type=int,
        default=150,
        help="Maximum number of atoms in generated structure",
    )
    parser.add_argument(
        "--str_name",
        type=int,
        default=-1,
        help="Index for extracting structure name from POSCAR path",
    )

    # --compress_data mode
    parser.add_argument(
        "--output_dir",
        type=str,
        default="compress_dft_data",
        help="Output directory path.",
    )
    parser.add_argument(
        "--num_process",
        type=int,
        default=4,
        help="Number of processes to use with joblib.",
    )

    # --divide_data mode
    parser.add_argument(
        "--threshold_vlarge_s",
        type=float,
        default=300.0,
        help="Stress threshold (GPa) for stress-very-large structures.",
    )
    parser.add_argument(
        "--threshold_vlarge_f",
        type=float,
        default=100.0,
        help="Force threshold (eV/ang.) for force-very-large structures.",
    )
    parser.add_argument(
        "--threshold_large_f",
        type=float,
        default=10.0,
        help="Force threshold (eV/ang.) for force-large structures.",
    )
    parser.add_argument(
        "--threshold_close_minima",
        type=float,
        default=1.0,
        help="Force threshold (eV/ang.) for minima-close structures.",
    )
    parser.add_argument(
        "--divide_ratio",
        type=float,
        default=0.1,
        help="Ratio of the dataset to be used for testing (e.g., 0.1 for 10% test data).",
    )

    args = parser.parse_args()

    if args.mlp_dev:
        polymlp_dev(
            input_path=args.input_path,
            elements=args.elements,
            train_data=args.train_data,
            test_data=args.test_data,
            weight_large_force=args.w_large_f,
            weight_vlarge_force=args.w_vlarge_f,
            weight_vlarge_stress=args.w_vlarge_s,
            include_vlarge_force=args.include_vlarge_f,
            include_vlarge_stress=args.include_vlarge_s,
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

    if args.gen_data:
        mlp_dataset(
            poscars=args.poscars,
            per_volume=args.per_volume,
            disp_max=args.disp_max,
            disp_grid=args.disp_grid,
            natom_lb=args.natom_lb,
            natom_ub=args.natom_ub,
            str_name=args.str_name,
        )

    if args.compress_data:
        compress_vasprun(
            vasp_paths=args.paths,
            output_dir=args.output_dir,
            num_process=args.num_process,
        )

    if args.divide_data:
        divide_dft_dataset(
            target_dirs=args.paths,
            threshold_vlarge_s=args.threshold_vlarge_s,
            threshold_vlarge_f=args.threshold_vlarge_f,
            threshold_large_f=args.threshold_large_f,
            threshold_close_minima=args.threshold_close_minima,
            divide_ratio=args.divide_ratio,
        )

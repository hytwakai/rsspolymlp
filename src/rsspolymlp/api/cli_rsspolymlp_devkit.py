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
        "--w_e_high",
        type=float,
        default=0.1,
        help="Weight to assign to datasets with relatively high or low energies.",
    )
    parser.add_argument(
        "--w_f_large",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some large forces.",
    )
    parser.add_argument(
        "--w_f_exlarge",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some extremly large forces.",
    )
    parser.add_argument(
        "--w_s_large",
        type=float,
        default=1.0,
        help="Weight to assign to datasets with some extremly large stress tensor.",
    )
    parser.add_argument(
        "--exclude_f_exlarge",
        action="store_true",
        help="Exclude force entries in the f_exlarge dataset.",
    )
    parser.add_argument(
        "--include_s_large",
        action="store_true",
        help="Include stress tensor entries in the s_large dataset.",
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
        "--th_e_high",
        type=float,
        default=10.0,
        help="Energy threshold (eV/atom) for structures classified as high energy (-e_high).",
    )
    parser.add_argument(
        "--th_e_low",
        type=float,
        default=None,
        help="Energy threshold (eV/atom) for structures classified as low energy (-e_high).",
    )
    parser.add_argument(
        "--th_f_small",
        type=float,
        default=3.0,
        help="Force threshold (eV/ang.) for structures with small forces (f_small).",
    )
    parser.add_argument(
        "--th_f_normal",
        type=float,
        default=10.0,
        help="Force threshold (eV/ang.) for structures with normal forces (f_normal).",
    )
    parser.add_argument(
        "--th_f_large",
        type=float,
        default=100.0,
        help="Force threshold (eV/ang.) for structures with large forces (f_large).",
    )
    parser.add_argument(
        "--th_s_large",
        type=float,
        default=200.0,
        help="Stress threshold (GPa) for structures with large stress tensor (s_large).",
    )
    parser.add_argument(
        "--th_s_small",
        type=float,
        default=None,
        help="Stress threshold (GPa) for structures with small stress tensor (s_small)",
    )
    parser.add_argument(
        "--include_stress",
        action="store_true",
        help="Include stress tensor term in energy evaluation."
    )
    parser.add_argument(
        "--divide_ratio",
        type=float,
        default=0.1,
        help="Ratio of the dataset to be used for testing (e.g., 0.1 for 10 percent test data).",
    )

    args = parser.parse_args()

    if args.mlp_dev:
        polymlp_dev(
            input_path=args.input_path,
            elements=args.elements,
            train_data=args.train_data,
            test_data=args.test_data,
            weight_e_high=args.w_e_high,
            weight_f_large=args.w_f_large,
            weight_f_exlarge=args.w_f_exlarge,
            weight_s_large=args.w_s_large,
            include_f_exlarge=not args.exclude_f_exlarge,
            include_s_large=args.include_s_large,
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
            threshold_e_high=args.th_e_high,
            threshold_e_low=args.th_e_low,
            threshold_f_small=args.th_f_small,
            threshold_f_normal=args.th_f_normal,
            threshold_f_large=args.th_f_large,
            threshold_s_large=args.th_s_large,
            threshold_s_small=args.th_s_small,
            include_stress=args.include_stress,
            divide_ratio=args.divide_ratio,
        )

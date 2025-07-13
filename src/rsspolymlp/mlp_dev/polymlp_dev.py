import glob
import os
import shutil

from rsspolymlp.common.atomic_energy import atomic_energy


def prepare_polymlp_input_file(
    input_path: str,
    element_list: list[str],
    training_data_paths: list[str],
    test_data_paths: list[str],
    w_large_force: float = 1.0,
    w_wo_force: float = 1.0,
    include_wo_force: bool = False,
    alpha_param: list[int] = None,
):
    """
    Build or extend a polymlp input file.

    The function relies on specific keywords in the *data_path string* to determine:
        - "large_force":
            Dataset containing structures with some large forces;
            a small weight is assigned.
        - "wo_force":
            Dataset containing structures with extremely large forces
            (e.g., due to close atomic positions); force training is disabled.
        - otherwise:
            Other dataset includes forces and stresses; default weight is used.
    """
    # Copy polymlp input files and append element info
    for src in glob.glob(input_path + "/polymlp*"):
        dst = os.path.basename(src)
        shutil.copyfile(src, dst)
        with open(dst, "a") as f:
            f.write("\n")
            f.write(f"n_type {len(element_list)}\n")
            f.write("elements " + " ".join(element_list) + "\n")
    if os.path.isfile(input_path + "/polymlp_cost.yaml"):
        shutil.copyfile(input_path + "/polymlp_cost.yaml", "./polymlp_cost.yaml")

    main_input = "polymlp.in" if os.path.isfile("polymlp.in") else "polymlp1.in"

    with open(main_input, "a") as f:
        # Write atomic energy for each element
        f.write(
            "atomic_energy "
            + " ".join(str(atomic_energy(e)) for e in element_list)
            + "\n\n"
        )

        # Write training and test data
        for data_path in training_data_paths:
            if "large_force" in data_path:
                f.write(f"train_data {data_path}/* True {w_large_force}\n")
            elif "wo_force" in data_path:
                f.write(f"train_data {data_path}/* {include_wo_force} {w_wo_force}\n")
            else:
                f.write(f"train_data {data_path}/* True 1.0\n")
        f.write("\n")
        for data_path in test_data_paths:
            if not os.path.isdir(data_path):
                continue
            if "large_force" in data_path:
                f.write(f"test_data {data_path}/* True {w_large_force}\n")
            elif "wo_force" in data_path:
                f.write(f"test_data {data_path}/* {include_wo_force} {w_wo_force}\n")
            else:
                f.write(f"test_data {data_path}/* True 1.0\n")

    # Replace alpha parameters if specified
    if alpha_param is not None:
        with open(main_input, "r") as f:
            content = f.read()
        content = content.replace(
            "reg_alpha_params -4 3 8",
            f"reg_alpha_params {alpha_param[0]} {alpha_param[1]} {alpha_param[2]}",
        )
        with open(main_input, "w") as f:
            f.write(content)


if __name__ == "__main__":

    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    prepare_polymlp_input_file(
        args.input_path,
        args.elements,
        args.train_data,
        args.test_data,
        args.w_large_force,
        args.w_wo_force,
        args.include_wo_force,
        args.alpha_param,
    )

    input_files = sorted(glob.glob("polymlp*.in"))
    cmd = ["pypolymlp", "-i"] + input_files
    subprocess.run(cmd, check=True)

import glob
import os
import shutil

from rsspolymlp.common.atomic_energy import atomic_energy


def prepare_polymlp_input_file(
    input_path: str,
    element_list: list[str],
    training_data_paths: list[str],
    test_data_paths: list[str],
    weight_e_high: float = 0.1,
    weight_f_large: float = 1.0,
    weight_f_exlarge: float = 1.0,
    weight_s_large: float = 1.0,
    include_f_exlarge: bool = True,
    include_s_large: bool = False,
    alpha_param: list[int] = None,
):
    """
    Generate or update a polymlp input file for model training.

    This function uses keywords in the dataset path names to determine weights:
        - "-e_high":
            Indicates datasets containing structures with relatively high or low energies.
            The weight specified by `weight_e_high` is applied, and it may be combined
            with other weight multipliers if applicable.
        - "f_large":
            Indicates datasets containing some structures with moderately large forces
            (default threshold: ~10 eV/Å). The weight specified by `weight_f_large`
            is applied (default: 1.0).
        - "f_exlarge":
            Indicates datasets with some structures exhibiting extremely large forces
            (default threshold: ~100 eV/Å), typically due to very short interatomic distances.
            The weight specified by `weight_f_exlarge` is applied (default: 1.0).
        - "s_large":
            Indicates datasets with extremely large stresses (default threshold: ~200 GPa),
            which often coincide with many large forces.
            Force training is disabled for these datasets.
        - Others:
            Treated as standard datasets with typical force and stress values.
            Default weights are applied.

    Parameters:
        input_path (str): Directory where the polymlp input files will be written.
        elements (list[str]): List of atomic element symbols.
        train_data_paths (list[str]): Paths to training datasets.
        test_data_paths (list[str]): Paths to test datasets.
        weight_e_high (float): Weight assigned to "-e_high" datasets.
        weight_f_large (float): Weight assigned to "f_large" datasets.
        weight_f_exlarge (float): Weight assigned to "f_exlarge" datasets.
        include_f_exlarge (bool): Whether to include force entries in "f_exlarge".
        include_s_large (bool): Whether to include force entries in "s_large".
        alpha_params (list[int]): List of three integers for specifying regularization strength.
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

        # Write training data
        for data_path in training_data_paths:
            f_include = True
            if "-e_high" in data_path:
                all_weight = weight_e_high
            else:
                all_weight = 1.0
            if "f_large" in data_path:
                all_weight *= weight_f_large
            elif "f_exlarge" in data_path:
                f_include = include_f_exlarge
                all_weight *= weight_f_exlarge
            elif "s_large" in data_path:
                f_include = include_s_large
                all_weight *= weight_s_large
            f.write(f"train_data {data_path}/* {f_include} {all_weight}\n")
        f.write("\n")

        # Write test data
        for data_path in test_data_paths:
            if not os.path.isdir(data_path):
                continue
            f_include = True
            if "-e_high" in data_path:
                all_weight = weight_e_high
            else:
                all_weight = 1.0
            if "f_large" in data_path:
                all_weight *= weight_f_large
            elif "f_exlarge" in data_path:
                f_include = include_f_exlarge
                all_weight *= weight_f_exlarge
            elif "s_large" in data_path:
                f_include = include_s_large
                all_weight *= weight_s_large
            f.write(f"test_data {data_path}/* {f_include} {all_weight}\n")

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

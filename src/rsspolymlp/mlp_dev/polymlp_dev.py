import glob
import os
import shutil

from rsspolymlp.common.atomic_energy import atomic_energy


def prepare_polymlp_input_file(
    input_path: str,
    element_list: list[str],
    training_data_paths: list[str],
    test_data_paths: list[str],
    weight_large_force: float = 1.0,
    weight_vlarge_force: float = 1.0,
    weight_vlarge_stress: float = 1.0,
    include_vlarge_force: bool = True,
    include_vlarge_stress: bool = False,
    alpha_param: list[int] = None,
):
    """
    Generate or update a polymlp input file for model training.

    This function uses keywords in the dataset path names to determine weights:
        - "force-large":
            Indicates datasets containing some structures with moderately large forces
            (default threshold: ~10 eV/Å). The weight specified by `weight_large_force`
            is applied (default: 1.0).
        - "force-very-large":
            Indicates datasets with some structures exhibiting extremely large forces
            (default threshold: ~100 eV/Å), typically due to very short interatomic distances.
            The weight specified by `weight_vlarge_force` is applied (default: 1.0).
        - "stress-very-large":
            Indicates datasets with extremely large stresses (default threshold: ~300 GPa),
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
        weight_large_force (float): Weight assigned to "force-large" datasets.
        weight_vlarge_force (float): Weight assigned to "force-very-large" datasets.
        include_vlarge_force (bool): Whether to include force entries in "force-very-large".
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

        # Write training and test data
        for data_path in training_data_paths:
            if "force-large" in data_path:
                f.write(f"train_data {data_path}/* True {weight_large_force}\n")
            elif "force-very-large" in data_path:
                f.write(
                    f"train_data {data_path}/* {include_vlarge_force} {weight_vlarge_force}\n"
                )
            elif "stress-very-large" in data_path:
                f.write(
                    f"train_data {data_path}/* {include_vlarge_stress} {weight_vlarge_stress}\n"
                )
            else:
                f.write(f"train_data {data_path}/* True 1.0\n")
        f.write("\n")
        for data_path in test_data_paths:
            if not os.path.isdir(data_path):
                continue
            if "force-large" in data_path:
                f.write(f"test_data {data_path}/* True {weight_large_force}\n")
            elif "force-very-large" in data_path:
                f.write(
                    f"test_data {data_path}/* {include_vlarge_force} {weight_vlarge_force}\n"
                )
            elif "stress-very-large" in data_path:
                f.write(
                    f"test_data {data_path}/* {include_vlarge_stress} {weight_vlarge_stress}\n"
                )
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

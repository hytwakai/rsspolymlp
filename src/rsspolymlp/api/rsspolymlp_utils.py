import glob
import subprocess

from rsspolymlp.mlp_dev.polymlp_dev import prepare_polymlp_input_file


def polymlp_dev(
    input_path: str,
    elements: list[str],
    train_data: list[str],
    test_data: list[str],
    w_large_force: float = 1.0,
    w_wo_force: float = 1.0,
    include_wo_force: bool = False,
    alpha_param: list[int] = None,
):
    prepare_polymlp_input_file(
        input_path=input_path,
        element_list=elements,
        training_data_paths=train_data,
        test_data_paths=test_data,
        w_large_force=w_large_force,
        w_wo_force=w_wo_force,
        include_wo_force=include_wo_force,
        alpha_param=alpha_param,
    )

    input_files = sorted(glob.glob("polymlp*.in"))
    cmd = ["pypolymlp", "-i"] + input_files
    subprocess.run(cmd, check=True)

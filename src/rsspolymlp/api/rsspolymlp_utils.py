import glob
import os
import subprocess

import numpy as np

from pypolymlp.utils.count_time import PolymlpCost
from rsspolymlp.mlp_dev.estimate_cost import make_polymlp_yaml
from rsspolymlp.mlp_dev.pareto_opt_mlp import pareto_front, parse_mlp_property
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


def estimate_cost(mlp_paths: str, param_input: bool = False):
    cwd_dir = os.getcwd()

    for _path in mlp_paths:
        if param_input:
            pot_path = make_polymlp_yaml()
        else:
            pot_path = "./polymlp.yaml*"

        os.chdir(_path)
        pot = glob.glob(pot_path)

        PolymlpCost(pot=pot).run(n_calc=10)

        os.chdir(cwd_dir)


def pareto_opt_mlp(
    mlp_paths: list[str],
    error_path: str = "polymlp_error.yaml",
    rmse_path: str = "test/close_minima",
):

    res_dict = parse_mlp_property(mlp_paths, error_path=error_path, rmse_path=rmse_path)

    sort_idx = np.argsort(res_dict["cost"])
    res_dict = {key: np.array(_list)[sort_idx] for key, _list in res_dict.items()}

    rmse_e_time = []
    for i in range(len(res_dict["cost"])):
        rmse_e_time.append([res_dict["cost"][i], res_dict["rmse"][i][0]])
    pareto_e_idx = pareto_front(np.array(rmse_e_time))
    not_pareto_idx = np.ones(len(rmse_e_time), dtype=bool)
    not_pareto_idx[pareto_e_idx] = False

    rmse_ef_time = []
    for i in pareto_e_idx:
        rmse_ef_time.append(
            [res_dict["cost"][i], res_dict["rmse"][i][0], res_dict["rmse"][i][1]]
        )
    _pareto_ef_idx = pareto_front(np.array(rmse_ef_time))
    pareto_ef_idx = np.array(pareto_e_idx)[_pareto_ef_idx]

    os.makedirs("analyze_pareto", exist_ok=True)
    os.chdir("analyze_pareto")

    with open("pareto_optimum.yaml", "w") as f:
        print("units:", file=f)
        print("  cost:        'msec/atom/step'", file=f)
        print("  rmse_energy: 'meV/atom'", file=f)
        print("  rmse_force:  'eV/angstrom'", file=f)
        print("", file=f)
        print("pareto_optimum:", file=f)
        for idx in pareto_e_idx:
            print(f"  {res_dict['mlp_name'][idx]}:", file=f)
            print(f"    cost:        {res_dict['cost'][idx]}", file=f)
            print(f"    rmse_energy: {res_dict['rmse'][idx][0]}", file=f)
            print(f"    rmse_force:  {res_dict['rmse'][idx][1]}", file=f)
        print("", file=f)
        print("# Filter out solutions with worse force error at higher cost", file=f)
        print("pareto_optimum_include_force:", file=f)
        for idx in pareto_ef_idx:
            print(f"  {res_dict['mlp_name'][idx]}:", file=f)
            print(f"    cost:        {res_dict['cost'][idx]}", file=f)
            print(f"    rmse_energy: {res_dict['rmse'][idx][0]}", file=f)
            print(f"    rmse_force:  {res_dict['rmse'][idx][1]}", file=f)

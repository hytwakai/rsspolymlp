import glob
import os
import subprocess
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

from pypolymlp.utils.count_time import PolymlpCost
from rsspolymlp.mlp_dev.dataset.compress_vasprun import check_convergence, compress
from rsspolymlp.mlp_dev.dataset.divide_dataset import divide_dataset, divide_train_test
from rsspolymlp.mlp_dev.dataset.gen_mlp_dataset import gen_mlp_data
from rsspolymlp.mlp_dev.estimate_cost import make_polymlp_yaml
from rsspolymlp.mlp_dev.pareto_opt_mlp import pareto_front, parse_mlp_property
from rsspolymlp.mlp_dev.polymlp_dev import prepare_polymlp_input_file


def polymlp_dev(
    input_path: str,
    elements: list[str],
    train_data: list[str],
    test_data: list[str],
    weight_e_high: float = 0.1,
    weight_f_large: float = 1.0,
    weight_f_exlarge: float = 1.0,
    weight_s_large: float = 1.0,
    include_f_exlarge: bool = True,
    include_s_large: bool = False,
    alpha_param: list[int] = None,
):
    prepare_polymlp_input_file(
        input_path=input_path,
        element_list=elements,
        training_data_paths=train_data,
        test_data_paths=test_data,
        weight_e_high=weight_e_high,
        weight_f_large=weight_f_large,
        weight_f_exlarge=weight_f_exlarge,
        weight_s_large=weight_s_large,
        include_f_exlarge=include_f_exlarge,
        include_s_large=include_s_large,
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
    rmse_path: str = "test/minima-close",
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


def mlp_dataset(
    poscars=list[str],
    per_volume: float = 1.0,
    disp_max: float = 30,
    disp_grid: float = 1,
    natom_lb: int = 30,
    natom_ub: int = 150,
    str_name: int = -1,
):
    with open("struct_size.yaml", "w"):
        pass

    for poscar in poscars:
        gen_mlp_data(
            poscar=poscar,
            per_volume=per_volume,
            disp_max=disp_max,
            disp_grid=disp_grid,
            natom_lb=natom_lb,
            natom_ub=natom_ub,
            str_name=str_name,
        )


def compress_vasprun(
    vasp_paths: list[str], output_dir: str = "compress_dft_data", num_process: int = 4
):
    valid_paths, vasprun_status = check_convergence(vasp_paths=vasp_paths)

    judge_list = Parallel(n_jobs=num_process)(
        delayed(compress)(vasp_path + "/vasprun.xml", output_dir)
        for vasp_path in valid_paths
    )
    vasprun_status["success"] += sum(judge_list)
    vasprun_status["parse"] += len(judge_list) - sum(judge_list)

    with open("dataset_status.yaml", "a") as f:
        print(f"{os.path.dirname(vasp_paths[0])}:", file=f)
        print(f" - input:              {len(vasp_paths)}", file=f)
        print(f"   success:            {vasprun_status['success']}", file=f)
        print(f"   failed_calculation: {vasprun_status['fail']}", file=f)
        print(f"   failed_iteration:   {vasprun_status['fail_iteration']}", file=f)
        print(f"   failed_parse:       {vasprun_status['parse']}", file=f)

    print(f"{os.path.dirname(vasp_paths[0])}:")
    print(f" - input:              {len(vasp_paths)} structure")
    print(f"   success:            {vasprun_status['success']} structure")
    print(f"   failed calculation: {vasprun_status['fail']} structure")
    print(f"   failed iteration:   {vasprun_status['fail_iteration']} structure")
    print(f"   failed parse:       {vasprun_status['parse']} structure")


def divide_dft_dataset(
    target_dirs: str,
    threshold_e_high: float = 10.0,  # in eV/atom
    threshold_e_low: Optional[float] = None,
    threshold_f_small: float = 3.0,  # in eV/ang
    threshold_f_normal: float = 10.0,
    threshold_f_large: float = 100.0,
    threshold_s_large: float = 200.0,  # in GPa
    threshold_s_small: Optional[float] = None,
    include_stress: bool = False,
    divide_ratio: float = 0.1,
):
    vasprun_paths = []
    for target_dir in target_dirs:
        vasprun_paths.extend(sorted(glob.glob(target_dir + "/*")))

    vasprun_dict = divide_dataset(
        vasprun_paths=vasprun_paths,
        threshold_e_high=threshold_e_high,
        threshold_e_low=threshold_e_low,
        threshold_f_small=threshold_f_small,
        threshold_f_normal=threshold_f_normal,
        threshold_f_large=threshold_f_large,
        threshold_s_large=threshold_s_large,
        threshold_s_small=threshold_s_small,
        include_stress=include_stress,
    )

    output_dir = "dft_dataset"
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/dataset.yaml", "w") as f:
        print("arguments:", file=f)
        print("  path:", target_dir, file=f)
        print("  threshold_e_high:", threshold_e_high, file=f)
        print("  threshold_e_low:", threshold_e_low, file=f)
        print("  threshold_f_small:", threshold_f_small, file=f)
        print("  threshold_f_normal:", threshold_f_normal, file=f)
        print("  threshold_f_large:", threshold_f_large, file=f)
        print("  threshold_s_large:", threshold_s_large, file=f)
        print("  threshold_s_small:", threshold_s_small, file=f)
        print("  include_stress:", include_stress, file=f)
        print("", file=f)

    with open(f"{output_dir}/n_data.yaml", "w") as f:
        pass

    for data_name, vasprun_list in vasprun_dict.items():
        if len(vasprun_list) == 0:
            continue

        train_data, test_data = divide_train_test(
            data_name=data_name,
            vasprun_list=vasprun_list,
            divide_ratio=divide_ratio,
            output_dir=output_dir,
        )

        with open(f"{output_dir}/n_data.yaml", "a") as f:
            print(f"{data_name}:", file=f)
            print("  - train_data:", len(train_data), file=f)
            print("  - test_data:", len(test_data), file=f)

        with open(f"{output_dir}/dataset.yaml", "a") as f:
            print(f"{data_name}:", file=f)
            print("  train:", file=f)
            for p in train_data:
                print(f"    - {p}", file=f)
            print("  test:", file=f)
            for p in test_data:
                print(f"    - {p}", file=f)

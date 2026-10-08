import glob
import os
import shutil

import pytest
import yaml

from rsspolymlp.api.rsspolymlp import rss_opt, rss_summarize, rss_uniq_struct

test_dir = os.path.dirname(os.path.abspath(__file__))
potential = [
    os.path.join(
        test_dir,
        "../files/silicon/potentials/polymlp.lammps.1",
    ),
    os.path.join(
        test_dir,
        "../files/silicon/potentials/polymlp.lammps.2",
    ),
]


@pytest.mark.parametrize("pressure", [0.0, 100.0])
def test_rss_mlp(tmp_path, pressure):
    init_poscars_dir = os.path.join(
        test_dir,
        f"../files/silicon/opt_struct/p/{pressure}GPa",
    )

    work_dir = os.path.join(tmp_path, f"{pressure}GPa")
    os.makedirs(work_dir)
    os.chdir(work_dir)

    rss_opt(
        pot=potential,
        init_poscars_dir=init_poscars_dir,
        pressure=pressure,
        n_opt_str=100,
        gtol=1e-4,
        c_maxiter=0,
        not_stop_rss=True,
    )
    rss_uniq_struct()

    with open("rss_result/rss_results.yaml") as f:
        yaml_data = yaml.safe_load(f)

    if pressure == 0.0:
        expected_energy_diff = [
            0.00,
            5.19,
            7.08,
            10.83,
            44.79,
            56.47,
            58.09,
            63.62,
            64.89,
            65.39,
            141.02,
            247.53,
        ]
    elif pressure == 100.0:
        expected_energy_diff = [
            0.00,
            5.33,
            5.90,
            7.08,
            8.50,
            8.76,
            11.99,
            14.48,
            16.89,
            44.35,
            74.18,
            190.28,
            287.41,
        ]

    assert yaml_data["general_information"]["pressure_GPa"] == pressure
    assert (
        yaml_data["general_information"]["num_initial_structures"]
        == len(expected_energy_diff) * 5
    )
    assert (
        yaml_data["general_information"]["num_optimized_structures"]
        == len(expected_energy_diff) * 5
    )
    assert len(yaml_data["unique_structures"]) == len(expected_energy_diff)
    assert (
        yaml_data["invalid_layer_structures"]["valid_struct"]
        == len(expected_energy_diff) * 5
    )
    assert yaml_data["error_counts"]["total"] == 0
    assert yaml_data["invalid_layer_structures"]["invalid_struct"] == 0

    for i, data in enumerate(yaml_data["unique_structures"]):
        assert data["n_duplicates"] == 5
        assert data["energy_diff_meV_per_atom"] == pytest.approx(
            expected_energy_diff[i]
        )


@pytest.mark.parametrize("pressure", [0.0, 100.0])
def test_rss_summarize(tmp_path, pressure):
    init_poscars_dirs = glob.glob(
        os.path.join(
            test_dir,
            f"../files/silicon/opt_struct/pn/{pressure}GPa/*",
        )
    )

    for _dir in init_poscars_dirs:
        work_dir = os.path.join(tmp_path, f"{pressure}GPa/{_dir.split('/')[-1]}")
        os.makedirs(work_dir)
        os.chdir(work_dir)

        rss_opt(
            pot=potential,
            init_poscars_dir=_dir,
            pressure=pressure,
            n_opt_str=100,
            gtol=1e-4,
            c_maxiter=0,
            not_stop_rss=True,
        )
        rss_uniq_struct()

    os.chdir("../")
    result_paths = sorted(glob.glob("./*/rss_result/rss_results.json"))
    rss_summarize(
        result_paths=result_paths,
    )

    with open("Si1.yaml") as f:
        yaml_data = yaml.safe_load(f)

    if pressure == 0.0:
        expected_energy_diff = [
            0.00,
            5.19,
            7.08,
            10.83,
            44.79,
            56.47,
            58.09,
            63.62,
            64.89,
            65.39,
            141.02,
            247.53,
        ]
        expected_n_duplicates = [8, 2, 2, 4, 1, 1, 1, 1, 2, 1, 1, 1]
        num_opt = 25
    elif pressure == 100.0:
        expected_energy_diff = [
            0.00,
            5.33,
            5.90,
            7.08,
            8.50,
            8.76,
            11.99,
            14.48,
            16.89,
            44.35,
            74.18,
            190.28,
            287.41,
        ]
        expected_n_duplicates = [16, 2, 2, 3, 4, 4, 5, 3, 8, 1, 1, 1, 1]
        num_opt = 51

    assert yaml_data["general_information"]["num_optimized_structs"] == num_opt
    assert len(yaml_data["unique_structures"]) == len(expected_energy_diff)

    for i, data in enumerate(yaml_data["unique_structures"]):
        assert data["energy_diff_meV_per_atom"] == pytest.approx(
            expected_energy_diff[i]
        )
        assert data["n_duplicates"] == expected_n_duplicates[i]


@pytest.mark.parametrize("target_dir", ["5GPa_Bi6", "5GPa_Ca5Bi5"])
def test_rss_uniq_struct(tmp_path, target_dir):
    ref_path = f"{test_dir}/../files/Bi_based_alloy/Ca-Bi"
    with open(f"{ref_path}/{target_dir}/rss_result/rss_results.yaml") as f:
        ref_data = yaml.safe_load(f)

    n_duplicates_ref = []
    for data in ref_data["unique_structures"]:
        n_duplicates_ref.append(data["n_duplicates"])

    os.makedirs(f"{tmp_path}/{target_dir}/rss_result")
    os.chdir(f"{tmp_path}/{target_dir}")
    shutil.copytree(f"{ref_path}/{target_dir}/log", "log")
    shutil.copytree(f"{ref_path}/{target_dir}/opt_struct", "opt_struct")
    shutil.copy(f"{ref_path}/{target_dir}/rss_result/finish.dat", "rss_result")
    shutil.copy(f"{ref_path}/{target_dir}/rss_result/success.dat", "rss_result")

    rss_uniq_struct()
    with open("./rss_result/rss_results.yaml") as f:
        yaml_data = yaml.safe_load(f)

    n_duplicates = []
    for data in yaml_data["unique_structures"]:
        n_duplicates.append(data["n_duplicates"])

    assert n_duplicates_ref == n_duplicates

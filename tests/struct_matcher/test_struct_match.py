import glob
import os

import numpy as np
import pytest
import yaml

from rsspolymlp.analysis.struct_matcher.struct_match import matching_metric_tensor
from rsspolymlp.analysis.struct_matcher.test.gen_rand_lattice import (
    generate_random_lattice,
    generate_random_unimodular_matrix,
)
from rsspolymlp.api.rsspolymlp_utils import struct_compare, struct_matcher

test_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("assign_method", ["redrep", "greedy", "HK"])
def test_struct_matcher(tmp_path, assign_method):
    files = [
        f"{test_dir}/../files/struct_matcher/case1/POSCAR*",
        f"{test_dir}/../files/silicon/opt_struct/p/0.0GPa/*",
        f"{test_dir}/../files/silicon/opt_struct/p/100.0GPa/*",
    ]
    for file in files:
        dir_name = file.split("/")[-2]
        os.makedirs(f"{tmp_path}/{dir_name}")
        os.chdir(f"{tmp_path}/{dir_name}")
        struct_matcher(
            poscar_paths=glob.glob(file),
            assign_method=assign_method,
        )

        n_duplicates = []
        with open("unique_struct.yaml") as f:
            yaml_data = yaml.safe_load(f)
        for data in yaml_data["unique_structures"]:
            n_duplicates.append(data["n_duplicates"])

        if dir_name == "case1":
            assert sorted(n_duplicates) == [4, 7]
        if dir_name == "0.0GPa" or dir_name == "100.0GPa":
            assert n_duplicates == [5] * len(n_duplicates)


def test_struct_compare(tmp_path):
    files = [
        f"{test_dir}/../files/struct_matcher/case2/POSCAR*",
        f"{test_dir}/../files/struct_matcher/case3/POSCAR*",
    ]
    for file in files:
        os.makedirs(f"{tmp_path}/{file.split('/')[-2]}")
        os.chdir(f"{tmp_path}/{file.split('/')[-2]}")
        judge = struct_compare(
            poscar_paths=glob.glob(file),
        )
        assert judge is True


def test_metric_tensor_matching():
    crystal_systems = [
        "triclinic",
        "monoclinic",
        "orthorhombic",
        "tetragonal",
        "hexagonal",
        "rhombohedral",
        "cubic",
    ]
    for crystal_system in crystal_systems:
        for i in range(10):
            rand_axis = generate_random_lattice(crystal_system, 100)
            rand_U = generate_random_unimodular_matrix()
            trans_axis = rand_U @ rand_axis
            G1 = rand_axis @ rand_axis.T
            G2 = trans_axis @ trans_axis.T
            M = matching_metric_tensor(G1, G2, spg_number=1)
            judge = (np.array_equal(m, rand_U) for m in M)
            assert any(judge)

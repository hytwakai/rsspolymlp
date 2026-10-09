import glob
import os
import shutil

import pytest
import yaml

from rsspolymlp.api.rsspolymlp_utils import struct_compare, struct_matcher

test_dir = os.path.dirname(os.path.abspath(__file__))


def test_struct_matcher(tmp_path):
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
    files = [f"{test_dir}/../files/struct_matcher/case2/POSCAR*"]
    for file in files:
        os.makedirs(f"{tmp_path}/{file.split('/')[-2]}")
        os.chdir(f"{tmp_path}/{file.split('/')[-2]}")
        judge = struct_compare(
            poscar_paths=glob.glob(file),
        )
        assert judge is True

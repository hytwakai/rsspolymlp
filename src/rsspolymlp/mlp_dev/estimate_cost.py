import glob
import os
import shutil

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.io_polymlp import save_mlp
from pypolymlp.core.parser_polymlp_params import ParamsParser
from pypolymlp.utils.count_time import PolymlpCost


def make_input():
    param_paths = glob.glob("./polymlp*.in")
    count = 1
    for param_path in param_paths:
        shutil.copy(param_path, "polymlp.input")
        with open("polymlp.input", "a") as f:
            print("n_type 1", file=f)
            print("elements Ca", file=f)
        polymlp = PypolymlpCalc(require_mlp=False)
        polymlp.load_structures_from_files(
            poscars=[f"{base_dir}/model_selection/param/POSCAR_example"]
        )
        polymlp.run_features(
            develop_infile="./polymlp.input",
            features_force=False,
            features_stress=False,
        )
        polymlp.save_features()
        feature = np.load("features.npy")
        params = ParamsParser("polymlp.input", parse_vasprun_locations=False).params
        save_mlp(
            params,
            np.random.rand(feature.shape[1]),
            np.random.rand(feature.shape[1]),
            filename=f"proto.polymlp.yaml.{count}",
        )
        count += 1

    pot_path = "./proto.polymlp.yaml*"
    return pot_path


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        required=True,
        help="Directory paths containing polymlp.yaml.",
    )
    parser.add_argument(
        "--param_input",
        action="store_true",
        help="",
    )
    args = parser.parse_args()

    cwd_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for _path in args.path:
        if args.param_input:
            pot_path = make_input()
        else:
            pot_path = "./polymlp.yaml*"

        os.chdir(_path)
        pot = glob.glob(pot_path)
        PolymlpCost(pot=pot).run(n_calc=10)
        os.chdir(cwd_dir)

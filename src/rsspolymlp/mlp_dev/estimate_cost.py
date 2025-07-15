import glob
import shutil

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc
from pypolymlp.core.data_format import PolymlpStructure
from pypolymlp.core.io_polymlp import save_mlp
from pypolymlp.core.parser_polymlp_params import ParamsParser


def make_polymlp_yaml():
    param_paths = glob.glob("./polymlp*.in")
    count = 1
    for param_path in param_paths:
        shutil.copy(param_path, "polymlp.input")
        with open("polymlp.input", "a") as f:
            print("n_type 1", file=f)
            print("elements Z", file=f)

        polymlp = PypolymlpCalc(require_mlp=False)

        axis = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])
        positions = np.array([[0.0, 0.0, 0.0]]).T
        n_atoms = np.array([1])
        types = np.array([0])
        elements = ["Z"]
        volume = np.linalg.det(axis)
        unitcell = PolymlpStructure(
            axis,
            positions,
            n_atoms,
            elements,
            types,
            volume,
        )
        polymlp.structures = unitcell

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

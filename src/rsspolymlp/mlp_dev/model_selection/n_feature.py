import argparse
import glob
import os
import shutil

import numpy as np

from pypolymlp.api.pypolymlp_calc import PypolymlpCalc

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    default=None,
    help="Directory path including model parameter candidates",
)
parser.add_argument(
    "--n_elements",
    type=int,
    default=1,
    help="Number of element symbols",
)
args = parser.parse_args()

grid_path = args.path
path_all = sorted(glob.glob(grid_path + "/*"))

base_dir = os.path.dirname(os.path.abspath(__file__))
polymlp = PypolymlpCalc(require_mlp=False)
polymlp.load_structures_from_files(poscars=[f"{base_dir}/POSCAR_example"])

for path in path_all:
    n_feature = 0
    for file_name in [0, 1, 2, 3]:
        if file_name == 0:
            polymlp_in = f"{path}/polymlp.in"
        else:
            polymlp_in = f"{path}/polymlp{file_name}.in"

        if os.path.isfile(polymlp_in):
            shutil.copy(polymlp_in, "./polymlp.in")
        else:
            continue

        if args.n_elements == 1:
            with open("./polymlp.in", "a") as f:
                print("n_type 1", file=f)
                print("elements Ca", file=f)
        elif args.n_elements == 2:
            with open("./polymlp.in", "a") as f:
                print("n_type 2", file=f)
                print("elements Ca Bi", file=f)

        polymlp.run_features(
            develop_infile="./polymlp.in",
            features_force=False,
            features_stress=False,
        )
        polymlp.save_features()
        feature = np.load("./features.npy")
        n_feature += feature.shape[1]

        os.remove("./polymlp.in")
        os.remove("./features.npy")

    print(n_feature)
    with open(f"{path}/n_feature.dat", "w") as f:
        print(n_feature, file=f)
    print(path)

import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    default=None,
    help="Directory path including model parameter candidates",
)
parser.add_argument(
    "--max_n",
    type=int,
    default=80000,
    help="",
)
args = parser.parse_args()

grid_path = args.path
path_all = sorted(glob.glob(grid_path + "/*"))

count = 0
for path in path_all:
    with open(f"{path}/n_feature.dat", "r") as f:
        n_feature = int([i.strip() for i in f][0])
    if n_feature < args.max_n:
        count += 1
        os.makedirs(f"{grid_path}_reduce", exist_ok=True)
        shutil.copytree(f"{path}", f"{grid_path}_reduce/polymlp-" + str(count).zfill(4))

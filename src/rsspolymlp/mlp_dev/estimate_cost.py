import argparse
import glob
import os

from pypolymlp.utils.count_time import PolymlpCost

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    nargs="+",
    required=True,
    help="Directory paths containing polymlp.yaml.",
)
args = parser.parse_args()

for _path in args.path:
    os.chdir(_path)
    if os.path.isfile("polymlp_cost.yaml"):
        continue

    pot = glob.glob("./polymlp.yaml*")
    PolymlpCost(pot=pot).run(n_calc=10)

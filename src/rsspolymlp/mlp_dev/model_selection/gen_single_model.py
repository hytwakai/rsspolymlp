import argparse
import os

from rsspolymlp.mlp_dev.model_selection.pypolymlp_gridsearch import PolymlpGridSearch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="Selected model type for generating model parameter candidates (2, 3, 4, or pair)",
)
args = parser.parse_args()

model_type = args.model_type

if model_type is not None:
    os.makedirs(f"./polymlps_single_m{model_type}", exist_ok=True)
polymlp = PolymlpGridSearch(elements=("Z"))

if model_type == str(2):
    polymlp.set_params(
        cutoffs=(6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
        nums_gaussians=(7, 10, 13),
        model_types=(2),
        maxps=(2, 3),
        gaussian_width=1.0,
        gtinv=True,
        gtinv_order_ub=4,
        gtinv_maxl_ub=(12, 6, 1, 1, 1),
        gtinv_maxl_int=(6, 2, 1, 1, 1),
    )
elif model_type == str(3):
    polymlp.set_params(
        cutoffs=(6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
        nums_gaussians=(7, 10, 13, 16),
        model_types=(3),
        maxps=(2, 3),
        gaussian_width=1.0,
        gtinv=True,
        gtinv_order_ub=6,
        gtinv_maxl_ub=(12, 8, 4, 1, 1),
        gtinv_maxl_int=(12, 4, 4, 1, 1),
    )
elif model_type == str(4):
    polymlp.set_params(
        cutoffs=(6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
        nums_gaussians=(7, 10, 13, 16),
        model_types=(4),
        maxps=(2, 3),
        gaussian_width=1.0,
        gtinv=True,
        gtinv_order_ub=4,
        gtinv_maxl_ub=(12, 8, 4, 1, 1),
        gtinv_maxl_int=(6, 4, 4, 1, 1),
    )
elif model_type == "pair":
    polymlp.set_params(
        cutoffs=(6.0, 7.0, 8.0, 9.0, 10.0, 12.0),
        nums_gaussians=(7, 10, 13, 16),
        model_types=(2, 3, 4),
        maxps=(2, 3),
        gaussian_width=1.0,
        gtinv=False,
        gtinv_order_ub=4,
        gtinv_maxl_ub=(12, 8, 4, 1, 1),
        gtinv_maxl_int=(4, 4, 4, 1, 1),
    )

if not model_type == "pair":
    polymlp.enum_gtinv_models()
else:
    polymlp.enum_pair_models()
polymlp.save_models(path=f"./polymlps_single_m{model_type}")

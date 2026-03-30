import argparse
import os

import numpy as np

from pypolymlp.core.data_format import PolymlpModelParams, PolymlpParams
from rsspolymlp.mlp_dev.model_selection.pypolymlp_gridsearch import (
    GtinvAttrs,
    save_hybrid_models,
    save_hybrid_models_3,
)


def get_gtinv_maxl(model_type):
    if model_type == 2:
        gtinv_maxl = [
            np.array([6]),
            np.array([12]),
            np.array([6, 2]),
            np.array([12, 2]),
            np.array([6, 2, 1]),
            np.array([12, 2, 1]),
            np.array([6, 4]),
            np.array([12, 4]),
            np.array([6, 4, 1]),
            np.array([12, 4, 1]),
            np.array([6, 6]),
            np.array([12, 6]),
            np.array([6, 6, 1]),
            np.array([12, 6, 1]),
        ]
    elif model_type == 4:
        gtinv_maxl = [
            np.array([6]),
            np.array([12]),
            np.array([6, 4]),
            np.array([12, 4]),
            np.array([6, 4, 4]),
            np.array([12, 4, 4]),
            np.array([12, 8]),
            np.array([12, 8, 4]),
        ]
    else:
        gtinv_maxl = None

    return gtinv_maxl


def get_model(cutoff, model_type, n_gauss, gtinv_maxl):
    if cutoff < 6:
        cut_gauss = cutoff - 0.5
    else:
        cut_gauss = cutoff - 1.0

    model = PolymlpModelParams(
        cutoff=cutoff,
        model_type=model_type,
        max_p=2,
        max_l=max(gtinv_maxl),
        feature_type="gtinv",
        gtinv=GtinvAttrs(
            model_type=model_type,
            order=len(gtinv_maxl) + 1,
            max_l=gtinv_maxl,
        ),
        pair_params_in1=(1.0, 1.0, 1),
        pair_params_in2=(0, cut_gauss, n_gauss),
    )
    return model


def get_params(model):
    params = PolymlpParams(
        n_type=1,
        elements="Z",
        model=model,
        regression_alpha=(-4, 3, 8),
        include_force=True,
        include_stress=True,
    )
    return params


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=int,
    default=None,
    help="Selected model type for generating model parameter candidates (2 or 4)",
)
args = parser.parse_args()

model_type = args.model_type
gtinv_maxl = get_gtinv_maxl(model_type)

os.makedirs(f"./polymlps_hybrid_m{model_type}", exist_ok=True)

mlp_count = 1
for cut in (4.0, 5.0, 6.0, 7.0):
    for n_gauss in (7, 10, 13):
        for i in range(len(gtinv_maxl)):
            for model_back in [0, 2]:
                if i - model_back < 0:
                    continue

                params = []
                for h in range(2):
                    _gtinv_maxl = gtinv_maxl[i - model_back * h]
                    cutoff = cut * (1.0 + 0.5 * h)
                    model = get_model(cutoff, model_type, n_gauss, _gtinv_maxl)
                    params.append([get_params(model)])

                save_hybrid_models(
                    params[0], params[1], f"./polymlps_hybrid_m{model_type}", mlp_count
                )
                mlp_count += 1

for cut in (4.0, 5.0, 6.0, 7.0):
    for n_gauss in (7, 10, 13):
        for i in range(len(gtinv_maxl)):
            for model_back in [0, 2]:
                if i - model_back < 0:
                    continue

                params = []
                for h in range(3):
                    _gtinv_maxl = gtinv_maxl[i - model_back * h]
                    cutoff = cut * (1.0 + 0.5 * h)
                    model = get_model(cutoff, model_type, n_gauss, _gtinv_maxl)
                    params.append([get_params(model)])

                save_hybrid_models_3(
                    params[0],
                    params[1],
                    params[2],
                    f"./polymlps_hybrid_m{model_type}",
                    mlp_count,
                )
                mlp_count += 1

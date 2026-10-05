import os

import pytest

from rsspolymlp.api.rsspolymlp import rss_opt, rss_uniq_struct

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

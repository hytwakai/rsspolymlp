import os

from rsspolymlp.api.rsspolymlp import rss_opt, rss_uniq_struct

press = 5.0
base_dir = os.getcwd()

for target_dir in ["5GPa_Bi6", "5GPa_Ca5Bi5"]:
    os.chdir(f"{base_dir}/{target_dir}")
    potential = [
        "../../potentials/CaBi_polymlp.lammps",
    ]
    rss_opt(
        pot=potential,
        init_poscars_dir="opt_struct",
        pressure=press,
        n_opt_str=10000,
        gtol=1e-4,
        c_maxiter=0,
        not_stop_rss=True,
    )
    rss_uniq_struct()

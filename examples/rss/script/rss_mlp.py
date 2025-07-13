import os

import numpy as np

from rsspolymlp.api.rsspolymlp import rss_init_struct, rss_run_parallel, rss_uniq_struct

atom_num_set = np.arange(1, 9)
pressure_set = [0.0]
base_dir = os.getcwd()

for pressure in pressure_set:
    for n in atom_num_set:
        pairs = [(i, n - i) for i in range(n + 1)]
        for i, j in pairs:
            dir_path = os.path.join(base_dir, f"../rss_mlp/Al-Cu/{pressure}GPa/{i}_{j}")
            os.makedirs(dir_path, exist_ok=True)
            os.chdir(dir_path)

            print("- rss-init-struct")
            rss_init_struct(
                elements=["Al", "Cu"],
                atom_counts=[i, j],
                num_init_str=300,
            )

            print("- rss-parallel")
            potential = "../../../../potential/AlCu_polymlp.lammps"
            rss_run_parallel(
                pot=potential,
                pressure=pressure,
                n_opt_str=200,
            )

            print("- rss-uniq-struct")
            rss_uniq_struct()

            os.chdir(base_dir)
            print(f"{pressure}GPa/{i}_{j} finished")

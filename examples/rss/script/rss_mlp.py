import os
import subprocess

import numpy as np

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

            potential = "../../../../potential/AlCu_polymlp.lammps"
            subprocess.run(
                f"rss-init-struct --elements Al Cu --atom_counts {i} {j} --num_init_str 300",
                shell=True,
                check=True,
            )
            subprocess.run(
                f"rss-parallel --pot {potential} --num_opt_str 200 --pressure {pressure}",
                shell=True,
                check=True,
            )
            subprocess.run("rss-uniq-struct", shell=True, check=True)

            os.chdir(base_dir)
            print(f"{pressure}GPa/{i}_{j} finished")

import os
import subprocess
import shutil

import numpy as np

atom_num_set = np.arange(1, 9)
pressure_set = [0.0]
base_dir = os.getcwd()

for pressure in pressure_set:
    for n in atom_num_set:
        pairs = [(i, n - i) for i in range(n + 1)]
        for i, j in pairs:
            dir_path = os.path.join(base_dir, f"../myresult/rss_mlp/Al-Cu/{pressure}GPa/{i}_{j}")
            os.makedirs(dir_path, exist_ok=True)
            os.chdir(dir_path)

            if os.path.isfile("rss_result/finish.log"):
                subprocess.run("mv rss_result/finish.log rss_result/finish.dat", shell=True, check=True)
            if os.path.isfile("rss_result/success.log"):
                subprocess.run("mv rss_result/success.log rss_result/success.dat", shell=True, check=True)
            if os.path.isfile("rss_result/rss_results.log"):
                subprocess.run("rm rss_result/rss_results.log", shell=True, check=True)
            # subprocess.run("rss-uniq-struct --cutoff 8.0", shell=True, check=True)

            os.chdir(base_dir)
            print(f"{pressure}GPa/{i}_{j} finished")
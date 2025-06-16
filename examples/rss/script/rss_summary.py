import os
import subprocess

pressure_set = [0.0]
base_dir = os.getcwd()

for pressure in pressure_set:
    dir_path = f"../rss_summary/Al-Cu/{pressure}GPa"
    os.makedirs(dir_path, exist_ok=True)
    os.chdir(dir_path)

    rss_paths = f"../../../rss_mlp/Al-Cu/{pressure}GPa/*"
    print("- rss-summarize start")
    subprocess.run(
        f"rss-summarize --elements Al Cu --use_joblib --rss_paths {rss_paths}",
        shell=True,
        check=True,
    )
    print("- rss-summarize finished")
    print("- rss-ghost-minima")
    subprocess.run(
        "rss-ghost-minima --result_paths ./json/*",
        shell=True,
        check=True,
    )
    subprocess.run(
        "rss-ghost-minima --compare_dft --dft_dir ./ghost_minima_dft",
        shell=True,
        check=True,
    )
    print("- rss-phase-analysis")
    subprocess.run(
        (
            "rss-phase-analysis --elements Al Cu "
            "--result_paths ./json/* "
            "--ghost_minima_file ghost_minima/ghost_minima_detection.yaml "
            "--thresholds 10 20 30 40 50"
        ),
        shell=True,
        check=True,
    )
    print("- plot-binary")
    subprocess.run(
        "plot-binary --elements Al Cu --threshold 30",
        shell=True,
        check=True,
    )

    os.chdir(base_dir)

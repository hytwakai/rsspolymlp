import glob
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
    print("- rss-outlier")
    subprocess.run(
        "rss-outlier --result_paths ./json/*",
        shell=True,
        check=True,
    )
    print(
        f"Detect potential {len(glob.glob('./outlier/outlier_candidates/*'))} outliers"
    )
    print("- rss-outlier-detect")
    subprocess.run(
        "rss-outlier-detect --dft_path ./outlier_candidates_dft",
        shell=True,
        check=True,
    )
    print("- rss-phase-analysis")
    subprocess.run(
        (
            "rss-phase-analysis --elements Al Cu "
            "--result_paths ./json/* "
            "--outlier_file outlier/outlier_detection.log "
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

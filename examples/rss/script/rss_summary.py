import glob
import os

from rsspolymlp.api.api_plot import plot_binary
from rsspolymlp.api.api_rss_postprocess import (
    rss_ghost_minima_cands,
    rss_ghost_minima_validate,
    rss_phase_analysis,
    rss_summarize,
)

pressure_set = [0.0]
base_dir = os.getcwd()

for pressure in pressure_set:
    dir_path = f"../rss_summary/Al-Cu/{pressure}GPa"
    os.makedirs(dir_path, exist_ok=True)
    os.chdir(dir_path)

    print("- rss-summarize start")
    rss_paths = sorted(glob.glob(f"../../../rss_mlp/Al-Cu/{pressure}GPa/*"))
    rss_summarize(
        elements=["Al", "Cu"],
        rss_paths=rss_paths,
        use_joblib=True,
    )
    print("- rss-summarize finished")

    print("- rss-ghost-minima")
    rss_ghost_minima_cands(glob.glob("./json/*"))
    rss_ghost_minima_validate("./ghost_minima_dft")

    print("- rss-phase-analysis")
    rss_phase_analysis(
        elements=["Al", "Cu"],
        result_paths=glob.glob("./json/*"),
        ghost_minima_file="ghost_minima/ghost_minima_detection.yaml",
        thresholds=[10, 20, 30, 40, 50],
    )
    print("- plot-binary")
    plot_binary(elements=["Al", "Cu"], threshold=30)

    os.chdir(base_dir)

import glob
import os

from rsspolymlp.api.rsspolymlp import (
    rss_ghost_minima_cands,
    rss_ghost_minima_validate,
    rss_phase_analysis,
    rss_summarize,
)
from rsspolymlp.api.rsspolymlp_plot import plot_binary

pressure_set = [0.0]
base_dir = os.getcwd()

for pressure in pressure_set:
    dir_path = f"../rss_summary/Al-Cu/{pressure}GPa"
    os.makedirs(dir_path, exist_ok=True)
    os.chdir(dir_path)

    print("- summarize start")
    result_paths = sorted(
        glob.glob(f"../../../rss_mlp/Al-Cu/{pressure}GPa/*/rss_result/rss_results.json")
    )
    rss_summarize(
        result_paths=result_paths,
    )
    print("- summarize finished")

    print("- ghost-minima")
    rss_ghost_minima_cands(glob.glob("./json/*"))
    rss_ghost_minima_validate("./ghost_minima_dft")

    print("- phase-analysis")
    rss_phase_analysis(
        elements=["Al", "Cu"],
        input_paths=glob.glob("./json/*"),
        ghost_minima_file="ghost_minima/ghost_minima_detection.yaml",
        thresholds=[10, 20, 30, 40, 50],
    )
    print("- plot-binary")
    plot_binary(threshold=30)

    os.chdir(base_dir)

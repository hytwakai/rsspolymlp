import argparse
import os
import shutil

from rsspolymlp.analysis.rss_summarize import load_rss_results


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_paths",
        nargs="*",
        type=str,
        required=True,
        help="Path(s) to RSS result log file(s).",
    )
    args = parser.parse_args()

    os.makedirs("outlier_candidates", exist_ok=True)
    for res_path in args.result_paths:
        logname = os.path.basename(res_path).split(".log")[0]
        rss_results = load_rss_results(res_path, absolute_path=True, get_warning=True)

        for idx, result in enumerate(rss_results):
            if result.get("is_weak_outlier"):
                dest = f"outlier_candidates/POSCAR_{logname}_No{idx + 1}"
                shutil.copy(result["poscar"], dest)


if __name__ == "__main__":
    run()

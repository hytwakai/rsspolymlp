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

    # Prepare output directory: remove existing files if already exists
    out_dir = "outlier_candidates"
    if os.path.exists(out_dir):
        for filename in os.listdir(out_dir):
            if "POSCAR" in filename:
                file_path = os.path.join(out_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    else:
        os.makedirs(out_dir)

    # Copy weak outlier POSCARs
    for res_path in args.result_paths:
        logname = os.path.basename(res_path).split(".log")[0]
        rss_results = load_rss_results(res_path, absolute_path=True, get_warning=True)

        for idx, result in enumerate(rss_results):
            if result.get("is_weak_outlier"):
                dest = f"outlier_candidates/POSCAR_{logname}_No{idx + 1}"
                shutil.copy(result["poscar"], dest)


if __name__ == "__main__":
    run()

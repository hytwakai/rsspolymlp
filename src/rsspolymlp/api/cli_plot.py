import argparse

from rsspolymlp.api.api_plot import plot_binary


def run_plot_binary():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", nargs=2, type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    plot_binary(args.elements, threshold=args.threshold)

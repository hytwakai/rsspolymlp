import argparse

from rsspolymlp.api.rsspolymlp_plot import plot_binary


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Mode: Plotting RSS results in binary system",
    )

    parser.add_argument("--elements", nargs="+", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    if args.binary:
        plot_binary(args.elements, threshold=args.threshold)

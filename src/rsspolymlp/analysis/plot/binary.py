import argparse
import ast
import json

import numpy as np

from rsspolymlp.utils.matplot_util.custom_plt import CustomPlt
from rsspolymlp.utils.matplot_util.make_plot import MakePlot


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--elements",
        nargs=2,
        type=str,
        required=True,
        help="Two chemical elements, e.g., La Bi",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold value for energy above the convex hull in meV/atom ",
    )
    args = parser.parse_args()

    custom_template = CustomPlt(
        label_size=8,
        label_pad=3.0,
        legend_size=7,
        xtick_size=7,
        ytick_size=7,
        xtick_pad=3.0,
        ytick_pad=3.0,
    )
    plt = custom_template.get_custom_plt()
    plotter = MakePlot(
        plt=plt,
        column_size=1,
        height_ratio=0.8,
    )
    plotter.initialize_ax()

    plotter.set_visuality(n_color=4, n_line=4, n_marker=1, color_type="grad")

    phase_res = load_plot_data(args.threshold)
    plotter.ax_plot(
        phase_res["comp_ch"][:, 1],
        phase_res["fe_ch"],
        plot_type="closed",
        label=None,
        plot_size=0.7,
        line_size=1,
        zorder=2,
    )

    for key, _dict in phase_res["rss_result_fe"].items():
        plotter.set_visuality(n_color=3, n_line=0, n_marker=0, color_type="grad")
        if args.threshold is not None:
            _energies = phase_res["not_near_ch"][key]["formation_e"]
            _comps = np.full_like(_energies, fill_value=key[1])
            plotter.ax_scatter(
                _comps, _energies, plot_type="open", label=None, plot_size=0.4
            )

            plotter.set_visuality(n_color=1, n_line=0, n_marker=1)
            _energies = phase_res["near_ch"][key]["formation_e"]
            _comps = np.full_like(_energies, fill_value=key[1])
            plotter.ax_scatter(
                _comps, _energies, plot_type="open", label=None, plot_size=0.5
            )
        else:
            _energies = _dict["formation_e"]
            _comps = np.full_like(_energies, fill_value=key[1])
            plotter.ax_scatter(
                _comps, _energies, plot_type="open", label=None, plot_size=0.4
            )

    fe_min = np.min(phase_res["fe_ch"])
    plotter.finalize_ax(
        xlabel=rf"$x$ in {args.elements[0]}$_{{1-x}}${args.elements[1]}$_{{x}}$",
        ylabel="Formation energy (eV/atom)",
        x_limits=[0, 1],
        x_grid=[0.2, 0.1],
        y_limits=[fe_min * 1.1, 0],
    )

    plt.tight_layout()
    plt.savefig(
        f"phase_analysis/binary_plot_{args.elements[0]}{args.elements[1]}.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )


def load_plot_data(threshold):
    res = {}

    res["comp_ch"] = np.load("phase_analysis/data/comp_ch.npy")
    res["fe_ch"] = np.load("phase_analysis/data/fe_ch.npy")
    with open("phase_analysis/data/rss_result_fe.json", "r") as f:
        data = json.load(f)
    res["rss_result_fe"] = convert_json_to_ndarray(data)

    res["not_near_ch"] = None
    res["near_ch"] = None
    if threshold is not None:
        with open(
            f"phase_analysis/threshold_{threshold}meV/not_near_ch.json", "r"
        ) as f:
            data1 = json.load(f)
        with open(f"phase_analysis/threshold_{threshold}meV/near_ch.json", "r") as f:
            data2 = json.load(f)
        res["not_near_ch"] = convert_json_to_ndarray(data1)
        res["near_ch"] = convert_json_to_ndarray(data2)

    return res


def convert_json_to_ndarray(data):
    converted = {
        ast.literal_eval(k): {key: np.array(val) for key, val in v.items()}
        for k, v in data.items()
    }
    return converted

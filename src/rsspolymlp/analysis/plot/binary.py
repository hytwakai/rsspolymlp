import argparse

import numpy as np

from rsspolymlp.analysis.convex_hull import ConvexHullAnalyzer
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
        "--result_paths",
        nargs="+",
        type=str,
        required=True,
        help="Paths to RSS result log files",
    )
    parser.add_argument(
        "--outlier_file",
        type=str,
        default=None,
        help="Path to file listing outlier structure names to be excluded",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Threshold values for energy above the convex hull in meV/atom ",
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

    ch_analyzer = ConvexHullAnalyzer(
        args.elements, args.result_paths, args.outlier_file
    )
    ch_analyzer.run_calc()

    fe_ch = ch_analyzer.fe_ch
    comp_ch = ch_analyzer.comp_ch
    rss_result_fe = ch_analyzer.rss_result_fe

    if args.thresholds is not None:
        threshold_list = sorted(args.thresholds)
        for threshold in threshold_list:
            near_ch, not_near_ch = ch_analyzer.get_struct_near_ch(threshold)

    plotter.set_visuality(n_color=4, n_line=4, n_marker=1, color_type="grad")
    plotter.ax_plot(
        comp_ch[:, 1],
        fe_ch,
        plot_type="closed",
        label=None,
        plot_size=0.7,
        line_size=1,
        zorder=2,
    )
    fe_min = np.min(fe_ch)

    for key, _dict in rss_result_fe.items():
        plotter.set_visuality(n_color=3, n_line=0, n_marker=0, color_type="grad")
        if args.thresholds is not None:
            _energies = not_near_ch[key]["formation_e"]
            _comps = np.full_like(_energies, fill_value=key[1])
            plotter.ax_scatter(
                _comps, _energies, plot_type="open", label=None, plot_size=0.4
            )

            plotter.set_visuality(n_color=1, n_line=0, n_marker=1)
            _energies = near_ch[key]["formation_e"]
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

    plotter.finalize_ax(
        xlabel=rf"$x$ in {args.elements[0]}$_{{1-x}}${args.elements[1]}$_{{x}}$",
        ylabel="Formation energy (eV/atom)",
        x_limits=[0, 1],
        x_grid=[0.2, 0.1],
        y_limits=[fe_min * 1.1, 0],
    )

    plt.tight_layout()
    plt.savefig(
        f"ch/ch_plot_{args.elements[0]}{args.elements[1]}.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )


if __name__ == "__main__":
    run()

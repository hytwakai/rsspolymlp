import argparse
import os

import numpy as np


def pareto_front(points: np.ndarray):
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]

    _pareto_front_idx = []
    min_obj2 = np.inf
    for i in range(len(sorted_points)):
        if sorted_points[i][1] < min_obj2:
            _pareto_front_idx.append(i)
            min_obj2 = sorted_points[i][1]

    pareto_front_idx = sorted_idx[_pareto_front_idx]
    return pareto_front_idx


parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    nargs="+",
    required=True,
    help="Directory paths containing polymlp_error.yaml and polymlp_cost.yaml.",
)
parser.add_argument(
    "--plot",
    action="store_true",
    help="If enabled, the errors and costs of the polynomial MLPs are shown in a plot.",
)
parser.add_argument(
    "--rmse_path",
    type=str,
    default="test/close_minima",
    help="A part of the path name of the dataset used to compute the energy RMSE "
    "for identifying Pareto-optimal MLPs.",
)
args = parser.parse_args()

cwd_path = os.getcwd()
mlp_paths = args.path

res_dict = {"cost": [], "e_rmse": [], "mlp_name": []}
for mlp_path in mlp_paths:
    os.chdir(mlp_path)

    if not os.path.isfile("polymlp_cost.yaml") or not os.path.isfile(
        "polymlp_error.yaml"
    ):
        print(mlp_path, "failed")
        os.chdir(cwd_path)
        continue

    with open("polymlp_cost.yaml") as f:
        lines = [line.strip() for line in f]
    res_dict["cost"].append(
        next(float(line.split()[-1]) for line in lines if "single_core:" in line)
    )
    with open("polymlp_error.yaml") as f:
        lines = [line.strip() for line in f]
    res_dict["e_rmse"].append(
        next(
            float(lines[i + 1].split()[-1])
            for i, line in enumerate(lines)
            if args.rmse_path in line
        )
    )
    res_dict["mlp_name"].append(mlp_path)

    os.chdir(cwd_path)

sort_idx = np.argsort(res_dict["cost"])
res_dict = {key: np.array(_list)[sort_idx] for key, _list in res_dict.items()}

rmse_time = []
for i in range(len(res_dict["cost"])):
    rmse_time.append([res_dict["cost"][i], res_dict["e_rmse"][i]])

pareto_idx = pareto_front(np.array(rmse_time))
not_pareto_idx = np.ones(len(rmse_time), dtype=bool)
not_pareto_idx[pareto_idx] = False

os.makedirs("analyze_pareto", exist_ok=True)
os.chdir("analyze_pareto")

with open("pareto_optimum.yaml", "w") as f:
    print("units:", file=f)
    print("  cost:        'msec/atom/step'", file=f)
    print("  energy_rmse: 'meV/atom'", file=f)
    print("", file=f)
    print("pareto_optimum:", file=f)
    for idx in pareto_idx:
        print(f"  {res_dict['mlp_name'][idx]}:", file=f)
        print(f"    cost:        {res_dict['cost'][idx]}", file=f)
        print(f"    energy_rmse: {res_dict['e_rmse'][idx]}", file=f)

if args.plot:
    from rsspolymlp.utils.matplot_util.custom_plt import CustomPlt
    from rsspolymlp.utils.matplot_util.make_plot import MakePlot

    custom_template = CustomPlt(
        label_size=8,
        label_pad=4.0,
        legend_size=7,
        xtick_size=7,
        ytick_size=7,
        xtick_pad=4.0,
        ytick_pad=4.0,
    )
    plt = custom_template.get_custom_plt()
    plotter = MakePlot(
        plt=plt,
        column_size=1,
        height_ratio=1,
    )
    plotter.initialize_ax()

    plotter.set_visuality(n_color=4, n_line=4, n_marker=1, color_type="grad")

    plotter.set_visuality(n_color=3, n_line=0, n_marker=0, color_type="grad")
    plotter.ax_scatter(
        res_dict["cost"][not_pareto_idx],
        res_dict["e_rmse"][not_pareto_idx],
        plot_type="open",
        label=None,
        plot_size=0.5,
    )

    plotter.set_visuality(n_color=1, n_line=-1, n_marker=1)
    plotter.ax_plot(
        res_dict["cost"][pareto_idx],
        res_dict["e_rmse"][pareto_idx],
        plot_type="closed",
        label=None,
        plot_size=0.7,
    )

    plotter.finalize_ax(
        xlabel="Computational time (ms/step/atom) (single CPU core)",
        ylabel="RMSE (meV/atom)",
        x_limits=[1e-2, 30],
        y_limits=[0, 30],
        xlog=True,
    )

    plt.tight_layout()
    plt.savefig(
        "./pareto_opt_mlp.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )

import os

import numpy as np


def parse_mlp_property(mlp_paths: list[str]):
    cwd_path = os.getcwd()
    res_dict = {"cost": [], "rmse": [], "mlp_name": []}

    for mlp_path in mlp_paths:
        os.chdir(mlp_path)

        if not os.path.isfile("polymlp_cost.yaml") or not os.path.isfile(
            args.error_path
        ):
            print(mlp_path, "failed")
            os.chdir(cwd_path)
            continue

        with open("polymlp_cost.yaml") as f:
            lines = [line.strip() for line in f]
        res_dict["cost"].append(
            next(float(line.split()[-1]) for line in lines if "single_core:" in line)
        )
        with open(args.error_path) as f:
            lines = [line.strip() for line in f]
        for i, line in enumerate(lines):
            if args.rmse_path in line:
                res_dict["rmse"].append(
                    [float(lines[i + 1].split()[-1]), float(lines[i + 2].split()[-1])]
                )
        res_dict["mlp_name"].append(mlp_path)

        os.chdir(cwd_path)

    return res_dict


def pareto_front(points: np.ndarray):
    sorted_idx = np.argsort(points[:, 0])
    sorted_points = points[sorted_idx]

    pareto = []
    pareto_val = []
    for i, p in enumerate(sorted_points):
        if pareto_val == []:
            pareto.append(i)
            pareto_val.append(p)
        elif not any(p > o for p, o in zip(p[1:], np.array(pareto_val)[-1, 1:])):
            pareto.append(i)
            pareto_val.append(p)

    pareto_front_idx = sorted_idx[pareto]
    return pareto_front_idx


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        nargs="+",
        required=True,
        help="Directory paths containing polymlp_error.yaml and polymlp_cost.yaml.",
    )
    # Optional argumants
    parser.add_argument(
        "--error_path",
        type=str,
        default="polymlp_error.yaml",
        help="File name for storing MLP prediction errors.",
    )
    parser.add_argument(
        "--rmse_path",
        type=str,
        default="test/close_minima",
        help="A part of the path name of the dataset used to compute the energy RMSE "
        "for identifying Pareto-optimal MLPs.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If enabled, the errors and costs of the polynomial MLPs are shown in a plot.",
    )
    parser.add_argument(
        "--include_force",
        action="store_true",
        help="Filtered Pareto-optimal MLPs are shown as closed squares in the plot.",
    )
    parser.add_argument(
        "--rmse_max",
        type=float,
        default=30,
        help="Y axis maximum in the plot",
    )
    args = parser.parse_args()

    mlp_paths = args.path

    res_dict = parse_mlp_property(mlp_paths)

    sort_idx = np.argsort(res_dict["cost"])
    res_dict = {key: np.array(_list)[sort_idx] for key, _list in res_dict.items()}

    rmse_e_time = []
    for i in range(len(res_dict["cost"])):
        rmse_e_time.append([res_dict["cost"][i], res_dict["rmse"][i][0]])
    pareto_e_idx = pareto_front(np.array(rmse_e_time))
    not_pareto_idx = np.ones(len(rmse_e_time), dtype=bool)
    not_pareto_idx[pareto_e_idx] = False

    rmse_ef_time = []
    for i in pareto_e_idx:
        rmse_ef_time.append(
            [res_dict["cost"][i], res_dict["rmse"][i][0], res_dict["rmse"][i][1]]
        )
    _pareto_ef_idx = pareto_front(np.array(rmse_ef_time))
    pareto_ef_idx = np.array(pareto_e_idx)[_pareto_ef_idx]

    os.makedirs("analyze_pareto", exist_ok=True)
    os.chdir("analyze_pareto")

    with open("pareto_optimum.yaml", "w") as f:
        print("units:", file=f)
        print("  cost:        'msec/atom/step'", file=f)
        print("  rmse_energy: 'meV/atom'", file=f)
        print("  rmse_force:  'eV/angstrom'", file=f)
        print("", file=f)
        print("pareto_optimum:", file=f)
        for idx in pareto_e_idx:
            print(f"  {res_dict['mlp_name'][idx]}:", file=f)
            print(f"    cost:        {res_dict['cost'][idx]}", file=f)
            print(f"    rmse_energy: {res_dict['rmse'][idx][0]}", file=f)
            print(f"    rmse_force:  {res_dict['rmse'][idx][1]}", file=f)
        print("", file=f)
        print("# Filter out solutions with worse force error at higher cost", file=f)
        print("pareto_optimum_include_force:", file=f)
        for idx in pareto_ef_idx:
            print(f"  {res_dict['mlp_name'][idx]}:", file=f)
            print(f"    cost:        {res_dict['cost'][idx]}", file=f)
            print(f"    rmse_energy: {res_dict['rmse'][idx][0]}", file=f)
            print(f"    rmse_force:  {res_dict['rmse'][idx][1]}", file=f)

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

        plotter.set_visuality(n_color=3, n_line=0, n_marker=0, color_type="grad")
        plotter.ax_scatter(
            res_dict["cost"][not_pareto_idx],
            res_dict["rmse"][not_pareto_idx][:, 0],
            plot_type="open",
            label=None,
            plot_size=0.5,
        )

        if args.include_force:
            close_index = pareto_ef_idx
        else:
            close_index = pareto_e_idx
        plotter.set_visuality(n_color=1, n_line=0, n_marker=1)
        plotter.ax_plot(
            res_dict["cost"][pareto_e_idx],
            res_dict["rmse"][pareto_e_idx][:, 0],
            plot_type="open",
            label=None,
            plot_size=0.7,
        )
        plotter.set_visuality(n_color=1, n_line=-1, n_marker=1)
        plotter.ax_plot(
            res_dict["cost"][close_index],
            res_dict["rmse"][close_index][:, 0],
            plot_type="closed",
            label=None,
            plot_size=0.7,
        )

        plotter.finalize_ax(
            xlabel="Computational time (ms/step/atom) (single CPU core)",
            ylabel="RMSE (meV/atom)",
            x_limits=[1e-2, 30],
            y_limits=[0, args.rmse_max],
            xlog=True,
        )

        plt.tight_layout()
        plt.savefig(
            "./pareto_opt_mlp.png",
            bbox_inches="tight",
            pad_inches=0.01,
            dpi=600,
        )

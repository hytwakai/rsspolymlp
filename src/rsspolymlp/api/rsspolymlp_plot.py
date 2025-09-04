import glob
import json
import os

import numpy as np

from pypolymlp.core.interface_vasp import Vasprun
from pypolymlp.core.units import EVtoGPa
from rsspolymlp.analysis.load_plot_data import load_plot_data
from rsspolymlp.common.atomic_energy import atomic_energy
from rsspolymlp.mlp_dev.pareto_opt_mlp import pareto_front, parse_mlp_property
from rsspolymlp.utils.matplot_util.custom_plt import CustomPlt
from rsspolymlp.utils.matplot_util.make_plot import MakePlot


def plot_binary(threshold=None):
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

    phase_res = load_plot_data(threshold=threshold)
    plotter.ax_plot(
        phase_res["hull_comp"][:, 1],
        phase_res["hull_e"],
        plot_type="closed",
        label=None,
        plot_size=0.7,
        line_size=0.7,
        zorder=2,
    )

    for key, _dict in phase_res["composition_data"].items():
        plotter.set_visuality(n_color=3, n_line=0, n_marker=0, color_type="grad")
        if threshold is not None:
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

    elements = phase_res["elements"]
    fe_min = np.min(phase_res["hull_e"])
    plotter.finalize_ax(
        xlabel=rf"$x$ in {elements[0]}$_{{1-x}}${elements[1]}$_{{x}}$",
        ylabel="Formation energy (eV/atom)",
        x_limits=[0, 1],
        x_grid=[0.2, 0.1],
        y_limits=[fe_min * 1.1, 0],
    )

    plt.tight_layout()
    plt.savefig(
        f"phase_analysis/binary_plot_{elements[0]}{elements[1]}.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )


def pareto_opt_mlp(
    mlp_paths: list[str],
    error_path: str = "polymlp_error.yaml",
    rmse_path: str = "test/minima-close",
    include_force: bool = False,
    rmse_max: float = 30,
):

    res_dict = parse_mlp_property(mlp_paths, error_path=error_path, rmse_path=rmse_path)

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

    if include_force:
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
        y_limits=[0, rmse_max],
        xlog=True,
    )

    plt.tight_layout()
    plt.savefig(
        "./pareto_opt_mlp.png",
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=600,
    )


def plot_rss_error(
    mlp_jsons: list[str],
    dft_dirs: list[str],
    stress_tensor: bool = False,
    e_low: float = -5,
    e_high: float = 10,
    file_name: str = "rss_error.png",
):
    energies = []
    poscar_name = []

    for json_path in mlp_jsons:
        logname = os.path.basename(json_path).split(".json")[0]
        with open(json_path) as f:
            loaded_dict = json.load(f)
        rss_results = loaded_dict["rss_results"]
        pressure = loaded_dict["pressure"]

        for res in rss_results:
            poscar_name.append(f"POSCAR_{logname}_No{res['struct_no']}")
            mlp_energy = res["energy"]
            if not stress_tensor:
                st = res["structure"]
                vol_per_atom = st["volume"] / len(st["elements"])
                mlp_energy -= pressure * vol_per_atom / EVtoGPa

            energies.append({"poscar_name": poscar_name, "mlp_energy": mlp_energy})

    for dft_dir in dft_dirs:
        base_dir = os.getcwd()
        os.chdir(dft_dir)
        os.makedirs("rss_error", exist_ok=True)

        if stress_tensor and os.path.isfile("rss_error/enthalpy.npy"):
            dft_result = np.load("rss_error/enthalpy.npy")
        elif not stress_tensor and os.path.isfile("rss_error/energy.npy"):
            dft_result = np.load("rss_error/energy.npy")
        else:
            dft_result = []
            dir_glob = glob.glob(dft_dir + "/*")
            for dft_path in dir_glob:
                _poscar_name = dft_path.split("/")[-1]
                if _poscar_name in set(poscar_name):
                    try:
                        vasprun = Vasprun(dft_path + "/vasprun.xml")
                    except Exception:
                        print(dft_path + "/vasprun.xml error")
                        continue
                    dft_energy = vasprun.energy
                    polymlp_st = vasprun.structure
                    for element in polymlp_st.elements:
                        dft_energy -= atomic_energy(element)
                    dft_energy /= len(polymlp_st.elements)

                    if stress_tensor:
                        pressure = np.mean(
                            [(vasprun.stress / 10).tolist()[i][i] for i in range(3)]
                        )
                        vol_per_atom = polymlp_st.volume / len(polymlp_st.elements)
                        dft_energy += pressure * vol_per_atom / EVtoGPa
                        np_file = "rss_error/enthalpy.npy"
                    else:
                        np_file = "rss_error/energy.npy"
                    dft_result.append((_poscar_name, dft_energy))

            np.save(np_file, np.array(dft_result))

        for res in dft_result:
            energies[poscar_name.index(res[0])]["dft_energy"] = float(res[1])
        os.chdir(base_dir)

    mlp_energy = []
    dft_energy = []
    for _dict in energies:
        if "mlp_energy" in _dict and "dft_energy" in _dict:
            mlp_energy.append(_dict["mlp_energy"])
            dft_energy.append(_dict["dft_energy"])

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
        column_size=0.9,
        height_ratio=1.0,
    )
    plotter.initialize_ax()

    plotter.set_visuality(n_color=3, n_line=0, n_marker=0, color_type="grad")
    plotter.ax_plot([-10, 10], [-10, 10], plot_type="open", label=None, plot_size=0.5)

    plotter.set_visuality(n_color=4, n_line=4, n_marker=0, color_type="grad")
    plotter.ax_scatter(
        dft_energy,
        mlp_energy,
        plot_type="open",
        label=None,
        plot_size=0.9,
        zorder=2,
    )

    plotter.finalize_ax(
        xlabel="DFT energy (eV/atom)",
        ylabel="MLP energy (eV/atom)",
        x_limits=[e_low, e_high],
        y_limits=[e_low, e_high],
    )

    ax = plotter.get_ax
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(
        file_name,
        bbox_inches="tight",
        pad_inches=0.01,
        dpi=500,
    )

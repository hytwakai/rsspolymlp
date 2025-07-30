import logging
import warnings

import numpy as np
import scipy
from scipy.interpolate import interp1d
from scipy.optimize import leastsq

from pypolymlp.core.units import EVtoGPa


def vinet(v, *p):
    """Return Vinet EOS.
    p[0] = E_0
    p[1] = B_0
    p[2] = B'_0
    p[3] = V_0
    """
    x = (v / p[3]) ** (1.0 / 3)
    xi = 3.0 / 2 * (p[2] - 1)
    return p[0] + (
        9 * p[1] * p[3] / (xi**2) * (1 + (xi * (1 - x) - 1) * np.exp(xi * (1 - x)))
    )


class EOSFit:

    def __init__(
        self,
        energies: np.ndarray,
        volumes: np.ndarray,
        init_parameter: list[float] = None,
        pressure: float = None,
        volume_range: list[float] = None,
    ):
        """
        Fit an equation of state (EOS) to energy-volume data and
        enable Gibbs energy interpolation at arbitrary pressures.
        """
        sort_idx = np.argsort(-np.array(volumes))
        self._energies = np.array(energies)[sort_idx]
        self._volumes = np.array(volumes)[sort_idx]

        self._eos = vinet
        self.parameters = None
        self._volume_range = (
            volume_range
            if volume_range is not None
            else [np.min(self._volumes) * 0.99, np.max(self._volumes) * 1.01]
        )

        if init_parameter is None:
            # Default initial parameters: [E0, B0, B0', V0]
            self._init_parameter = [
                energies[len(energies) // 2],
                0.6,
                4.0,
                volumes[len(volumes) // 2],
            ]
        else:
            self._init_parameter = init_parameter

        if pressure is not None:
            self._energies += pressure * self._volumes / EVtoGPa

        self.fit()
        self.interpolate_gibbs_from_pressure(self._volume_range)

    def fit(self):
        """Fit EOS parameters to energy-volume data using least squares."""
        warnings.filterwarnings("error")

        def residuals(p, eos, v, e):
            return eos(v, *p) - e

        try:
            result = leastsq(
                residuals,
                self._init_parameter,
                args=(self._eos, self._volumes, self._energies),
                full_output=1,
            )
        except RuntimeError:
            logging.exception("Fitting to EOS failed.")
            raise
        except (RuntimeWarning, scipy.optimize.optimize.OptimizeWarning):
            logging.exception("Difficulty in fitting to EOS.")
            raise
        else:
            self.parameters = result[0]

        print(f"RMSE of EOS fit (energy): {self.rmse*1000:.6f} meV")

    @property
    def rmse(self) -> float:
        """Return RMSE between fitted EOS and input energy data."""
        if self.parameters is None:
            raise ValueError("Fit must be performed before calculating RMSE.")
        predicted = self._eos(self._volumes, *self.parameters)
        error = predicted - self._energies
        return np.sqrt(np.mean(error**2))

    def interpolate_gibbs_from_pressure(
        self, volume_range: list[float], n_grid: int = None
    ):
        """
        Precompute interpolation function for G(P) using volume range.
        Stores:
            self.g_interp: interpolator
            self.pressure_lb / pressure_ub: pressure bounds
        """
        if n_grid is None:
            # Pressure grid is set with ~0.01 GPa resolution.
            pressures, _ = self._get_pressure_and_gibbs_from_volumes(
                [min(volume_range), max(volume_range)]
            )
            pressure_range = abs(pressures[1] - pressures[0])
            n_grid = int(round(pressure_range * 100))

        volumes = np.linspace(min(volume_range), max(volume_range), n_grid)
        pressures, gibbs_energies = self._get_pressure_and_gibbs_from_volumes(volumes)

        sort_idx = np.argsort(pressures)
        sorted_pressures = pressures[sort_idx]
        sorted_gibbs = gibbs_energies[sort_idx]

        self.pressure_lb = sorted_pressures[0]
        self.pressure_ub = sorted_pressures[-1]

        self.g_interp = interp1d(
            sorted_pressures,
            sorted_gibbs,
            kind="linear",
            bounds_error=True,
        )

    def _get_pressure_and_gibbs_from_volumes(
        self, volumes: np.ndarray, eps: float = 1e-4
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate pressure and Gibbs energy for each volume using numerical derivatives."""
        gibbs_list = []
        pressure_list = []
        energies = self.get_energy(volumes)

        for vol, e in zip(volumes, energies):
            e_forward, e_backward = self.get_energy([vol + eps, vol - eps])
            dE_dV = (e_forward - e_backward) / (2 * eps)
            pressure_eVA3 = -dE_dV
            gibbs = e + pressure_eVA3 * vol
            pressure_GPa = pressure_eVA3 * EVtoGPa
            gibbs_list.append(gibbs)
            pressure_list.append(pressure_GPa)

        return np.array(pressure_list), np.array(gibbs_list)

    def get_energy(self, volumes: np.ndarray) -> np.ndarray:
        """Return predicted energy values for given volumes using fitted EOS."""
        return vinet(np.array(volumes), *self.parameters)

    def get_gibbs(self, pressures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return interpolated Gibbs free energy values at given pressures in GPa."""
        pressures = np.asarray(pressures)
        in_range_mask = (self.pressure_lb <= pressures) & (
            pressures <= self.pressure_ub
        )
        if not np.any(in_range_mask):
            raise ValueError(
                "All provided pressures are outside the interpolation range "
                f"[{self.pressure_lb:.3f}, {self.pressure_ub:.3f}] GPa."
            )
        valid_pressures = pressures[in_range_mask]
        return valid_pressures, self.g_interp(valid_pressures)

    def rmse_from_enthalpy_data(
        self,
        pressures: np.ndarray,
        reference_gibbs: np.ndarray,
    ) -> float:
        """
        Compute RMSE between interpolated Gibbs free energies and reference values."""
        _, predicted_gibbs = self.get_gibbs(pressures)
        error = np.array(predicted_gibbs) - np.array(reference_gibbs)
        return np.sqrt(np.mean(error**2))

    def eos_plot(self, fig_name="EV_plot.png"):
        from rsspolymlp.utils.matplot_util.custom_plt import CustomPlt
        from rsspolymlp.utils.matplot_util.make_plot import MakePlot

        custom_template = CustomPlt(
            label_size=8,
            label_pad=3.0,
            xtick_size=7,
            ytick_size=7,
            xtick_pad=3.0,
            ytick_pad=3.0,
        )
        plt = custom_template.get_custom_plt()
        plotter = MakePlot(
            plt=plt,
            column_size=0.85,
            height_ratio=0.95,
        )
        plotter.initialize_ax()
        plotter.set_visuality(n_color=4, n_line=4, n_marker=0, color_type="grad")

        volumes = np.linspace(min(self._volume_range), max(self._volume_range), 200)
        energies = self.get_energy(volumes)

        plotter.ax_plot(
            volumes,
            energies,
            plot_type="closed",
            label=None,
            plot_size=0.0,
            line_size=0.7,
            zorder=1,
        )
        plotter.ax_scatter(
            self._volumes,
            self._energies,
            plot_type="open",
            label=None,
            plot_size=0.7,
            zorder=2,
        )

        plotter.finalize_ax(
            xlabel=r"Volume ($\rm{\AA}^3$/atom)",
            ylabel="Energy (eV/atom)",
        )
        plt.tight_layout()
        plt.savefig(
            fig_name,
            bbox_inches="tight",
            pad_inches=0.01,
            dpi=400,
        )
        plt.close()

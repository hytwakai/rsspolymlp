from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class CustomPlt:
    """Template for configuring Matplotlib styles for consistent visualization.

    Attributes
    ----------
    title_size : float, optional
        Font size for the title, default is 9.0.
    title_pad : float, optional
        Padding for the title, default is 5.0.
    label_size : float, optional
        Font size for axis labels, default is 9.0.
    label_pad : float, optional
        Padding for axis labels, default is 3.0.
    legend_size : float, optional
        Font size for legend, default is 7.0.
    xtick_size : float, optional
        Font size for x-axis tick labels, default is 8.0.
    ytick_size : float, optional
        Font size for y-axis tick labels, default is 8.0.
    xtick_pad : float, optional
        Padding for x-axis ticks, default is 3.0.
    ytick_pad : float, optional
        Padding for y-axis ticks, default is 3.0.
    """

    title_size: float = 9.0
    title_pad: float = 3.0
    label_size: float = 9.0
    label_pad: float = 3.0
    legend_size: float = 6.0
    xtick_size: float = 8.0
    ytick_size: float = 8.0
    xtick_pad: float = 3.0
    ytick_pad: float = 3.0
    font_mode: str = "sans"

    def __post_init__(self):
        if self.font_mode == "serif":
            font_settings = {
                "font.family": "serif",
                "font.serif": "cmr10",
                "mathtext.fontset": "cm",
            }
        else:
            font_settings = {
                "font.family": "sans-serif",
                "font.sans-serif": "Arial",
                "mathtext.fontset": "cm",
            }
        self.customrc = {
            # Font settings
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.formatter.use_mathtext": True,
            "axes.labelsize": self.label_size,
            "axes.labelpad": self.label_pad,
            "axes.titlesize": self.title_size,
            "axes.titlepad": self.title_pad,
            "legend.fontsize": self.legend_size,
            **font_settings,
            # Spine settings
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.axisbelow": True,
            "axes.linewidth": 0.5,
            # Plot settings
            "grid.linewidth": 0.3,
            # Tick settings
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.labelsize": self.xtick_size,
            "ytick.labelsize": self.ytick_size,
            "xtick.major.pad": self.xtick_pad,
            "ytick.major.pad": self.ytick_pad,
            "xtick.major.width": 0.4,
            "ytick.major.width": 0.4,
            "xtick.minor.width": 0.3,
            "ytick.minor.width": 0.3,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
        }

    def get_custom_plt(self):
        """Apply the predefined Matplotlib style settings using Seaborn.

        Returns
        -------
        matplotlib.pyplot
            Updated Matplotlib configuration with the custom styles applied.
        """

        sns.set_context("paper")
        plt.rcParams.update(self.customrc)
        return plt

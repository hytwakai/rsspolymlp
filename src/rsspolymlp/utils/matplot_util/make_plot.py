import numpy as np
from matplotlib import gridspec
from matplotlib.ticker import LogLocator, MultipleLocator

from rsspolymlp.utils.matplot_util.custom_plt import CustomPlt


class MakePlot:
    """Class for creating and managing plots using CustomPlt."""

    def __init__(
        self,
        plt: CustomPlt,
        column_size=1.0,
        height_ratio=1.0,
        plot_grid=[(1,), (1,)],
    ):
        """Init method.

        Parameters
        ----------
        plt : CustomPlt
            plt object generating from CustomPlt.
        column_size : float, optional
            Scaling factor for the size of the figure, by default 1 column width.
        height_ratio : float, optional
            Ratio of height to width for the figure, by default 1.0.
        plot_grid : list of tuples, optional
            Grid structure for subplots, by default [(1,), (1,)].
        """

        self.plt = plt
        self.column_size = column_size
        self.height_ratio = height_ratio
        self.plot_grid = plot_grid
        self.dpi = 200

        if self.plot_grid == [(1,), (1,)]:
            self.fig, self.ax = self.plt.subplots()
        else:
            self.fig = self.plt.figure()
            self.gs = gridspec.GridSpec(
                len(plot_grid[0]),
                len(plot_grid[1]),
                height_ratios=plot_grid[0],
                width_ratios=plot_grid[1],
            )
        width = 3.375 * column_size
        height = width * height_ratio
        self.fig.set_size_inches(width, height)
        self.fig.set_dpi(self.dpi)

    def set_visuality(self, n_color=0, n_line=0, n_marker=0, color_type="class"):
        """Setting color, line, and marker styles.

        Parameters
        ----------
        n_color : int, optional
            Index for selecting color, by default blue.
        n_line : int, optional
            Index for selecting line style, by default dashed.
        n_marker : int, optional
            Index for selecting marker style, by default circle.
        color_type : str, optional
            Color scheme 'class' (classification) or 'grad' (gradation), by default 'class'.
        """

        if color_type == "class":
            color_array = [
                (5.5, 36.9, 69.4),  # blue
                (100, 60, 0.4),  # orange
                (0.4, 82.4, 0.4),  # green
                (100, 0.4, 0.4),  # red
                (49.8, 2.4, 74.1),  # purple
                (0.4, 60.8, 66.3),  # lightblue
            ]
            color_array = [tuple(np.array(c) / 100) for c in color_array]
            n_color = n_color % 6
        if color_type == "grad":
            color_array = [
                (134, 143, 155),
                (102, 113, 128),
                (71, 85, 102),
                (41, 58, 77),
                (8, 33, 53),
            ]
            color_array = [tuple(np.array(c) / 255) for c in color_array]
            n_color = n_color % 5
        line_array = [
            "--",
            "-.",
            (0, (3, 1, 1, 1, 1, 1)),  # -..
            ":",
            "-",
        ]
        marker_array = ["o", "s", "v", "^", "D", "p"]

        self.color = color_array[n_color]
        self.line = line_array[n_line % 5]
        self.marker = marker_array[n_marker % 6]

    def initialize_ax(self, index_row=None, index_column=None):
        """Initialize subplot axis if necessary."""
        if not self.plot_grid == [(1,), (1,)]:
            self.ax = self.plt.subplot(self.gs[index_row, index_column])
        else:
            pass

    def ax_plot(
        self,
        x,
        y,
        plot_type,
        label=None,
        plot_size=1.0,
        line_size=1.0,
        zorder=1,
    ):
        """Plot data on the axis.

        Parameters
        ----------
        x : array-like
            X-axis data.
        y : array-like
            Y-axis data.
        plot_type : str
            Type of plot ('open', 'closed').
        label : str, optional
            Label for legend, by default None.
        plot_size : int, optional
            Size of the markers and lines, by default 1.0.
        zorder : int, optional
            Layer order for the plot, by default 1.
        """

        self._plot_size = 4.0 * plot_size
        self._line_size = line_size
        if plot_type in ["open", "closed"]:
            self.ax.plot(
                x,
                y,
                markeredgewidth=self._plot_size * 0.15,
                markersize=self._plot_size,
                lw=self._line_size * 0.8,
                linestyle=self.line,
                marker=self.marker,
                label=label,
                zorder=zorder,
                markerfacecolor="none" if plot_type == "open" else self.color,
                c=self.color,
            )

    def ax_scatter(
        self,
        x,
        y,
        plot_type,
        label=None,
        plot_size=1.0,
        zorder=1,
    ):
        """Plot data on the axis.

        Parameters
        ----------
        x : array-like
            X-axis data.
        y : array-like
            Y-axis data.
        plot_type : str
            Type of scatter ('open', 'closed').
        label : str, optional
            Label for legend, by default None.
        plot_size : int, optional
            Size of the markers and lines, by default 1.0.
        zorder : int, optional
            Layer order for the plot, by default 1.
        """

        self._scatter_size = (4.0 * plot_size) ** 2
        if plot_type in ["open", "closed"]:
            self.ax.scatter(
                x,
                y,
                s=self._scatter_size,
                linewidths=self._scatter_size**0.5 * 0.12,
                facecolor="none" if plot_type == "open" else self.color,
                edgecolors=self.color,
                marker=self.marker,
                label=label,
                zorder=zorder,
            )
            if plot_type == "open":
                self.ax.scatter(
                    x,
                    y,
                    s=self._scatter_size * 0.06,
                    facecolor=self.color,
                    edgecolors="none",
                    marker="o",
                    zorder=zorder,
                )

    def ax_hist(self, dist, bins=50, range=None):
        if range is None:
            range = [min(dist), max(dist)]
        self.ax.hist(
            dist,
            bins=bins,
            range=range,
            color=self.color,
        )

    def finalize_ax(
        self,
        xlabel=None,
        ylabel=None,
        title=None,
        x_limits=None,
        x_limits_hide=False,
        x_grid=None,
        y_limits=None,
        y_limits_hide=False,
        y_grid=None,
        legend_ncol=None,
        legend_length=3.0,
        xlog=False,
        ylog=False,
        xlog_grid=None,
        ylog_grid=None,
        twiny=None,
    ):
        """Finalize axis settings and apply value range, grid, labels, and log scale if needed.

        Parameters
        ----------
        xlabel : str, optional
            Label for the x-axis, by default None.
        ylabel : str, optional
            Label for the y-axis, by default None.
        title : str, optional
            Title of the plot, by default None.
        xlimits : tuple, optional
            Tuple (min, max) for x-axis limits, by default None.
        xgrid : tuple, optional
            Tuple (major, minor) grid intervals for x-axis, by default None.
        y_limits : tuple, optional
            Tuple (min, max) for y-axis limits, by default None.
        y_grid : tuple, optional
            Tuple (major, minor) grid intervals for y-axis, by default None.
        legend_ncol : int, optional
            Number of columns in the legend, by default None (no legend).
        legend_length : float, optional
            Length of legend handles, by default 3.0.
        xlog : bool, optional
            Whether to use logarithmic scale for x-axis, by default False.
        ylog : bool, optional
            Whether to use logarithmic scale for y-axis, by default False.
        """
        self._set_axis_scale(xlog, ylog, xlog_grid, ylog_grid)
        self._set_axis_limits(x_limits, x_limits_hide, y_limits, y_limits_hide)
        self._set_axis_grid(x_limits, y_limits, x_grid, y_grid)
        self._set_axis_labels(xlabel, ylabel, title)
        self._set_twiny(twiny, x_limits, x_grid)
        if legend_ncol:
            self.ax.legend(frameon=False, ncol=legend_ncol, handlelength=legend_length)
        self.ax.grid(linestyle="dashed")

    def _set_axis_scale(self, xlog, ylog, xlog_grid, ylog_grid):
        """Set log scale for axes if needed."""
        if xlog:
            self.ax.set_xscale("log")
            self.ax.xaxis.set_minor_locator(
                LogLocator(numticks=13, subs=np.arange(0.1, 1, 0.1))
            )
            if xlog_grid:
                self.ax.set_xticks(xlog_grid)
        if ylog:
            self.ax.set_yscale("log")
            self.ax.yaxis.set_minor_locator(
                LogLocator(numticks=13, subs=np.arange(0.1, 1, 0.1))
            )
            if ylog_grid:
                self.ax.set_yticks(ylog_grid)

    def _set_axis_limits(self, x_limits, x_limits_hide, y_limits, y_limits_hide):
        """Set axis limits if specified."""
        if x_limits:
            self.ax.set_xlim(*x_limits)
        if x_limits_hide:
            self.ax.tick_params(labelbottom=False)
        if y_limits:
            self.ax.set_ylim(*y_limits)
        if y_limits_hide:
            self.ax.tick_params(labelleft=False)

    def _set_axis_grid(self, x_limits, y_limits, x_grid, y_grid):
        """Set grid spacing for axes."""
        if x_grid:
            if x_grid[1]:
                self.ax.xaxis.set_minor_locator(MultipleLocator(x_grid[1]))
            if x_grid[0]:
                x_values = np.round(
                    np.append(np.arange(*x_limits, x_grid[0]), x_limits[1]), 6
                )
                self.ax.set_xticks(x_values)
        if y_grid:
            if y_grid[1]:
                self.ax.yaxis.set_minor_locator(MultipleLocator(y_grid[1]))
            if y_grid[0]:
                y_values = np.round(
                    np.append(np.arange(*y_limits, y_grid[0]), y_limits[1]), 6
                )
                self.ax.set_yticks(y_values)

    def _set_axis_labels(self, xlabel, ylabel, title):
        """Set labels and title for axes."""
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        if title:
            self.ax.set_title(title)

    def _set_twiny(self, twiny, x_limits, x_grid):
        if twiny:
            ax2 = self.ax.twiny()
            if x_grid[1]:
                ax2.xaxis.set_minor_locator(MultipleLocator(x_grid[1]))
            if x_grid[0]:
                x_values = np.round(
                    np.append(np.arange(*x_limits, x_grid[0]), x_limits[1]), 6
                )
                ax2.set_xticks(x_values)
            ax2.tick_params(labeltop=False, labelbottom=False)

    @property
    def get_fig(self):
        return self.fig

    @property
    def get_ax(self):
        return self.ax

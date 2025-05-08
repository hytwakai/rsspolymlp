import numpy as np
from matplot_util.custom_plt import CustomPlt
from matplot_util.make_plot import MakePlot

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Set a custom template with modified label size, title size, and padding
custom_template = CustomPlt()

# Get plt object applied the custom style
plt = custom_template.get_custom_plt()

# Set fig and ax objects
plotter = MakePlot(
    plt=plt,
    column_size=1.0,  # 1 column width.
    height_ratio=1.0,
)

# Set color, line, and marker styles
plotter.set_visuality()

# Plot data on the axis
plotter.ax_plot(
    x,
    y,
    plot_type="open",  # Type of plot component
    plot_size=1.5,  # Size of the markers and lines
)

# Finalize axis settings and apply value range, grid, labels
plotter.finalize_ax(
    xlabel="X value",
    ylabel="Y value",
    title="Example plot (sine wave)",
)

plt.tight_layout()
plt.savefig("images/example.png", dpi=600, bbox_inches="tight", pad_inches=0.01)

plt.cla()

x_set = []
y_set = []
for i in range(4):
    x = np.linspace(0, 10, 100)
    x_set.append(x)
    y_set.append(np.sin(x - 1 * i))

custom_template = CustomPlt(
    title_size=10.0,  # Title font size
    label_size=9.0,  # Label font size
    legend_size=7.0,  # Legend font size
    xtick_size=8.0,  # Font size for x-axis tick labels
    ytick_size=8.0,  # Font size for y-axis tick labels
)
plt = custom_template.get_custom_plt()

plotter = MakePlot(plt=plt)

# plot style images
plotter.set_visuality()
plotter.ax_plot(x_set[0], y_set[0], label="Open plot", plot_type="open")
plotter.set_visuality(n_color=1, n_line=1)
plotter.ax_plot(x_set[1], y_set[1], label="Closed plot", plot_type="closed")
plotter.set_visuality(n_color=2, n_line=2)
plotter.ax_scatter(x_set[2], y_set[2], label="Open scatter", plot_type="open")
plotter.set_visuality(n_color=3, n_line=3)
plotter.ax_scatter(x_set[3], y_set[3], label="Closed scatter", plot_type="closed")

plotter.finalize_ax(
    xlabel="X value",
    ylabel="Y value",
    title="Sine Wave Plots",
    x_limits=[0, 10],  # (min, max) for x-axis limits
    x_grid=[2, 1],  # (major, minor) grid intervals for x-axis
    y_limits=[-1.5, 1.5],  # (min, max) for y-axis limits
    y_grid=[0.5, 0.25],  # (major, minor) grid intervals for y-axis
    legend_ncol=2,  # Number of columns in the legend
)

plt.tight_layout()
plt.savefig("images/single_plot.png", dpi=600, bbox_inches="tight", pad_inches=0.01)

plt.cla()

custom_template = CustomPlt()
plt = custom_template.get_custom_plt()

plotter = MakePlot(
    plt=plt,
    column_size=2.0,  # 2 column width
    height_ratio=0.7,
    plot_grid=[(1, 1), (1, 1, 1)],  # Grid structure for subplots (2*3)
)

for i in range(2):
    for g in range(3):
        # Initialize subplot axis
        plotter.initialize_ax(index_row=i, index_column=g)

        x = np.linspace(0, 10, 100)
        y = np.sin(x - 0.5 * (3 * i + g))
        plotter.set_visuality(n_color=3 * i + g, n_line=3 * i + g, n_marker=3 * i + g)
        plotter.ax_plot(x, y, plot_type="open", label="Open plot")
        plotter.finalize_ax(
            xlabel="X value",
            ylabel="Y value",
            title=f"Plot [{i+1}, {g+1}]",
            x_limits=[0, 10],
            x_grid=[2, 1],
            y_limits=[-1.5, 1.5],
            y_grid=[0.5, 0.25],
            legend_ncol=1,
        )

plotter.fig.text(0.5, -0.015, r"Superlabel X", ha="center", fontsize=11)
plotter.fig.text(-0.015, 0.5, r"Superlabel Y", va="center", rotation="vertical", fontsize=11)
plt.tight_layout()
plt.subplots_adjust(hspace=0.34, wspace=0.32)
plt.savefig("images/multiple_plot.png", dpi=600, bbox_inches="tight", pad_inches=0.01)

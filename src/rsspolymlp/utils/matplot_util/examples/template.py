from matplot_util.custom_plt import CustomPlt
from matplot_util.make_plot import MakePlot

# initiarize plt and fig objects
custom_template = CustomPlt(
    title_size=9.0,
    title_pad=3.0,
    label_size=9.0,
    label_pad=3.0,
    legend_size=7.0,
    xtick_size=8.0,
    ytick_size=8.0,
    xtick_pad=3.0,
    ytick_pad=3.0,
)
plt = custom_template.get_custom_plt()
plotter = MakePlot(
    plt=plt,
    column_size=1.0,
    height_ratio=1.0,
    plot_grid=[(1,), (1,)],
)


x = []
y = []
plotter.initialize_ax(index_row=None, index_column=None)
plotter.set_visuality(n_color=0, n_line=0, n_marker=0)
plotter.ax_plot(x, y, plot_type="open", label=None, plot_size=1.0, line_size=1.0)
plotter.ax_scatter(x, y, plot_type="open", label=None, plot_size=1.0)
plotter.finalize_ax(
    xlabel=None,
    ylabel=None,
    title=None,
    x_limits=None,
    x_grid=None,
    y_limits=None,
    y_grid=None,
    legend_ncol=None,
)


# plotter.fig.text(0.5, -0.015, r"Superlabel X", ha="center", fontsize=11)
# plotter.fig.text(-0.015, 0.5, r"Superlabel Y", va="center", rotation="vertical", fontsize=11)
plt.tight_layout()
# plt.subplots_adjust(hspace=0.34, wspace=0.32)
plt.savefig("figname.png", dpi=600, bbox_inches="tight", pad_inches=0.01)


"""
Other settings

- plotter.get_fig
- plotter.get_ax

- plotter.set_visuality()
color_type="class" or "grad"
n_color (class) : [0]orange, [1]blue, [2]green, [3]red, [4]purple, [5]lightblue
n_line : [0]-- [1]-. [2]-.. [3]... [4]-
n_marker : [0]o, [1]s, [2]v, [3]^, [4]D, [5]p

- plotter.ax_plot
- plotter.ax_scatter
zorder=1

- plotter.finalize_ax()
legend_length=3.0,
xlog=False,
ylog=False,
twiny=False,

"""

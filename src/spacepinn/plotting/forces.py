import matplotlib.pyplot as plt

from .helpers import register_plot_artifact_if_possible, set_time_axis_labels


def plot_thrust(plotter, plot_legend=True):
    plotter.fig_thrust, plotter.ax_thrust = plt.subplots(figsize=plotter.figsize)
    fig, ax = plotter.fig_thrust, plotter.ax_thrust

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.t,
            result.F_mag,
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
    legend = set_time_axis_labels(ax, "Thrust magnitude", plot_legend=plot_legend)
    plotter.style_axes(ax)
    plotter.style_legend(legend)

    fig.tight_layout()
    figure_path = plotter._build_figure_path("thrust")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_gravity(plotter, legend_mode=None):
    plotter.fig_gravity, plotter.ax_gravity = plt.subplots(figsize=plotter.figsize)
    fig, ax = plotter.fig_gravity, plotter.ax_gravity

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.t,
            result.a_mag,
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=f"{label} RFM",
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        ax.plot(
            result.t,
            result.G_mag,
            linestyle="dashed" if exp["linestyle"] == "solid" else exp["linestyle"],
            color=exp["color"],
            label=f"{label} Gravity",
            linewidth=plotter.secondary_linewidth,
            zorder=exp["zorder"],
        )
    set_time_axis_labels(ax, "Gravity / Required Force magnitude", plot_legend=False)
    legend = None
    if legend_mode == "spread":
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = max(1, min(4, len(handles)))
            legend = ax.legend(
                handles,
                labels,
                loc="lower left",
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
                mode="expand",
                ncol=ncol,
                borderaxespad=0.0,
            )
    elif legend_mode == "compact":
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=2,
                fontsize=9,
                columnspacing=1.0,
                handlelength=1.6,
                labelspacing=0.35,
                borderaxespad=0.0,
            )
    plotter.style_axes(ax)
    plotter.style_legend(legend)

    fig.tight_layout()
    figure_path = plotter._build_figure_path("gravity")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()

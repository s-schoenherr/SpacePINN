import matplotlib.pyplot as plt

from .helpers import get_quiver_data, register_plot_artifact_if_possible


def plot_orbit_traj(plotter, radii_names, plot_gravity=False, plot_thrust=True):
    plotter.fig_orb, plotter.ax_orb = plt.subplots(figsize=plotter.figsize)
    fig, ax = plotter.fig_orb, plotter.ax_orb

    for label, exp in plotter.experiments.items():
        result, color, quiver_scale, zorder = exp["result"], exp["color"], exp["quiver_scale"], exp["zorder"]
        ax.plot(
            result.r[:, 0],
            result.r[:, 1],
            linestyle=exp["linestyle"],
            color=color,
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=zorder,
        )
        if plot_thrust or plot_gravity:
            r_q, G_q, T_q = get_quiver_data(result)
            if plot_gravity:
                ax.quiver(
                    r_q[:, 0],
                    r_q[:, 1],
                    G_q[:, 0],
                    G_q[:, 1],
                    color=color,
                    scale=quiver_scale,
                    label=f"Gravity/{quiver_scale}",
                )
            if plot_thrust:
                ax.quiver(
                    r_q[:, 0],
                    r_q[:, 1],
                    T_q[:, 0],
                    T_q[:, 1],
                    color="k",
                    scale=quiver_scale,
                    label=f"Thrust/{quiver_scale}",
                )
    circle_color = "black"
    for radius, name in radii_names:
        circle = plt.Circle(
            (0, 0),
            radius,
            color=circle_color,
            fill=False,
            linestyle="dashed",
            linewidth=plotter.secondary_linewidth,
            label=name,
        )
        ax.add_patch(circle)
        circle_color = "darkgrey"
    ax.scatter(0, 0, color="green", marker="o", s=100, label="Earth")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    figure_path = plotter._build_figure_path("orbit_traj")
    fig.savefig(
        figure_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    register_plot_artifact_if_possible(figure_path)

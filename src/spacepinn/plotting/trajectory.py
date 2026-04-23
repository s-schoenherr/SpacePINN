import matplotlib.pyplot as plt

from .helpers import (
    all_z_components_smaller_than_one,
    get_gravity_sources,
    get_quiver_data,
    plot_masses_2d,
    plot_masses_3d,
    register_plot_artifact_if_possible,
)


def plot_traj_3d_projection(plotter, plot_quiver=True, plot_legend=True):
    plotter.fig_traj2d, plotter.ax_traj2d = plt.subplots(2, 2, figsize=plotter.figsize)
    fig, ax = plotter.fig_traj2d, plotter.ax_traj2d
    z_smaller_1 = True

    for label, exp in plotter.experiments.items():
        res, color, quiver_scale, zorder = exp["result"], exp["color"], exp["quiver_scale"], exp["zorder"]
        trajectory_linestyle = exp.get("trajectory_linestyle", exp["linestyle"])
        x, y, z = res.r[:, 0], res.r[:, 1], res.r[:, 2]
        ax[0, 0].plot(
            x, y, label=label, color=color, linestyle=trajectory_linestyle, linewidth=plotter.main_linewidth, zorder=zorder
        )
        ax[0, 1].plot(x, z, color=color, linestyle=trajectory_linestyle, linewidth=plotter.main_linewidth, zorder=zorder)
        ax[1, 0].plot(y, z, color=color, linestyle=trajectory_linestyle, linewidth=plotter.main_linewidth, zorder=zorder)

        ax[0, 0].scatter(*res.r0[:2], color="r", marker="o")
        ax[0, 0].scatter(*res.rN[:2], color="r", marker="x")
        ax[0, 1].scatter(res.r0[0], res.r0[2], color="r", marker="o")
        ax[0, 1].scatter(res.rN[0], res.rN[2], color="r", marker="x")
        ax[1, 0].scatter(res.r0[1], res.r0[2], color="r", marker="o")
        ax[1, 0].scatter(res.rN[1], res.rN[2], color="r", marker="x")

        if z_smaller_1 and not all_z_components_smaller_than_one(z):
            z_smaller_1 = False

        if plot_quiver:
            r_q, G_q, T_q = get_quiver_data(
                res,
                step=exp.get("quiver_step", 10),
                count=exp.get("quiver_count"),
            )
            ax[0, 0].quiver(
                r_q[:, 0],
                r_q[:, 1],
                G_q[:, 0],
                G_q[:, 1],
                color=color,
                scale=quiver_scale,
                label=rf"Gravity/{quiver_scale}",
            )
            ax[0, 0].quiver(
                r_q[:, 0],
                r_q[:, 1],
                T_q[:, 0],
                T_q[:, 1],
                color="k",
                scale=quiver_scale,
                label=rf"Thrust/{quiver_scale}",
            )
            ax[0, 1].quiver(
                r_q[:, 0],
                r_q[:, 2],
                G_q[:, 0],
                G_q[:, 2],
                color=color,
                scale=quiver_scale,
                label=f"Gravity/{quiver_scale}",
            )
            ax[0, 1].quiver(
                r_q[:, 0],
                r_q[:, 2],
                T_q[:, 0],
                T_q[:, 2],
                color="k",
                scale=quiver_scale,
                label=f"Thrust/{quiver_scale}",
            )
            ax[1, 0].quiver(
                r_q[:, 1],
                r_q[:, 2],
                G_q[:, 1],
                G_q[:, 2],
                color=color,
                scale=quiver_scale,
                label=f"Gravity/{quiver_scale}",
            )
            ax[1, 0].quiver(
                r_q[:, 1],
                r_q[:, 2],
                T_q[:, 1],
                T_q[:, 2],
                color="k",
                scale=quiver_scale,
                label=f"Thrust/{quiver_scale}",
            )

    if z_smaller_1:
        ax[0, 1].set_ylim(-1, 1)
        ax[1, 0].set_ylim(-1, 1)

    ax[0, 0].scatter(*res.r0[:2], color="r", marker="o", label=r"$r(t=0)$")
    ax[0, 0].scatter(*res.rN[:2], color="r", marker="x", label=r"$r(t=1)$")
    ax[0, 0].set_xlabel("x")
    ax[0, 0].set_ylabel("y")

    ax[0, 1].scatter(res.r0[0], res.r0[2], color="r", marker="o", label=r"$r(t=0)$")
    ax[0, 1].scatter(res.rN[0], res.rN[2], color="r", marker="x", label=r"$r(t=1)$")
    ax[0, 1].set_xlabel("x")
    ax[0, 1].set_ylabel("z")

    ax[1, 0].scatter(res.r0[1], res.r0[2], color="r", marker="o", label=r"$r(t=0)$")
    ax[1, 0].scatter(res.rN[1], res.rN[2], color="r", marker="x", label=r"$r(t=1)$")
    ax[1, 0].set_xlabel("y")
    ax[1, 0].set_ylabel("z")

    plot_masses_3d(ax, get_gravity_sources(res), projection="2d")

    if plot_legend:
        handles, labels = ax[0, 0].get_legend_handles_labels()
        ax[1, 1].legend(handles, labels, loc="center")
        ax[1, 1].set_frame_on(False)
        ax[1, 1].set_xticks([])
        ax[1, 1].set_yticks([])

    plt.tight_layout()
    plt.show()
    figure_path = plotter._build_figure_path("traj2d")
    fig.savefig(
        figure_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    register_plot_artifact_if_possible(figure_path)


def plot_traj_2d(plotter, plot_quiver=True, plot_legend=True):
    if plotter.dim == 3:
        return plot_traj_3d_projection(plotter, plot_quiver, plot_legend)

    plotter.fig_traj2d, plotter.ax_traj2d = plt.subplots(figsize=plotter.figsize)
    fig, ax = plotter.fig_traj2d, plotter.ax_traj2d

    for label, exp in plotter.experiments.items():
        result, color, linestyle, quiver_scale, zorder = (
            exp["result"],
            exp["color"],
            exp.get("trajectory_linestyle", exp["linestyle"]),
            exp["quiver_scale"],
            exp["zorder"],
        )
        ax.plot(
            result.r[:, 0],
            result.r[:, 1],
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=zorder,
        )

        if plot_quiver:
            r_q, G_q, T_q = get_quiver_data(
                result,
                step=exp.get("quiver_step", 10),
                count=exp.get("quiver_count"),
            )
            ax.quiver(
                r_q[:, 0],
                r_q[:, 1],
                G_q[:, 0],
                G_q[:, 1],
                color=color,
                scale=quiver_scale,
                label="_nolegend_",
            )

    ax.plot(
        exp["result"].r0[0],
        exp["result"].r0[1],
        "o",
        color="red",
        label=r"$r(t=0)$",
    )
    ax.plot(
        exp["result"].rN[0],
        exp["result"].rN[1],
        "x",
        color="red",
        markersize=8,
        markeredgewidth=1.8,
        label=r"$r(t=1)$",
    )
    plot_masses_2d(ax, get_gravity_sources(exp["result"]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    if plot_legend:
        ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    figure_path = plotter._build_figure_path("traj2d")
    fig.savefig(
        figure_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_traj_3d(plotter, plot_quiver=True, plot_legend=True):
    plotter.fig_3d = plt.figure(figsize=plotter.figsize)
    plotter.ax_3d = plotter.fig_3d.add_subplot(111, projection="3d")
    fig, ax = plotter.fig_3d, plotter.ax_3d
    z_smaller_1 = True

    for label, exp in plotter.experiments.items():
        res = exp["result"]
        r_q, G_q, T_q = get_quiver_data(
            res,
            step=exp.get("quiver_step", 10),
            count=exp.get("quiver_count"),
        )
        trajectory_linestyle = exp.get("trajectory_linestyle", exp["linestyle"])
        ax.plot3D(
            res.r[:, 0],
            res.r[:, 1],
            res.r[:, 2],
            label=label,
            color=exp["color"],
            linestyle=trajectory_linestyle,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        if plot_quiver:
            ax.quiver(
                r_q[:, 0],
                r_q[:, 1],
                r_q[:, 2],
                G_q[:, 0],
                G_q[:, 1],
                G_q[:, 2],
                color=exp["color"],
                label="Gravity",
            )
            ax.quiver(
                r_q[:, 0],
                r_q[:, 1],
                r_q[:, 2],
                T_q[:, 0],
                T_q[:, 1],
                T_q[:, 2],
                color="k",
                label="Thrust",
            )
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")

        if z_smaller_1 and not all_z_components_smaller_than_one(res.r[:, 2]):
            z_smaller_1 = False

        ax.scatter(*res.r0, marker="o", color="red")
        ax.scatter(*res.rN, marker="x", color="red")

    if z_smaller_1:
        ax.set_zlim(-1, 1)

    ax.scatter(*res.r0, marker="o", color="red", label=r"$r(t=0)$")
    ax.scatter(*res.rN, marker="x", color="red", label=r"$r(t=1)$")

    plot_masses_3d(ax, get_gravity_sources(res), projection="3d")
    if plot_legend:
        ax.legend(loc="upper center", ncol=3)
    figure_path = plotter._build_figure_path("traj3d")
    fig.savefig(
        figure_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    register_plot_artifact_if_possible(figure_path)
    plt.show()

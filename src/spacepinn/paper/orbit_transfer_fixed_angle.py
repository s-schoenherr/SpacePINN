from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spacepinn
from spacepinn.config.config_orbit_transfer import (
    H_HEO,
    H_LEO,
    R_HEO,
    R_LEO,
    circular_ot_kinematic_polar_config,
)
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import (
    PAPER_BASELINE_MAX_ITERATION,
    paper_baseline_solver_kwargs,
)
from spacepinn.paper._plot_style import (
    LOSS_AXES_RECT,
    LOSS_FIGSIZE,
    MAIN_AXES_RECT,
    MAIN_FIGSIZE,
    MAIN_LINEWIDTH,
    SECONDARY_LINEWIDTH,
    configure_paper_plotter,
)
from spacepinn.paper._baseline_summary import print_baseline_delta_v_summary
from spacepinn.opengoddard.circular_orbit_transfer_goddard import (
    kinematic_ot_goddard,
)
from spacepinn.plotting.helpers import get_quiver_data, register_plot_artifact_if_possible
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "hohnmann_transfer"
FIG_PREFIX = "hohnmann_transfer"
TIME_AXIS_PADDING_FRACTION = 0.02
BASELINE_LABEL = "Baseline (OpenGoddard)"
KINEMATIC_LABEL = "PINN with exact BC"
DIRECT_COLLOCATION_COLOR = PALETTE["opengoddard"]
PINN_EMPHASIS_LINEWIDTH = 4.2
ORBIT_RADII = [
    (R_LEO, f"LEO ({H_LEO:.0f} km above Earth)"),
    (R_HEO, f"MEO ({H_HEO:.0f} km above Earth)"),
]
EARTH_MARKER_SIZE = 420
PAPER_N_ADAM = 10_000
PAPER_N_LBFGS = 0
PAPER_CONVERGENCE_THRESHOLD = 1e-5
OPENGODDARD_MAX_ITERATION = PAPER_BASELINE_MAX_ITERATION
OPENGODDARD_THRUST_CAP = 5e-3
def physical_time_minutes(result) -> tuple:
    t_total_minutes = float(result.t_total) / 60.0
    t_minutes = np.asarray(result.t, dtype=float).reshape(-1) * t_total_minutes
    t_padding = TIME_AXIS_PADDING_FRACTION * t_total_minutes
    return t_minutes, t_padding


def _paper_label(label: str) -> str:
    if label == KINEMATIC_LABEL:
        return KINEMATIC_LABEL
    if label == BASELINE_LABEL:
        return BASELINE_LABEL
    return label


def build_config(*, smoke: bool | None = None) -> dict:
    config = deepcopy(circular_ot_kinematic_polar_config)
    config["label"] = KINEMATIC_LABEL
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["optimizer"]["n_adam"] = PAPER_N_ADAM
    config["optimizer"]["n_lbfgs"] = PAPER_N_LBFGS
    config["optimizer"]["convergence_threshold"] = PAPER_CONVERGENCE_THRESHOLD

    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    if smoke_enabled:
        config["optimizer"]["n_adam"] = 1
        config["optimizer"]["n_lbfgs"] = 0
    return config


def build_baseline_entry(*, smoke: bool | None = None) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    conventional_result = kinematic_ot_goddard(
        BASELINE_LABEL,
        thrust_cap=OPENGODDARD_THRUST_CAP,
        **paper_baseline_solver_kwargs(smoke_enabled=smoke_enabled),
    )
    conventional_result["color"] = DIRECT_COLLOCATION_COLOR
    conventional_result["linestyle"] = "solid"
    conventional_result["trajectory_linestyle"] = "solid"
    return {
        "label": conventional_result["label"],
        "result": conventional_result["result"],
        "model": conventional_result.get("model"),
        "config": conventional_result.get("config"),
        "plotting": {
            key: conventional_result[key]
            for key in ("linestyle", "trajectory_linestyle", "color", "quiver_scale")
            if key in conventional_result
        },
        "source": "opengoddard",
    }


def plot_loss_figure(entries: list[dict], *, output_dir: str) -> None:
    loss_entries = [entry for entry in entries if entry["label"] != BASELINE_LABEL]
    plotter = TrajectoryPlotter(
        loss_entries,
        dim=2,
        figsize=LOSS_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    configure_paper_plotter(plotter)

    fig = plt.figure(figsize=LOSS_FIGSIZE)
    ax = fig.add_axes(LOSS_AXES_RECT)
    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.loss,
            linestyle="solid",
            label=f"{label} Total Loss",
            color=PALETTE["kinematic"],
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        if result.loss_bc:
            ax.plot(
                result.loss_bc,
                linestyle="--",
                label=label + r" $\omega_{BC}$$L_{BC}$",
                color=PALETTE["kinematic"],
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )
        if result.loss_physics:
            ax.plot(
                result.loss_physics,
                linestyle="-.",
                label=label + r" $\omega_P$$L_{P}$",
                color=PALETTE["kinematic"],
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )

    visible_lengths = [len(entry["result"].loss) for entry in loss_entries]
    if visible_lengths:
        ax.set_xlim(0, max(visible_lengths))
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0)
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("loss")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_thrust_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    configure_paper_plotter(plotter)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = physical_time_minutes(result)
        linestyle = "solid" if label == BASELINE_LABEL else exp["linestyle"]
        linewidth = PINN_EMPHASIS_LINEWIDTH if label == KINEMATIC_LABEL else plotter.main_linewidth
        ax.plot(
            t_minutes,
            result.F_mag,
            linestyle=linestyle,
            color=exp["color"],
            label=label,
            linewidth=linewidth,
            zorder=exp["zorder"],
        )

    max_t_minutes = 0.0
    max_t_padding = 0.0
    for exp in plotter.experiments.values():
        _, t_padding = physical_time_minutes(exp["result"])
        max_t_minutes = max(max_t_minutes, exp["result"].t_total / 60.0)
        max_t_padding = max(max_t_padding, t_padding)
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Thrust magnitude / km s$^{-2}$")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="upper center", framealpha=0.95, facecolor="white", edgecolor="0.3")
    plotter.style_legend(legend)
    ax.set_xlim(-max_t_padding, max_t_minutes + max_t_padding)
    ax.set_yscale("log")
    figure_path = plotter._build_figure_path("thrust")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_gravity_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    configure_paper_plotter(plotter)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = physical_time_minutes(result)
        linestyle = "solid" if label == BASELINE_LABEL else exp["linestyle"]
        rfm_linewidth = PINN_EMPHASIS_LINEWIDTH if label == KINEMATIC_LABEL else plotter.main_linewidth
        gravity_linewidth = PINN_EMPHASIS_LINEWIDTH if label == KINEMATIC_LABEL else plotter.secondary_linewidth
        ax.plot(
            t_minutes,
            result.a_mag,
            linestyle=linestyle,
            color=exp["color"],
            label=f"{_paper_label(label)} RFM",
            linewidth=rfm_linewidth,
            zorder=exp["zorder"],
        )
        ax.plot(
            t_minutes,
            result.G_mag,
            linestyle="dashed" if linestyle == "solid" else linestyle,
            color=exp["color"],
            label=f"{_paper_label(label)} Gravity",
            linewidth=gravity_linewidth,
            zorder=exp["zorder"],
        )

    max_t_minutes = 0.0
    max_t_padding = 0.0
    max_force = 0.0
    for exp in plotter.experiments.values():
        _, t_padding = physical_time_minutes(exp["result"])
        max_t_minutes = max(max_t_minutes, exp["result"].t_total / 60.0)
        max_t_padding = max(max_t_padding, t_padding)
        result = exp["result"]
        max_force = max(
            max_force,
            float(np.max(np.asarray(result.a_mag, dtype=float))),
            float(np.max(np.asarray(result.G_mag, dtype=float))),
        )
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Gravity / Required Force magnitude / km s$^{-2}$")
    ax.set_xlim(-max_t_padding, max_t_minutes + max_t_padding)
    ax.set_yscale("log")
    ax.set_ylim(top=max_force * 8.0 if max_force > 0 else None)
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        columnspacing=1.0,
        handlelength=1.6,
        labelspacing=0.28,
        borderaxespad=0.35,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.3",
    )
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("gravity")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_orbit_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    configure_paper_plotter(plotter)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result, color, quiver_scale, zorder = exp["result"], exp["color"], exp["quiver_scale"], exp["zorder"]
        linestyle = "solid" if label == BASELINE_LABEL else exp["linestyle"]
        linewidth = PINN_EMPHASIS_LINEWIDTH if label == KINEMATIC_LABEL else plotter.main_linewidth
        ax.plot(
            result.r[:, 0],
            result.r[:, 1],
            linestyle=linestyle,
            color=color,
            label=label,
            linewidth=linewidth,
            zorder=zorder,
        )
        r_q, _, T_q = get_quiver_data(result)
        ax.quiver(
            r_q[:, 0],
            r_q[:, 1],
            T_q[:, 0],
            T_q[:, 1],
            color="k",
            scale=quiver_scale,
            label="_nolegend_",
        )

    reference_result = entries[0]["result"]
    ax.plot(
        reference_result.r0[0],
        reference_result.r0[1],
        "o",
        color="red",
        markersize=6,
        label=r"$\mathbf{r}(t_0)=(R_{\mathrm{LEO}},0)$",
    )
    ax.plot(
        reference_result.rN[0],
        reference_result.rN[1],
        "x",
        color="red",
        markersize=8,
        markeredgewidth=1.8,
        label=r"$\mathbf{r}(T)=(R_{\mathrm{MEO}},\pi)$",
    )

    circle_color = "black"
    for radius, name in ORBIT_RADII:
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
    ax.scatter(0, 0, color="green", marker="o", s=EARTH_MARKER_SIZE, label="_nolegend_", zorder=5)
    ax.annotate(
        "Earth",
        (0, 0),
        xytext=(10, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=plotter.legend_fontsize + 2.0,
        color="black",
        zorder=6,
    )
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="lower right", framealpha=0.95, facecolor="white", edgecolor="0.3")
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("orbit_traj")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def main(*, skip_plots: bool = False, print_summary: bool = True, smoke: bool | None = None):
    collection_run = run_experiment_collection(
        configs=[build_config(smoke=smoke)],
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
        additional_entries=[
            capture_baseline_entry(
                lambda: build_baseline_entry(smoke=smoke),
                log_filename="baseline_opengoddard.log",
            ),
        ],
    )

    if print_summary:
        print_collection_run_summary(collection_run)
        print_baseline_delta_v_summary(
            collection_run["entries"],
            title=COLLECTION_LABEL,
            baseline_labels=(BASELINE_LABEL,),
        )

    if not skip_plots:
        plotter = TrajectoryPlotter(
            collection_run["entries"],
            dim=2,
            figsize=MAIN_FIGSIZE,
            fig_prefix=FIG_PREFIX,
            output_dir=collection_run["plot_output_dir"],
        )
        plotter.main_linewidth = MAIN_LINEWIDTH
        plotter.secondary_linewidth = SECONDARY_LINEWIDTH
        plotter.plot_traj_2d()

        plot_thrust_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
        plot_gravity_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
        plot_orbit_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
        plot_loss_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
    return collection_run


if __name__ == "__main__":
    main()

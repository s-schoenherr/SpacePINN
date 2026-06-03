from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spacepinn.config.config_orbit_transfer import H_GEO, H_HEO, H_LEO, R_GEO, R_HEO, R_LEO
from spacepinn.paper.style import (
    LOSS_AXES_RECT,
    LOSS_FIGSIZE,
    MAIN_AXES_RECT,
    MAIN_FIGSIZE,
    MAIN_LINEWIDTH,
    SECONDARY_LINEWIDTH,
    configure_paper_plotter,
)
from spacepinn.plotting.helpers import get_quiver_data, register_plot_artifact_if_possible
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter


FIXED_FIG_PREFIX = "fixed_terminal_angle"
FREE_FIG_PREFIX = "free_terminal_angle"
TIME_AXIS_PADDING_FRACTION = 0.02
FIXED_PINN_LABEL = "PINN with exact BC"
BASELINE_LABEL = "Baseline (OpenGoddard)"
PINN_EMPHASIS_LINEWIDTH = 4.2
FREE_MAIN_LINEWIDTH = 2.8
FREE_SECONDARY_LINEWIDTH = 2.4


def _physical_time_minutes(result) -> tuple[np.ndarray, float]:
    t_total_minutes = float(result.t_total) / 60.0
    t_minutes = np.asarray(result.t, dtype=float).reshape(-1) * t_total_minutes
    return t_minutes, TIME_AXIS_PADDING_FRACTION * t_total_minutes


def _fixed_paper_label(label: str) -> str:
    if label == FIXED_PINN_LABEL:
        return FIXED_PINN_LABEL
    if label == BASELINE_LABEL:
        return BASELINE_LABEL
    return label


def _target_orbit_label(target_orbit: str) -> str:
    if target_orbit == "heo":
        return f"MEO ({H_HEO:.0f} km above Earth)"
    return f"GEO ({H_GEO:.0f} km above Earth)"


def _target_radius_symbol(target_orbit: str) -> str:
    if target_orbit == "heo":
        return r"$R_{\mathrm{MEO}}$"
    return r"$R_{\mathrm{GEO}}$"


def _as_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _default_terminal_angle_guess(target_orbit: str) -> float:
    return float(np.pi * (2.10 if target_orbit == "geo" else 1.0))


def _initial_terminal_angle_guess_point(entry: dict, *, target_orbit: str) -> np.ndarray:
    target_radius = R_HEO if target_orbit == "heo" else R_GEO
    config = entry.get("config") or {}
    optimizer_config = config.get("optimizer") or {}
    initial_rN = optimizer_config.get("rN")
    if initial_rN is not None:
        initial_rN_values = np.asarray(
            initial_rN.detach().cpu().numpy() if hasattr(initial_rN, "detach") else initial_rN,
            dtype=float,
        ).reshape(-1)
        if len(initial_rN_values) >= 2:
            target_radius = float(initial_rN_values[0])
            alpha_guess = float(initial_rN_values[1])
            return np.array([target_radius * np.cos(alpha_guess), target_radius * np.sin(alpha_guess)], dtype=float)

    extra_parameters = config.get("extra_parameters") or {}
    alpha_parameter = extra_parameters.get("alpha_N")
    alpha_guess = _as_float(alpha_parameter) if alpha_parameter is not None else _default_terminal_angle_guess(target_orbit)
    return np.array([target_radius * np.cos(alpha_guess), target_radius * np.sin(alpha_guess)], dtype=float)


def _time_bounds(plotter: TrajectoryPlotter) -> tuple[float, float]:
    max_t_minutes = 0.0
    max_t_padding = 0.0
    for exp in plotter.experiments.values():
        _, t_padding = _physical_time_minutes(exp["result"])
        max_t_minutes = max(max_t_minutes, float(exp["result"].t_total) / 60.0)
        max_t_padding = max(max_t_padding, t_padding)
    return max_t_minutes, max_t_padding


def _plot_fixed_loss(entries: list[dict], *, output_dir: str) -> None:
    loss_entries = [entry for entry in entries if entry["label"] != BASELINE_LABEL]
    plotter = TrajectoryPlotter(loss_entries, dim=2, figsize=LOSS_FIGSIZE, fig_prefix=FIXED_FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)

    fig = plt.figure(figsize=LOSS_FIGSIZE)
    ax = fig.add_axes(LOSS_AXES_RECT)
    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(result.loss, linestyle="solid", label=f"{label} Total Loss", color=PALETTE["kinematic"], linewidth=plotter.main_linewidth, zorder=exp["zorder"])
        if result.loss_bc:
            ax.plot(result.loss_bc, linestyle="--", label=label + r" $\omega_{BC}$$L_{BC}$", color=PALETTE["kinematic"], linewidth=plotter.secondary_linewidth, zorder=exp["zorder"])
        if result.loss_physics:
            ax.plot(result.loss_physics, linestyle="-.", label=label + r" $\omega_P$$L_{P}$", color=PALETTE["kinematic"], linewidth=plotter.secondary_linewidth, zorder=exp["zorder"])

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


def _plot_fixed_thrust(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIXED_FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = _physical_time_minutes(result)
        linestyle = "solid" if label == BASELINE_LABEL else exp["linestyle"]
        linewidth = PINN_EMPHASIS_LINEWIDTH if label == FIXED_PINN_LABEL else plotter.main_linewidth
        ax.plot(t_minutes, result.F_mag, linestyle=linestyle, color=exp["color"], label=label, linewidth=linewidth, zorder=exp["zorder"])

    max_t_minutes, max_t_padding = _time_bounds(plotter)
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


def _plot_fixed_gravity(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIXED_FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = _physical_time_minutes(result)
        linestyle = "solid" if label == BASELINE_LABEL else exp["linestyle"]
        rfm_linewidth = PINN_EMPHASIS_LINEWIDTH if label == FIXED_PINN_LABEL else plotter.main_linewidth
        gravity_linewidth = PINN_EMPHASIS_LINEWIDTH if label == FIXED_PINN_LABEL else plotter.secondary_linewidth
        ax.plot(t_minutes, result.a_mag, linestyle=linestyle, color=exp["color"], label=f"{_fixed_paper_label(label)} RFM", linewidth=rfm_linewidth, zorder=exp["zorder"])
        ax.plot(t_minutes, result.G_mag, linestyle="dashed" if linestyle == "solid" else linestyle, color=exp["color"], label=f"{_fixed_paper_label(label)} Gravity", linewidth=gravity_linewidth, zorder=exp["zorder"])

    max_t_minutes, max_t_padding = _time_bounds(plotter)
    max_force = max(
        max(
            float(np.max(np.asarray(exp["result"].a_mag, dtype=float))),
            float(np.max(np.asarray(exp["result"].G_mag, dtype=float))),
        )
        for exp in plotter.experiments.values()
    )
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Gravity / Required Force magnitude / km s$^{-2}$")
    ax.set_xlim(-max_t_padding, max_t_minutes + max_t_padding)
    ax.set_yscale("log")
    ax.set_ylim(top=max_force * 8.0 if max_force > 0 else None)
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="upper left", ncol=1, columnspacing=1.0, handlelength=1.6, labelspacing=0.28, borderaxespad=0.35, framealpha=0.95, facecolor="white", edgecolor="0.3")
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("gravity")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def _plot_fixed_orbit(entries: list[dict], *, output_dir: str, plot_quiver: bool = False) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIXED_FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result, color, quiver_scale, zorder = exp["result"], exp["color"], exp["quiver_scale"], exp["zorder"]
        linestyle = "solid" if label == BASELINE_LABEL else exp["linestyle"]
        linewidth = PINN_EMPHASIS_LINEWIDTH if label == FIXED_PINN_LABEL else plotter.main_linewidth
        ax.plot(result.r[:, 0], result.r[:, 1], linestyle=linestyle, color=color, label=label, linewidth=linewidth, zorder=zorder)
        if plot_quiver:
            r_q, _, T_q = get_quiver_data(result)
            ax.quiver(r_q[:, 0], r_q[:, 1], T_q[:, 0], T_q[:, 1], color="k", scale=quiver_scale, label="_nolegend_")

    reference_result = entries[0]["result"]
    ax.plot(reference_result.r0[0], reference_result.r0[1], "o", color="red", markersize=6, label=r"$\mathbf{r}(t_0)=(R_{\mathrm{LEO}},0)$")
    ax.plot(reference_result.rN[0], reference_result.rN[1], "x", color="red", markersize=8, markeredgewidth=1.8, label=r"$\mathbf{r}(T)=(R_{\mathrm{MEO}},\pi)$")

    circle_color = "black"
    for radius, name in [(R_LEO, f"LEO ({H_LEO:.0f} km above Earth)"), (R_HEO, f"MEO ({H_HEO:.0f} km above Earth)")]:
        circle = plt.Circle((0, 0), radius, color=circle_color, fill=False, linestyle="dashed", linewidth=plotter.secondary_linewidth, label=name)
        ax.add_patch(circle)
        circle_color = "darkgrey"
    ax.scatter(0, 0, color="green", marker="o", s=420, label="_nolegend_", zorder=5)
    ax.annotate("Earth", (0, 0), xytext=(10, 8), textcoords="offset points", ha="left", va="bottom", fontsize=plotter.legend_fontsize + 2.0, color="black", zorder=6)
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


def plot_fixed_orbit_transfer(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIXED_FIG_PREFIX, output_dir=output_dir)
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = SECONDARY_LINEWIDTH
    plotter.plot_traj_2d(plot_quiver=False)
    _plot_fixed_thrust(entries, output_dir=output_dir)
    _plot_fixed_gravity(entries, output_dir=output_dir)
    _plot_fixed_orbit(entries, output_dir=output_dir, plot_quiver=False)
    _plot_fixed_loss(entries, output_dir=output_dir)


def _plot_free_thrust(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FREE_FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    plotter.main_linewidth = FREE_MAIN_LINEWIDTH
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = _physical_time_minutes(result)
        ax.plot(t_minutes, np.maximum(result.F_mag, 1e-12), linestyle=exp["linestyle"], color=exp["color"], label=label, linewidth=plotter.main_linewidth, zorder=exp["zorder"])

    max_t_minutes, max_t_padding = _time_bounds(plotter)
    ax.set_yscale("log")
    ax.set_xlim(-max_t_padding, max_t_minutes + max_t_padding)
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Thrust magnitude / km s$^{-2}$")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="upper center", framealpha=0.98, facecolor="white", edgecolor="0.3")
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("thrust")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)


def _plot_free_gravity(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FREE_FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    plotter.main_linewidth = FREE_MAIN_LINEWIDTH
    plotter.secondary_linewidth = FREE_SECONDARY_LINEWIDTH
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = _physical_time_minutes(result)
        ax.plot(t_minutes, np.maximum(result.a_mag, 1e-12), linestyle=exp["linestyle"], color=exp["color"], label=f"{label} RFM", linewidth=plotter.main_linewidth, zorder=exp["zorder"])
        ax.plot(t_minutes, np.maximum(result.G_mag, 1e-12), linestyle="dashed" if exp["linestyle"] == "solid" else exp["linestyle"], color=exp["color"], label=f"{label} Gravity", linewidth=plotter.secondary_linewidth, zorder=exp["zorder"])

    max_t_minutes, max_t_padding = _time_bounds(plotter)
    max_force = max(
        max(
            float(np.max(np.asarray(exp["result"].a_mag, dtype=float))),
            float(np.max(np.asarray(exp["result"].G_mag, dtype=float))),
        )
        for exp in plotter.experiments.values()
    )
    ax.set_yscale("log")
    ax.set_xlim(-max_t_padding, max_t_minutes + max_t_padding)
    ax.set_ylim(top=max_force * 8.0 if max_force > 0 else None)
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Gravity / Required Force magnitude / km s$^{-2}$")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="upper left", ncol=1, columnspacing=1.0, handlelength=1.6, labelspacing=0.28, borderaxespad=0.35, framealpha=0.98, facecolor="white", edgecolor="0.3")
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("gravity")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)


def _plot_free_orbit(entries: list[dict], *, output_dir: str, target_orbit: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FREE_FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    plotter.main_linewidth = FREE_MAIN_LINEWIDTH
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(result.r[:, 0], result.r[:, 1], linestyle=exp["linestyle"], color=exp["color"], label=label, linewidth=plotter.main_linewidth, zorder=exp["zorder"])

    reference_entry = entries[0]
    reference_result = reference_entry["result"]
    ax.plot(reference_result.r0[0], reference_result.r0[1], "o", color="red", markersize=6, label=r"$\mathbf{r}(t_0)=(R_{\mathrm{LEO}},0)$")
    terminal_point = _initial_terminal_angle_guess_point(reference_entry, target_orbit=target_orbit)
    ax.plot(terminal_point[0], terminal_point[1], "x", color="red", markersize=8, markeredgewidth=1.8, label=rf"$\mathbf{{r}}(T)$ = ({_target_radius_symbol(target_orbit)}, initial angle guess)")

    for idx, (radius, name) in enumerate(((R_LEO, f"LEO ({H_LEO:.0f} km above Earth)"), (R_HEO if target_orbit == "heo" else R_GEO, _target_orbit_label(target_orbit)))):
        ax.add_patch(plt.Circle((0, 0), radius, color="black" if idx == 0 else "darkgrey", fill=False, linestyle="dashed", linewidth=plotter.secondary_linewidth, label=name))

    ax.scatter(0, 0, color="green", marker="o", s=100, label="_nolegend_", zorder=5)
    ax.annotate("Earth", (0, 0), xytext=(0, -7), textcoords="offset points", ha="center", va="top", fontsize=plotter.legend_fontsize + 2.0, color="black", zorder=6)
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="upper left", framealpha=0.98, facecolor="white", edgecolor="0.3")
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("orbit_traj")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)


def plot_free_orbit_transfer(entries: list[dict], *, output_dir: str, target_orbit: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FREE_FIG_PREFIX, output_dir=output_dir)
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = SECONDARY_LINEWIDTH
    plotter.plot_loss()
    _plot_free_thrust(entries, output_dir=output_dir)
    _plot_free_gravity(entries, output_dir=output_dir)
    _plot_free_orbit(entries, output_dir=output_dir, target_orbit=target_orbit)

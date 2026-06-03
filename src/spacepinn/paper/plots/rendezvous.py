from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from spacepinn.config.config_orbit_transfer import R_EARTH
from spacepinn.paper.style import (
    LOSS_AXES_RECT,
    LOSS_FIGSIZE as PAPER_LOSS_FIGSIZE,
    MAIN_AXES_RECT,
    MAIN_FIGSIZE as PAPER_MAIN_FIGSIZE,
    configure_paper_plotter,
)
from spacepinn.problems.rendezvous_hold_point_eci import (
    TARGET_RADIUS_KM,
    TARGET_SPEED_KM_S,
    target_state_eci,
)
from spacepinn.plotting.helpers import register_plot_artifact_if_possible, set_time_axis_labels
from spacepinn.plotting.paper_style import PAPER_STYLE
from spacepinn.plotter import TrajectoryPlotter


FIG_PREFIX = "rendezvous"
PINN_LABEL = "PINN with exact BC"
BASELINE_LABEL = "OpenGoddard"
WARMSTART_BASELINE_LABEL = "OpenGoddard (PINN initial guess)"
BASELINE_PAPER_LABEL = "Baseline (OpenGoddard)"
WARMSTART_BASELINE_PAPER_LABEL = "Baseline (OpenGoddard, PINN initial guess)"
PINN_COLOR = "#2ca02c"
BASELINE_COLOR = "#4d4d4d"
WARMSTART_BASELINE_COLOR = "#1f77b4"
TARGET_COLOR = "#4d4d4d"
HOLD_POINT_COLOR = "#d62728"
MAIN_FIGSIZE = PAPER_MAIN_FIGSIZE
LOSS_FIGSIZE = PAPER_LOSS_FIGSIZE


def _paper_axes(
    *,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
    axes_rect: tuple[float, float, float, float] = MAIN_AXES_RECT,
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(axes_rect)
    return fig, ax


def _style_paper_axes(ax) -> None:
    ax.xaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    ax.yaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=PAPER_STYLE.tick_label_fontsize)
    ax.title.set_size(PAPER_STYLE.title_fontsize)


def _style_paper_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(PAPER_STYLE.legend_fontsize)
    frame = legend.get_frame()
    if frame is not None:
        frame.set_alpha(PAPER_STYLE.legend_framealpha)
        frame.set_edgecolor("black")
        frame.set_linewidth(1.0)
        frame.set_facecolor("white")


def _save_paper_figure(fig, figure_path: Path) -> None:
    fig.savefig(
        figure_path,
        bbox_inches=PAPER_STYLE.save_bbox_inches,
        pad_inches=PAPER_STYLE.save_pad_inches,
    )


def _time_seconds_from_result(result) -> np.ndarray:
    time_values = np.asarray(result.t, dtype=float).reshape(-1)
    if time_values.size == 0:
        return np.asarray([], dtype=float)
    if float(np.max(time_values)) <= 1.0001:
        return time_values * float(result.t_total)
    return time_values


def _target_history(*, t_seconds: np.ndarray, scenario: dict | None = None) -> np.ndarray:
    t_seconds = np.asarray(t_seconds, dtype=float).reshape(-1)
    target = (scenario or {}).get("target", {})
    radius_km = float(target.get("radius_km", TARGET_RADIUS_KM))
    speed_km_s = float(target.get("speed_km_s", TARGET_SPEED_KM_S))
    return np.asarray(
        [
            target_state_eci(t_seconds=float(t), radius_km=radius_km, speed_km_s=speed_km_s)["position"]
            for t in t_seconds
        ],
        dtype=float,
    )


def _relative_history_lvlh(*, target_history: np.ndarray, chaser_history: np.ndarray) -> np.ndarray:
    radial_unit = target_history / np.linalg.norm(target_history, axis=1, keepdims=True)
    along_track_unit = np.column_stack((-radial_unit[:, 1], radial_unit[:, 0]))
    relative_eci = np.asarray(chaser_history, dtype=float) - np.asarray(target_history, dtype=float)
    radial_component = np.sum(relative_eci * radial_unit, axis=1)
    along_component = np.sum(relative_eci * along_track_unit, axis=1)
    return np.column_stack((radial_component, along_component))


def _select_reference_entry(entries: list[dict]) -> dict:
    for entry in entries:
        if entry.get("label") == PINN_LABEL or entry.get("source") == "pinn":
            return entry
    return entries[0]


def _descriptive_plot_label(entry: dict) -> str:
    label = str(entry.get("label", "Method"))
    if label.startswith(PINN_LABEL):
        return PINN_LABEL
    if label == BASELINE_LABEL:
        return BASELINE_PAPER_LABEL
    if label == WARMSTART_BASELINE_LABEL:
        return WARMSTART_BASELINE_PAPER_LABEL
    return label


def _entry_visual_style(entry: dict) -> dict[str, object]:
    label = str(entry.get("label", ""))
    if label == PINN_LABEL or entry.get("source") == "pinn":
        return {
            "color": PINN_COLOR,
            "linestyle": "solid",
            "linewidth": 2.1,
            "zorder": 4,
            "alpha": 0.62,
            "marker": None,
            "markevery": None,
        }
    if label == BASELINE_LABEL:
        return {
            "color": BASELINE_COLOR,
            "linestyle": "--",
            "linewidth": 3.2,
            "zorder": 2,
            "alpha": 1.0,
            "marker": None,
            "markevery": None,
        }
    if label == WARMSTART_BASELINE_LABEL:
        return {
            "color": WARMSTART_BASELINE_COLOR,
            "linestyle": "-.",
            "linewidth": 3.2,
            "zorder": 3,
            "alpha": 1.0,
            "marker": None,
            "markevery": None,
        }
    return {
        "color": entry.get("color", PINN_COLOR),
        "linestyle": entry.get("linestyle", "solid"),
        "linewidth": 2.2,
        "zorder": int(entry.get("zorder", 2)),
        "alpha": 1.0,
        "marker": None,
        "markevery": None,
    }


def plot_orbit_overview_figure(entries: list[dict], *, output_dir: str, scenario: dict) -> None:
    if not entries:
        return

    fig, ax = _paper_axes()

    reference_entry = _select_reference_entry(entries)
    reference_result = reference_entry["result"]
    reference_target_history = _target_history(t_seconds=_time_seconds_from_result(reference_result), scenario=scenario)
    start_target = scenario["target"]["start"]["position"]
    end_target = scenario["target"]["end"]["position"]
    target_theta = np.unwrap(np.arctan2(reference_target_history[:, 1], reference_target_history[:, 0]))
    theta_margin = max(0.12 * float(target_theta[-1] - target_theta[0]), 0.08)
    leo_theta = np.linspace(float(target_theta[0] - theta_margin), float(target_theta[-1] + theta_margin), num=300)
    leo_arc = np.column_stack(
        (
            float(scenario["target"]["radius_km"]) * np.cos(leo_theta),
            float(scenario["target"]["radius_km"]) * np.sin(leo_theta),
        )
    )

    target_altitude_km = float(scenario["target"]["radius_km"]) - R_EARTH
    ax.plot(
        leo_arc[:, 0],
        leo_arc[:, 1],
        color=TARGET_COLOR,
        linestyle="--",
        linewidth=2.0,
        alpha=0.75,
        label=f"LEO ({target_altitude_km:.0f} km above Earth)",
        zorder=1,
    )

    all_positions = [reference_target_history, leo_arc]

    def _sort_key(entry: dict) -> tuple[int, str]:
        label = str(entry.get("label", ""))
        is_primary_pinn = label == PINN_LABEL
        is_any_pinn = entry.get("source") == "pinn"
        return (2 if is_primary_pinn else 1 if is_any_pinn else 0, label)

    for entry in sorted(entries, key=_sort_key):
        result = entry["result"]
        trajectory = np.asarray(result.r, dtype=float)
        visual = _entry_visual_style(entry)
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=visual["linewidth"],
            alpha=visual["alpha"],
            marker=visual["marker"],
            markevery=visual["markevery"],
            markersize=5.0 if visual["marker"] is not None else None,
            label=_descriptive_plot_label(entry),
            zorder=visual["zorder"],
        )
        all_positions.append(trajectory)

    ax.scatter(start_target[0], start_target[1], color=TARGET_COLOR, marker="s", s=40, label="Target at $t_0$", zorder=4)
    ax.scatter(end_target[0], end_target[1], color=TARGET_COLOR, marker="^", s=42, label="Target at $t_N$", zorder=4)

    stacked = np.vstack(all_positions)
    x_min, x_max = float(np.min(stacked[:, 0])), float(np.max(stacked[:, 0]))
    y_min, y_max = float(np.min(stacked[:, 1])), float(np.max(stacked[:, 1]))
    span = max(x_max - x_min, y_max - y_min, 5.0)
    pad = 0.08 * span
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * span + pad

    x_shift = 0.18 * span
    ax.set_xlim(cx - half - x_shift, cx + half - x_shift)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    _style_paper_axes(ax)
    legend = ax.legend(
        loc="lower left",
        ncol=1,
        framealpha=0.95,
        columnspacing=1.2,
        handlelength=2.4,
    )
    _style_paper_legend(legend)

    figure_path = Path(output_dir) / f"{FIG_PREFIX}_overview_orbit.pdf"
    _save_paper_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_lvlh_figures(entries: list[dict], *, output_dir: str, scenario: dict) -> None:
    if not entries:
        return

    relative_start = scenario["chaser"]["initial_relative_offset_km"]
    relative_hold_point = scenario["chaser"]["final_hold_point_offset_km"]
    fig, ax = _paper_axes()
    x_all = [relative_start[0], relative_hold_point[0], 0.0]
    y_all = [relative_start[1], relative_hold_point[1], 0.0]

    for entry in entries:
        result = entry["result"]
        visual = _entry_visual_style(entry)
        target_history = _target_history(t_seconds=_time_seconds_from_result(result), scenario=scenario)
        relative_history = _relative_history_lvlh(target_history=target_history, chaser_history=np.asarray(result.r, dtype=float))

        ax.plot(
            relative_history[:, 0],
            relative_history[:, 1],
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=max(2.6, float(visual["linewidth"])),
            label=_descriptive_plot_label(entry),
        )
        ax.scatter(relative_history[-1, 0], relative_history[-1, 1], color=visual["color"], marker="^", s=50)
        x_all.extend(relative_history[:, 0].tolist())
        y_all.extend(relative_history[:, 1].tolist())

    ax.scatter(relative_start[0], relative_start[1], color="red", marker="o", s=55, label="Initial relative state")
    ax.scatter(relative_hold_point[0], relative_hold_point[1], color="#d62728", marker="x", s=80, label="Desired hold point")
    ax.scatter(0.0, 0.0, color=TARGET_COLOR, marker="s", s=55, label="Target")

    x_values = np.asarray(x_all, dtype=float)
    y_values = np.asarray(y_all, dtype=float)
    x_span = max(float(np.max(x_values) - np.min(x_values)), 0.2)
    y_span = max(float(np.max(y_values) - np.min(y_values)), 0.2)
    padding = 0.15 * max(x_span, y_span)

    ax.set_xlabel("Radial relative offset / km")
    ax.set_ylabel("Along-track relative offset / km")
    ax.set_xlim(float(np.min(x_values) - padding), 0.1)
    ax.set_aspect("auto")
    ax.set_box_aspect(1)
    ax.set_ylim(bottom=-0.5, top=0.2)
    _style_paper_axes(ax)
    legend = ax.legend(
        loc="lower left",
        ncol=1,
        framealpha=0.95,
        columnspacing=1.0,
        handlelength=1.8,
        labelspacing=0.18,
        borderpad=0.28,
    )
    _style_paper_legend(legend)

    figure_path = Path(output_dir) / f"{FIG_PREFIX}_lvlh.pdf"
    _save_paper_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_separation_figure(entries: list[dict], *, output_dir: str, scenario: dict) -> None:
    if not entries:
        return

    fig, ax = _paper_axes(figsize=LOSS_FIGSIZE, axes_rect=LOSS_AXES_RECT)
    for entry in entries:
        result = entry["result"]
        target_history = _target_history(t_seconds=_time_seconds_from_result(result), scenario=scenario)
        separation_km = np.linalg.norm(np.asarray(result.r, dtype=float) - target_history, axis=1)
        time_seconds = _time_seconds_from_result(result)
        ax.plot(
            time_seconds,
            separation_km,
            color=entry.get("color", PINN_COLOR),
            linewidth=2.2,
            label=_descriptive_plot_label(entry),
        )
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Target separation / km")
    current_ymin, current_ymax = ax.get_ylim()
    ax.set_ylim(min(current_ymin, -0.34), current_ymax)
    ax.set_box_aspect(1)
    _style_paper_axes(ax)
    legend = ax.legend(loc="lower left")
    _style_paper_legend(legend)

    figure_path = Path(output_dir) / f"{FIG_PREFIX}_separation.pdf"
    _save_paper_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_thrust_figure(entries: list[dict], *, output_dir: str) -> None:
    if not entries:
        return

    fig, ax = _paper_axes()
    for entry in entries:
        result = entry["result"]
        visual = _entry_visual_style(entry)
        ax.plot(
            np.asarray(result.t, dtype=float).reshape(-1),
            np.clip(np.asarray(result.F_mag, dtype=float).reshape(-1), 1e-16, None),
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=visual["linewidth"],
            alpha=visual["alpha"],
            label=_descriptive_plot_label(entry),
            zorder=visual["zorder"],
        )
    ax.set_xlabel("Normalized time")
    ax.set_ylabel(r"Thrust magnitude / km s$^{-2}$")
    legend = ax.legend()
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    _style_paper_axes(ax)
    _style_paper_legend(legend)
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_thrust.pdf"
    _save_paper_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_gravity_figure(entries: list[dict], *, output_dir: str) -> None:
    if not entries:
        return

    fig, ax = _paper_axes()
    max_force = 0.0
    for entry in entries:
        result = entry["result"]
        visual = _entry_visual_style(entry)
        time_values = np.asarray(result.t, dtype=float).reshape(-1)
        rfm_values = np.clip(np.asarray(result.a_mag, dtype=float).reshape(-1), 1e-16, None)
        gravity_values = np.clip(np.asarray(result.G_mag, dtype=float).reshape(-1), 1e-16, None)
        max_force = max(max_force, float(np.max(rfm_values)), float(np.max(gravity_values)))
        ax.plot(
            time_values,
            rfm_values,
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=visual["linewidth"],
            alpha=visual["alpha"],
            label=f"{_descriptive_plot_label(entry)} RFM",
            zorder=visual["zorder"],
        )
        gravity_linestyle = ":" if visual["linestyle"] == "solid" else (0, (6, 2, 1.5, 2))
        ax.plot(
            time_values,
            gravity_values,
            color=visual["color"],
            linestyle=gravity_linestyle,
            linewidth=max(1.8, visual["linewidth"] - 0.4),
            alpha=min(1.0, visual["alpha"] + 0.12),
            label=f"{_descriptive_plot_label(entry)} Gravity",
            zorder=visual["zorder"] + 0.1,
        )

    set_time_axis_labels(ax, "Gravity / Required Force magnitude", plot_legend=False)
    ax.set_yscale("log")
    if max_force > 0:
        ax.set_ylim(top=max(1e-1, max_force * 8.0))
    ax.set_box_aspect(1)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(
            handles,
            labels,
            loc="upper left",
            ncol=1,
            framealpha=0.95,
            columnspacing=1.0,
            handlelength=1.8,
            labelspacing=0.28,
            borderaxespad=0.35,
        )
        _style_paper_legend(legend)

    _style_paper_axes(ax)
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_gravity.pdf"
    _save_paper_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_loss_figure(entries: list[dict], *, output_dir: str) -> None:
    pinn_entries = [entry for entry in entries if entry.get("source") == "pinn" and getattr(entry.get("result"), "loss", None)]
    if not pinn_entries:
        return
    plotter = TrajectoryPlotter(pinn_entries, dim=2, figsize=LOSS_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    fig, ax = _paper_axes(figsize=LOSS_FIGSIZE, axes_rect=LOSS_AXES_RECT)
    visible_lengths: list[int] = []

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        if not getattr(result, "loss", None):
            continue
        descriptive_label = _descriptive_plot_label({"label": label})
        visible_lengths.append(len(result.loss))
        ax.plot(result.loss, linestyle="solid", label=f"{descriptive_label} Total Loss", color=exp["color"], linewidth=plotter.main_linewidth, zorder=exp["zorder"])
        if getattr(result, "loss_bc", None):
            ax.plot(result.loss_bc, linestyle="--", label=descriptive_label + r" $\lambda_{BC}$$L_{BC}$", color=exp["color"], linewidth=plotter.secondary_linewidth, zorder=exp["zorder"])
        if getattr(result, "loss_physics", None):
            ax.plot(result.loss_physics, linestyle="-.", label=descriptive_label + r" $\lambda_{P}$$L_{P}$", color=exp["color"], linewidth=plotter.secondary_linewidth, zorder=exp["zorder"])

    if visible_lengths:
        ax.set_xlim(0, max(visible_lengths))
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    _style_paper_axes(ax)
    legend = ax.legend(loc="best")
    _style_paper_legend(legend)
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_loss.pdf"
    _save_paper_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_results(entries: list[dict], *, output_dir: str, scenario: dict) -> None:
    plot_loss_figure(entries, output_dir=output_dir)
    plot_thrust_figure(entries, output_dir=output_dir)
    plot_gravity_figure(entries, output_dir=output_dir)
    plot_orbit_overview_figure(entries, output_dir=output_dir, scenario=scenario)
    plot_lvlh_figures(entries, output_dir=output_dir, scenario=scenario)
    plot_separation_figure(entries, output_dir=output_dir, scenario=scenario)

from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import spacepinn
import torch

from spacepinn.config.config_orbit_transfer import R_EARTH, circular_ot_kinematic_polar_config
from spacepinn.config.transform_functions import kinematic_rendezvous_hold_point_eci_polar_fn
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._rendezvous_hold_point_eci_shared import (
    DEFAULT_T_FINAL_SECONDS,
    TARGET_RADIUS_KM,
    TARGET_SPEED_KM_S,
    build_scenario,
    target_state_eci,
)
from spacepinn.opengoddard.rendezvous_hold_point_eci_goddard import (
    kinematic_rendezvous_hold_point_eci_goddard,
)
from spacepinn.paper._aggregate_summary import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_summary import print_baseline_delta_v_summary
from spacepinn.paper._mc_mode import (
    add_single_mc_arguments,
    label_with_seed,
    plot_single_group_boxplots,
    representative_entries,
    resolve_mode,
    seed_sequence,
    single_group_key,
)
from spacepinn.paper._plot_style import (
    LOSS_AXES_RECT,
    LOSS_FIGSIZE as PAPER_LOSS_FIGSIZE,
    MAIN_AXES_RECT,
    MAIN_FIGSIZE as PAPER_MAIN_FIGSIZE,
    configure_paper_plotter,
)
from spacepinn.plotting.paper_style import PAPER_STYLE
from spacepinn.plotting.helpers import register_plot_artifact_if_possible, set_time_axis_labels
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import execute_single_experiment, print_collection_run_summary
from spacepinn.runner.context import RunCollectionContext

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "rendezvous_hold_point_eci"
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
EARTH_COLOR = "#1f77b4"
MAIN_FIGSIZE = PAPER_MAIN_FIGSIZE
LOSS_FIGSIZE = PAPER_LOSS_FIGSIZE
PAPER_N_ADAM = 100_000
PAPER_N_LBFGS = 0
PAPER_CONVERGENCE_THRESHOLD = 1e-7
BASELINE_MAX_ITERATION = 10
BASELINE_FTOL = 1e-11
BASELINE_SLSQP_MAXITER = 25
REPRESENTATIVE_SEED = 9058
MC_SEED_START = 9000
MC_NUM_SEEDS = 100


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


def _parse_args():
    parser = argparse.ArgumentParser(description="ECI rendezvous-to-hold-point experiment.")
    add_single_mc_arguments(parser, default_mode="single")
    parser.add_argument("--t-final-seconds", type=float, default=DEFAULT_T_FINAL_SECONDS)
    parser.add_argument("--n-adam", type=int, default=PAPER_N_ADAM)
    parser.add_argument("--n-lbfgs", type=int, default=PAPER_N_LBFGS)
    parser.add_argument("--convergence-threshold", type=float, default=PAPER_CONVERGENCE_THRESHOLD)
    parser.add_argument("--baseline-max-iteration", type=int, default=BASELINE_MAX_ITERATION)
    parser.add_argument("--baseline-ftol", type=float, default=BASELINE_FTOL)
    parser.add_argument("--baseline-slsqp-maxiter", type=int, default=BASELINE_SLSQP_MAXITER)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def build_config(
    *,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    n_adam: int = PAPER_N_ADAM,
    n_lbfgs: int = PAPER_N_LBFGS,
    convergence_threshold: float = PAPER_CONVERGENCE_THRESHOLD,
    seed: int | None = None,
    label_seed: bool = False,
    smoke: bool | None = None,
) -> dict:
    scenario = build_scenario(t_final_seconds=t_final_seconds)
    config = deepcopy(circular_ot_kinematic_polar_config)

    if seed is not None:
        config["seed"] = int(seed)
    config["label"] = label_with_seed(PINN_LABEL, seed) if label_seed and seed is not None else PINN_LABEL
    trainable_t_total = torch.nn.Parameter(torch.tensor(float(t_final_seconds), dtype=torch.float32), requires_grad=True)
    config["extra_parameters"] = {"t_total": trainable_t_total}
    config["pinn"]["output_transform_fn"] = partial(
        kinematic_rendezvous_hold_point_eci_polar_fn,
        x0=torch.tensor(scenario["chaser"]["start_position_polar"], dtype=torch.float32),
        v0=torch.tensor(scenario["chaser"]["start_velocity_polar"], dtype=torch.float32),
        target_radius=float(scenario["target"]["radius_km"]),
        target_speed=float(scenario["target"]["speed_km_s"]),
        hold_point_radial_offset=float(scenario["chaser"]["final_hold_point_offset_km"][0]),
    )
    config["optimizer"]["coordinate_system"] = "polar"
    config["optimizer"]["t_total"] = trainable_t_total
    config["optimizer"]["r0"] = torch.tensor(scenario["chaser"]["start_position_polar"], dtype=torch.float32)
    config["optimizer"]["rN"] = torch.tensor(scenario["chaser"]["end_position_polar"], dtype=torch.float32)
    config["optimizer"]["t_colloc"] = torch.linspace(0, 1, 200).view(-1, 1).requires_grad_(True)
    config["optimizer"]["n_adam"] = int(n_adam)
    config["optimizer"]["n_lbfgs"] = int(n_lbfgs)
    config["optimizer"]["convergence_threshold"] = float(convergence_threshold)
    config["plotting"]["color"] = PINN_COLOR
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["scenario"] = scenario

    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    if smoke_enabled:
        config["optimizer"]["n_adam"] = 1
        config["optimizer"]["n_lbfgs"] = 0
    return config


def build_baseline_entry(
    *,
    label: str,
    t_final_seconds: float,
    warm_start_result=None,
    max_iteration: int = BASELINE_MAX_ITERATION,
    ftol: float = BASELINE_FTOL,
    slsqp_maxiter: int = BASELINE_SLSQP_MAXITER,
) -> dict:
    result = kinematic_rendezvous_hold_point_eci_goddard(
        label=label,
        warm_start_result=warm_start_result,
        max_iteration=max_iteration,
        ftol=ftol,
        slsqp_maxiter=slsqp_maxiter,
        time_final_guess=float(getattr(warm_start_result, "t_total", t_final_seconds)),
    )
    color = WARMSTART_BASELINE_COLOR if warm_start_result is not None else BASELINE_COLOR
    return {
        "label": result["label"],
        "result": result["result"],
        "model": result.get("model"),
        "config": result.get("config"),
        "plotting": {
            "color": color,
            "linestyle": "solid",
            "trajectory_linestyle": "solid",
            "zorder": 2,
        },
        "source": "opengoddard",
    }


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


def _sync_dynamic_terminal_reference(result, *, scenario: dict) -> None:
    target_end = target_state_eci(t_seconds=float(result.t_total))
    hold_offset = float(scenario["chaser"]["final_hold_point_offset_km"][0])
    hold_position = target_end["position"] + hold_offset * target_end["radial_unit"]
    target_velocity = target_end["velocity"]

    result.rN = np.asarray(hold_position, dtype=float)
    if hasattr(result, "dynamics"):
        result.dynamics.rN = np.asarray(hold_position, dtype=float)

    result.dynamic_terminal_reference = {
        "target_position_km": np.asarray(target_end["position"], dtype=float),
        "target_velocity_km_s": np.asarray(target_velocity, dtype=float),
        "hold_point_position_km": np.asarray(hold_position, dtype=float),
        "hold_point_radial_offset_km": hold_offset,
    }


def _select_reference_entry(entries: list[dict]) -> dict:
    for entry in entries:
        if entry.get("label") == PINN_LABEL or entry.get("source") == "pinn":
            return entry
    return entries[0]


def _entries_for_lvlh_view(entries: list[dict]) -> list[dict]:
    return entries


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


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "entry"


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
        descriptive_label = _descriptive_plot_label(entry)
        visual = _entry_visual_style(entry)
        color = visual["color"]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=color,
            linestyle=visual["linestyle"],
            linewidth=visual["linewidth"],
            alpha=visual["alpha"],
            marker=visual["marker"],
            markevery=visual["markevery"],
            markersize=5.0 if visual["marker"] is not None else None,
            label=descriptive_label,
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
        color = visual["color"]
        target_history = _target_history(t_seconds=_time_seconds_from_result(result), scenario=scenario)
        relative_history = _relative_history_lvlh(target_history=target_history, chaser_history=np.asarray(result.r, dtype=float))

        ax.plot(
            relative_history[:, 0],
            relative_history[:, 1],
            color=color,
            linestyle=visual["linestyle"],
            linewidth=max(2.6, float(visual["linewidth"])),
            label=_descriptive_plot_label(entry),
        )
        ax.scatter(relative_history[-1, 0], relative_history[-1, 1], color=color, marker="^", s=50)
        x_all.extend(relative_history[:, 0].tolist())
        y_all.extend(relative_history[:, 1].tolist())

    ax.scatter(relative_start[0], relative_start[1], color="red", marker="o", s=55, label="Initial relative state")
    ax.scatter(relative_hold_point[0], relative_hold_point[1], color=HOLD_POINT_COLOR, marker="x", s=80, label="Desired hold point")
    ax.scatter(0.0, 0.0, color=TARGET_COLOR, marker="s", s=55, label="Target")

    x_values = np.asarray(x_all, dtype=float)
    y_values = np.asarray(y_all, dtype=float)
    x_span = max(float(np.max(x_values) - np.min(x_values)), 0.2)
    y_span = max(float(np.max(y_values) - np.min(y_values)), 0.2)
    padding = 0.15 * max(x_span, y_span)

    ax.set_xlabel("Radial relative offset / km")
    ax.set_ylabel("Along-track relative offset / km")
    ax.set_xlim(float(np.min(x_values) - padding), 0.1)
    y_lower = float(np.min(y_values) - padding)
    y_upper = float(np.max(y_values) + padding)
    y_legend_pad = 0.64 * (y_upper - y_lower)
    #ax.set_ylim(y_lower - y_legend_pad, y_upper)
    ax.set_aspect("auto")
    ax.set_box_aspect(1)
    ax.set_ylim(bottom=-.5, top=0.2)
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
        ax.plot(
            result.loss,
            linestyle="solid",
            label=f"{descriptive_label} Total Loss",
            color=exp["color"],
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        if getattr(result, "loss_bc", None):
            ax.plot(
                result.loss_bc,
                linestyle="--",
                label=descriptive_label + r" $\lambda_{BC}$$L_{BC}$",
                color=exp["color"],
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )
        if getattr(result, "loss_physics", None):
            ax.plot(
                result.loss_physics,
                linestyle="-.",
                label=descriptive_label + r" $\lambda_{P}$$L_{P}$",
                color=exp["color"],
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )

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


def monte_carlo_group_key(entry: dict) -> str | None:
    return single_group_key(entry, base_label=PINN_LABEL)


def _append_context_entry(
    collection_context: RunCollectionContext,
    collection_results: list[dict],
    *,
    entry: dict,
    config: dict | None = None,
    model=None,
    source: str,
) -> None:
    collection_context.add_entry(
        label=entry["label"],
        result=entry["result"],
        config=config if config is not None else entry.get("config"),
        model=model if model is not None else entry.get("model"),
        source=source,
        log_text=entry.get("log_text"),
        log_filename=entry.get("log_filename"),
    )
    collection_results.append(
        {
            "label": entry["label"],
            "source": source,
            "result": entry["result"],
            **entry.get("plotting", {}),
            "config": config if config is not None else entry.get("config"),
            "model": model if model is not None else entry.get("model"),
        }
    )


def _select_representative_result(entries: list[dict], *, representative_seed: int | None):
    if representative_seed is not None:
        suffix = f"seed={int(representative_seed)}"
        for entry in entries:
            if entry.get("source") == "pinn" and suffix in str(entry.get("label", "")):
                return entry["result"]
    pinn_entries = [entry for entry in entries if entry.get("source") == "pinn"]
    if not pinn_entries:
        return None
    return min(pinn_entries, key=lambda entry: float(entry["result"].delta_v))["result"]


def run_collection(
    *,
    mode: str = "single",
    representative_seed: int = REPRESENTATIVE_SEED,
    seed_start: int = MC_SEED_START,
    num_seeds: int = MC_NUM_SEEDS,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    n_adam: int = PAPER_N_ADAM,
    n_lbfgs: int = PAPER_N_LBFGS,
    convergence_threshold: float = PAPER_CONVERGENCE_THRESHOLD,
    baseline_max_iteration: int = BASELINE_MAX_ITERATION,
    baseline_ftol: float = BASELINE_FTOL,
    baseline_slsqp_maxiter: int = BASELINE_SLSQP_MAXITER,
    smoke: bool | None = None,
):
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    effective_baseline_max_iteration = 1 if smoke_enabled else int(baseline_max_iteration)
    seeds = (
        seed_sequence(start=seed_start, count=num_seeds, smoke=smoke_enabled)
        if mode == "mc"
        else [int(representative_seed)]
    )
    configs = [
        build_config(
            t_final_seconds=t_final_seconds,
            n_adam=n_adam,
            n_lbfgs=n_lbfgs,
            convergence_threshold=convergence_threshold,
            seed=seed,
            label_seed=mode == "mc",
            smoke=smoke_enabled,
        )
        for seed in seeds
    ]
    scenario = configs[0]["scenario"]

    collection_label = f"{COLLECTION_LABEL}_monte_carlo" if mode == "mc" else COLLECTION_LABEL
    collection_context = RunCollectionContext(label=collection_label, run_root=str(RUN_ROOT))
    collection_context.start()
    collection_results: list[dict] = []

    try:
        for config in configs:
            pinn_model, pinn_result = execute_single_experiment(config)
            _sync_dynamic_terminal_reference(pinn_result, scenario=config["scenario"])
            _append_context_entry(
                collection_context,
                collection_results,
                entry={
                    "label": config["label"],
                    "result": pinn_result,
                    "plotting": config.get("plotting", {}),
                },
                config=config,
                model=pinn_model,
                source="pinn",
            )

        cold_entry = capture_baseline_entry(
            lambda: build_baseline_entry(
                label=BASELINE_LABEL,
                t_final_seconds=t_final_seconds,
                max_iteration=effective_baseline_max_iteration,
                ftol=baseline_ftol,
                slsqp_maxiter=baseline_slsqp_maxiter,
            ),
            log_filename="baseline_opengoddard.log",
        )
        _sync_dynamic_terminal_reference(cold_entry["result"], scenario=scenario)
        _append_context_entry(
            collection_context,
            collection_results,
            entry=cold_entry,
            source="opengoddard",
        )

        warm_start_result = _select_representative_result(
            collection_results,
            representative_seed=representative_seed if mode == "mc" else None,
        )
        warm_entry = capture_baseline_entry(
            lambda: build_baseline_entry(
                label=WARMSTART_BASELINE_LABEL,
                t_final_seconds=t_final_seconds,
                warm_start_result=warm_start_result,
                max_iteration=effective_baseline_max_iteration,
                ftol=baseline_ftol,
                slsqp_maxiter=baseline_slsqp_maxiter,
            ),
            log_filename="baseline_opengoddard_pinn_warmstart.log",
        )
        _sync_dynamic_terminal_reference(warm_entry["result"], scenario=scenario)
        _append_context_entry(
            collection_context,
            collection_results,
            entry=warm_entry,
            source="opengoddard",
        )

        collection_context.finalize_success()
        return {
            "label": collection_label,
            "entries": collection_results,
            "run_id": collection_context.run_id,
            "run_dir": str(collection_context.run_dir),
            "plot_output_dir": str(collection_context.plot_dir),
            "summary_path": str(collection_context.summary_path),
            "manifest_path": str(collection_context.manifest_path),
            "config_path": str(collection_context.config_path),
            "scenario": scenario,
        }
    except Exception as error:
        collection_context.finalize_failure(error)
        raise


def main(
    *,
    mode: str = "single",
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    n_adam: int = PAPER_N_ADAM,
    n_lbfgs: int = PAPER_N_LBFGS,
    convergence_threshold: float = PAPER_CONVERGENCE_THRESHOLD,
    baseline_max_iteration: int = BASELINE_MAX_ITERATION,
    baseline_ftol: float = BASELINE_FTOL,
    baseline_slsqp_maxiter: int = BASELINE_SLSQP_MAXITER,
    skip_plots: bool = False,
    print_summary: bool = True,
    smoke: bool | None = None,
    representative_seed: int | None = None,
    seed_start: int | None = None,
    num_seeds: int | None = None,
):
    representative_seed = REPRESENTATIVE_SEED if representative_seed is None else int(representative_seed)
    collection_run = run_collection(
        mode=mode,
        representative_seed=representative_seed,
        seed_start=MC_SEED_START if seed_start is None else int(seed_start),
        num_seeds=MC_NUM_SEEDS if num_seeds is None else int(num_seeds),
        t_final_seconds=t_final_seconds,
        n_adam=n_adam,
        n_lbfgs=n_lbfgs,
        convergence_threshold=convergence_threshold,
        baseline_max_iteration=baseline_max_iteration,
        baseline_ftol=baseline_ftol,
        baseline_slsqp_maxiter=baseline_slsqp_maxiter,
        smoke=smoke,
    )
    if mode == "mc":
        persist_paper_monte_carlo_aggregate_summary(
            collection_run,
            title=collection_run["label"],
            baseline_labels=(BASELINE_LABEL, WARMSTART_BASELINE_LABEL),
            group_key=monte_carlo_group_key,
        )
    if print_summary:
        print_collection_run_summary(collection_run)
        if mode == "mc":
            grouped = {PINN_LABEL: [entry for entry in collection_run["entries"] if entry.get("source") == "pinn"]}
            print_monte_carlo_summary(grouped, title=collection_run["label"])
        print_baseline_delta_v_summary(
            collection_run["entries"],
            title="Rendezvous Hold Point ECI",
            baseline_labels=(BASELINE_LABEL, WARMSTART_BASELINE_LABEL),
            group_key=monte_carlo_group_key if mode == "mc" else None,
            include_variance=mode == "mc",
        )
    if not skip_plots:
        plot_entries = representative_entries(
            collection_run["entries"],
            representative_seed=representative_seed,
            base_label=PINN_LABEL,
        )
        plot_results(
            plot_entries,
            output_dir=collection_run["plot_output_dir"],
            scenario=collection_run["scenario"],
        )
        if mode == "mc":
            plot_single_group_boxplots(
                collection_run["entries"],
                output_dir=collection_run["plot_output_dir"],
                fig_prefix=FIG_PREFIX,
                base_label=PINN_LABEL,
                baseline_labels=(BASELINE_LABEL, WARMSTART_BASELINE_LABEL),
            )
    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode=resolve_mode(args),
        t_final_seconds=args.t_final_seconds,
        n_adam=args.n_adam,
        n_lbfgs=args.n_lbfgs,
        convergence_threshold=args.convergence_threshold,
        baseline_max_iteration=args.baseline_max_iteration,
        baseline_ftol=args.baseline_ftol,
        baseline_slsqp_maxiter=args.baseline_slsqp_maxiter,
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        representative_seed=args.representative_seed,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
    )

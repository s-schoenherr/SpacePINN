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

from spacepinn.config.config_orbit_transfer import circular_ot_kinematic_polar_config
from spacepinn.config.transform_functions import (
    kinematic_rendezvous_hold_point_eci_polar_alpha_guard_fn,
    kinematic_rendezvous_hold_point_eci_polar_fn,
)
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._rendezvous_hold_point_eci_shared import (
    DEFAULT_T_FINAL_SECONDS,
    TARGET_RADIUS_KM,
    build_scenario,
    target_state_eci,
)
from spacepinn.opengoddard.rendezvous_hold_point_eci_goddard import (
    kinematic_rendezvous_hold_point_eci_goddard,
)
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_summary import print_baseline_delta_v_summary
from spacepinn.plotting.helpers import register_plot_artifact_if_possible, set_time_axis_labels
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import execute_single_experiment, print_collection_run_summary
from spacepinn.runner.context import RunCollectionContext

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "rendezvous_hold_point_eci"
FIG_PREFIX = "rendezvous"
PINN_LABEL = "PINN with exact BC"
PINN_GUARDED_LABEL = "PINN with exact BC (alpha <= alpha_target)"
BASELINE_LABEL = "OpenGoddard"
WARMSTART_BASELINE_LABEL = "OpenGoddard (PINN initial guess)"
PINN_COLOR = "#2ca02c"
PINN_GUARDED_COLOR = "#bc7c00"
BASELINE_COLOR = "#4d4d4d"
WARMSTART_BASELINE_COLOR = "#1f77b4"
TARGET_COLOR = "#4d4d4d"
HOLD_POINT_COLOR = "#d62728"
EARTH_COLOR = "#1f77b4"
MAIN_FIGSIZE = (7.4, 6.8)
LOSS_FIGSIZE = MAIN_FIGSIZE
PAPER_N_ADAM = 100_000
PAPER_N_LBFGS = 0
PAPER_CONVERGENCE_THRESHOLD = 1e-7
BASELINE_MAX_ITERATION = 10
BASELINE_FTOL = 1e-11
BASELINE_SLSQP_MAXITER = 25


def _parse_args():
    parser = argparse.ArgumentParser(description="ECI rendezvous-to-hold-point experiment.")
    parser.add_argument("--t-final-seconds", type=float, default=DEFAULT_T_FINAL_SECONDS)
    parser.add_argument("--n-adam", type=int, default=PAPER_N_ADAM)
    parser.add_argument("--n-lbfgs", type=int, default=PAPER_N_LBFGS)
    parser.add_argument("--convergence-threshold", type=float, default=PAPER_CONVERGENCE_THRESHOLD)
    parser.add_argument("--baseline-max-iteration", type=int, default=BASELINE_MAX_ITERATION)
    parser.add_argument("--baseline-ftol", type=float, default=BASELINE_FTOL)
    parser.add_argument("--baseline-slsqp-maxiter", type=int, default=BASELINE_SLSQP_MAXITER)
    parser.add_argument("--include-alpha-guard", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def build_config(
    *,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    n_adam: int = PAPER_N_ADAM,
    n_lbfgs: int = PAPER_N_LBFGS,
    convergence_threshold: float = PAPER_CONVERGENCE_THRESHOLD,
    enforce_alpha_guard: bool = False,
    smoke: bool | None = None,
) -> dict:
    scenario = build_scenario(t_final_seconds=t_final_seconds)
    config = deepcopy(circular_ot_kinematic_polar_config)

    config["label"] = PINN_GUARDED_LABEL if enforce_alpha_guard else PINN_LABEL
    trainable_t_total = torch.nn.Parameter(torch.tensor(float(t_final_seconds), dtype=torch.float32), requires_grad=True)
    config["extra_parameters"] = {"t_total": trainable_t_total}
    config["pinn"]["output_transform_fn"] = partial(
        (
            kinematic_rendezvous_hold_point_eci_polar_alpha_guard_fn
            if enforce_alpha_guard
            else kinematic_rendezvous_hold_point_eci_polar_fn
        ),
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
    config["plotting"]["color"] = PINN_GUARDED_COLOR if enforce_alpha_guard else PINN_COLOR
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


def _target_history(*, t_seconds: np.ndarray) -> np.ndarray:
    t_seconds = np.asarray(t_seconds, dtype=float).reshape(-1)
    return np.asarray([target_state_eci(t_seconds=float(t))["position"] for t in t_seconds], dtype=float)


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
    return str(entry.get("label", "Method"))


def _entry_visual_style(entry: dict) -> dict[str, object]:
    label = str(entry.get("label", ""))
    if label == PINN_GUARDED_LABEL:
        return {
            "color": PINN_GUARDED_COLOR,
            "linestyle": "-",
            "linewidth": 2.3,
            "zorder": 3,
            "alpha": 0.9,
            "marker": None,
            "markevery": None,
        }
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


def _gravity_style(entry: dict) -> tuple[object, float]:
    label = str(entry.get("label", ""))
    if label == PINN_GUARDED_LABEL:
        return (0, (1.2, 1.2)), 2.0
    if label == PINN_LABEL or entry.get("source") == "pinn":
        return (0, (1.2, 1.2)), 1.9
    if label == BASELINE_LABEL:
        return (0, (7.0, 2.0, 1.5, 2.0)), 2.4
    if label == WARMSTART_BASELINE_LABEL:
        return (0, (4.5, 1.8)), 2.2
    return ":", 2.0


def plot_orbit_overview_figure(entries: list[dict], *, output_dir: str, scenario: dict) -> None:
    if not entries:
        return

    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    reference_entry = _select_reference_entry(entries)
    reference_result = reference_entry["result"]
    reference_target_history = _target_history(t_seconds=_time_seconds_from_result(reference_result))
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

    ax.plot(
        leo_arc[:, 0],
        leo_arc[:, 1],
        color=TARGET_COLOR,
        linestyle="--",
        linewidth=2.0,
        alpha=0.75,
        label="LEO",
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

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.grid(alpha=0.25)
    ax.legend(
        loc="lower left",
        ncol=1,
        framealpha=0.95,
        columnspacing=1.2,
        handlelength=2.4,
    )

    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_overview_orbit.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_lvlh_figures(entries: list[dict], *, output_dir: str, scenario: dict) -> None:
    if not entries:
        return

    relative_start = scenario["chaser"]["initial_relative_offset_km"]
    relative_hold_point = scenario["chaser"]["final_hold_point_offset_km"]
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    x_all = [relative_start[0], relative_hold_point[0], 0.0]
    y_all = [relative_start[1], relative_hold_point[1], 0.0]

    for entry in entries:
        result = entry["result"]
        visual = _entry_visual_style(entry)
        color = visual["color"]
        target_history = _target_history(t_seconds=_time_seconds_from_result(result))
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

    ax.scatter(relative_start[0], relative_start[1], color=PINN_COLOR, marker="o", s=55, label="Initial relative state")
    ax.scatter(relative_hold_point[0], relative_hold_point[1], color=HOLD_POINT_COLOR, marker="x", s=80, label="Desired hold point")
    ax.scatter(0.0, 0.0, color=TARGET_COLOR, marker="s", s=55, label="Target")

    x_values = np.asarray(x_all, dtype=float)
    y_values = np.asarray(y_all, dtype=float)
    x_span = max(float(np.max(x_values) - np.min(x_values)), 0.2)
    y_span = max(float(np.max(y_values) - np.min(y_values)), 0.2)
    padding = 0.15 * max(x_span, y_span)

    ax.set_xlabel("Radial offset / km")
    ax.set_ylabel("Along-track offset / km")
    ax.grid(alpha=0.25)
    ax.set_xlim(float(np.min(x_values) - padding), float(np.max(x_values) + padding))
    ax.set_ylim(float(np.min(y_values) - padding), float(np.max(y_values) + padding))
    ax.set_aspect("auto")
    ax.legend(
        loc="lower right",
        framealpha=0.95,
        columnspacing=1.0,
        handlelength=2.4,
    )

    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_lvlh.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_separation_figure(entries: list[dict], *, output_dir: str) -> None:
    if not entries:
        return

    fig, ax = plt.subplots(figsize=LOSS_FIGSIZE)
    for entry in entries:
        result = entry["result"]
        visual = _entry_visual_style(entry)
        target_history = _target_history(t_seconds=_time_seconds_from_result(result))
        separation_km = np.linalg.norm(np.asarray(result.r, dtype=float) - target_history, axis=1)
        time_seconds = _time_seconds_from_result(result)
        ax.plot(
            time_seconds,
            separation_km,
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=visual["linewidth"],
            alpha=visual["alpha"],
            label=entry["label"],
            zorder=visual["zorder"],
        )
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Target separation / km")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{FIG_PREFIX}_separation.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_thrust_figure(entries: list[dict], *, output_dir: str) -> None:
    if not entries:
        return

    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    for entry in entries:
        result = entry["result"]
        visual = _entry_visual_style(entry)
        is_primary_pinn = str(entry.get("label")) == PINN_LABEL
        linewidth = visual["linewidth"] + 0.55 if is_primary_pinn else visual["linewidth"]
        alpha = 0.82 if is_primary_pinn else visual["alpha"]
        ax.plot(
            _time_seconds_from_result(result),
            np.clip(np.asarray(result.F_mag, dtype=float).reshape(-1), 1e-16, None),
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=linewidth,
            alpha=alpha,
            label=entry["label"],
            zorder=visual["zorder"],
        )
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Thrust magnitude")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    ax.set_yscale("log")
    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_thrust.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_gravity_figure(entries: list[dict], *, output_dir: str) -> None:
    if not entries:
        return

    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    for entry in entries:
        result = entry["result"]
        visual = _entry_visual_style(entry)
        time_values = np.asarray(result.t, dtype=float).reshape(-1)
        ax.plot(
            time_values,
            np.clip(np.asarray(result.a_mag, dtype=float).reshape(-1), 1e-16, None),
            color=visual["color"],
            linestyle=visual["linestyle"],
            linewidth=visual["linewidth"],
            alpha=visual["alpha"],
            label=f"{entry['label']} RFM",
            zorder=visual["zorder"],
        )
        gravity_linestyle, gravity_linewidth = _gravity_style(entry)
        ax.plot(
            time_values,
            np.clip(np.asarray(result.G_mag, dtype=float).reshape(-1), 1e-16, None),
            color=visual["color"],
            linestyle=gravity_linestyle,
            linewidth=gravity_linewidth,
            alpha=1.0,
            label=f"{entry['label']} Gravity",
            zorder=visual["zorder"] + 0.25,
        )

    set_time_axis_labels(ax, "Gravity / Required Force magnitude", plot_legend=False)
    ax.set_yscale("log")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="lower left",
            ncol=2,
            framealpha=0.95,
            columnspacing=1.0,
            handlelength=1.8,
            fontsize=9,
        )

    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_gravity.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def plot_loss_figure(entries: list[dict], *, output_dir: str) -> None:
    pinn_entries = [entry for entry in entries if entry.get("source") == "pinn" and getattr(entry.get("result"), "loss", None)]
    if not pinn_entries:
        return
    plotter = TrajectoryPlotter(pinn_entries, dim=2, figsize=LOSS_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    plotter.plot_loss()


def plot_results(entries: list[dict], *, output_dir: str, scenario: dict) -> None:
    plot_loss_figure(entries, output_dir=output_dir)
    plot_thrust_figure(entries, output_dir=output_dir)
    plot_gravity_figure(entries, output_dir=output_dir)
    plot_orbit_overview_figure(entries, output_dir=output_dir, scenario=scenario)
    plot_lvlh_figures(entries, output_dir=output_dir, scenario=scenario)
    plot_separation_figure(entries, output_dir=output_dir)


def run_collection(
    *,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    n_adam: int = PAPER_N_ADAM,
    n_lbfgs: int = PAPER_N_LBFGS,
    convergence_threshold: float = PAPER_CONVERGENCE_THRESHOLD,
    baseline_max_iteration: int = BASELINE_MAX_ITERATION,
    baseline_ftol: float = BASELINE_FTOL,
    baseline_slsqp_maxiter: int = BASELINE_SLSQP_MAXITER,
    include_alpha_guard: bool = False,
    smoke: bool | None = None,
):
    config = build_config(
        t_final_seconds=t_final_seconds,
        n_adam=n_adam,
        n_lbfgs=n_lbfgs,
        convergence_threshold=convergence_threshold,
        enforce_alpha_guard=False,
        smoke=smoke,
    )
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    effective_baseline_max_iteration = 1 if smoke_enabled else int(baseline_max_iteration)

    collection_context = RunCollectionContext(label=COLLECTION_LABEL, run_root=str(RUN_ROOT))
    collection_context.start()
    collection_results: list[dict] = []

    try:
        pinn_model, pinn_result = execute_single_experiment(config)
        _sync_dynamic_terminal_reference(pinn_result, scenario=config["scenario"])
        collection_context.add_entry(
            label=config["label"],
            result=pinn_result,
            config=config,
            model=pinn_model,
            source="pinn",
        )
        collection_results.append(
            {
                "label": config["label"],
                "source": "pinn",
                "result": pinn_result,
                **config.get("plotting", {}),
                "model": pinn_model,
            }
        )

        if include_alpha_guard:
            guarded_config = build_config(
                t_final_seconds=t_final_seconds,
                n_adam=n_adam,
                n_lbfgs=n_lbfgs,
                convergence_threshold=convergence_threshold,
                enforce_alpha_guard=True,
                smoke=smoke,
            )
            guarded_model, guarded_result = execute_single_experiment(guarded_config)
            _sync_dynamic_terminal_reference(guarded_result, scenario=guarded_config["scenario"])
            collection_context.add_entry(
                label=guarded_config["label"],
                result=guarded_result,
                config=guarded_config,
                model=guarded_model,
                source="pinn",
            )
            collection_results.append(
                {
                    "label": guarded_config["label"],
                    "source": "pinn",
                    "result": guarded_result,
                    **guarded_config.get("plotting", {}),
                    "model": guarded_model,
                }
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
        _sync_dynamic_terminal_reference(cold_entry["result"], scenario=config["scenario"])
        collection_context.add_entry(
            label=cold_entry["label"],
            result=cold_entry["result"],
            config=cold_entry.get("config"),
            model=cold_entry.get("model"),
            source="opengoddard",
            log_text=cold_entry.get("log_text"),
            log_filename=cold_entry.get("log_filename"),
        )
        collection_results.append(
            {
                "label": cold_entry["label"],
                "source": "opengoddard",
                "result": cold_entry["result"],
                **cold_entry.get("plotting", {}),
                "model": cold_entry.get("model"),
            }
        )

        warm_entry = capture_baseline_entry(
            lambda: build_baseline_entry(
                label=WARMSTART_BASELINE_LABEL,
                t_final_seconds=t_final_seconds,
                warm_start_result=pinn_result,
                max_iteration=effective_baseline_max_iteration,
                ftol=baseline_ftol,
                slsqp_maxiter=baseline_slsqp_maxiter,
            ),
            log_filename="baseline_opengoddard_pinn_warmstart.log",
        )
        _sync_dynamic_terminal_reference(warm_entry["result"], scenario=config["scenario"])
        collection_context.add_entry(
            label=warm_entry["label"],
            result=warm_entry["result"],
            config=warm_entry.get("config"),
            model=warm_entry.get("model"),
            source="opengoddard",
            log_text=warm_entry.get("log_text"),
            log_filename=warm_entry.get("log_filename"),
        )
        collection_results.append(
            {
                "label": warm_entry["label"],
                "source": "opengoddard",
                "result": warm_entry["result"],
                **warm_entry.get("plotting", {}),
                "model": warm_entry.get("model"),
            }
        )

        collection_context.finalize_success()
        return {
            "label": COLLECTION_LABEL,
            "entries": collection_results,
            "run_id": collection_context.run_id,
            "run_dir": str(collection_context.run_dir),
            "plot_output_dir": str(collection_context.plot_dir),
            "summary_path": str(collection_context.summary_path),
            "manifest_path": str(collection_context.manifest_path),
            "config_path": str(collection_context.config_path),
            "scenario": config["scenario"],
        }
    except Exception as error:
        collection_context.finalize_failure(error)
        raise


def main(
    *,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    n_adam: int = PAPER_N_ADAM,
    n_lbfgs: int = PAPER_N_LBFGS,
    convergence_threshold: float = PAPER_CONVERGENCE_THRESHOLD,
    baseline_max_iteration: int = BASELINE_MAX_ITERATION,
    baseline_ftol: float = BASELINE_FTOL,
    baseline_slsqp_maxiter: int = BASELINE_SLSQP_MAXITER,
    include_alpha_guard: bool = False,
    skip_plots: bool = False,
    print_summary: bool = True,
    smoke: bool | None = None,
):
    collection_run = run_collection(
        t_final_seconds=t_final_seconds,
        n_adam=n_adam,
        n_lbfgs=n_lbfgs,
        convergence_threshold=convergence_threshold,
        baseline_max_iteration=baseline_max_iteration,
        baseline_ftol=baseline_ftol,
        baseline_slsqp_maxiter=baseline_slsqp_maxiter,
        include_alpha_guard=include_alpha_guard,
        smoke=smoke,
    )
    if print_summary:
        print_collection_run_summary(collection_run)
        print_baseline_delta_v_summary(
            collection_run["entries"],
            title="Rendezvous Hold Point ECI",
            baseline_labels=(BASELINE_LABEL, WARMSTART_BASELINE_LABEL),
        )
    if not skip_plots:
        plot_results(
            collection_run["entries"],
            output_dir=collection_run["plot_output_dir"],
            scenario=collection_run["scenario"],
        )
    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        t_final_seconds=args.t_final_seconds,
        n_adam=args.n_adam,
        n_lbfgs=args.n_lbfgs,
        convergence_threshold=args.convergence_threshold,
        baseline_max_iteration=args.baseline_max_iteration,
        baseline_ftol=args.baseline_ftol,
        baseline_slsqp_maxiter=args.baseline_slsqp_maxiter,
        include_alpha_guard=args.include_alpha_guard,
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
    )

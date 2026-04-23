from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import spacepinn
import torch
from spacepinn.config.config_orbit_transfer import (
    GM_EARTH,
    R_EARTH,
    R_LEO,
    Orbit,
    circular_ot_kinematic_polar_config,
)
from spacepinn.config.transform_functions import (
    kinematic_polar_positive_radius_landing_fixed_angle_fn,
    kinematic_polar_positive_radius_landing_fn,
)
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.opengoddard.descent_landing_2d_goddard import (
    kinematic_descent_landing_2d_goddard,
)
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import paper_baseline_solver_kwargs
from spacepinn.plotting.helpers import get_quiver_data, register_plot_artifact_if_possible
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "descent_landing_2d"
FIG_PREFIX = "descent_landing_2d"
MAIN_FIGSIZE = (6.0, 6.0)
ORBIT_FIGSIZE = (7.4, 7.4)
PINN_LABEL = "PINN with exact BC"
PINN_COLOR = "#2ca02c"
BASELINE_LABEL = "Baseline (OpenGoddard)"
BASELINE_COLOR = PALETTE["opengoddard"]
ORBIT_RADII = [(R_EARTH, "Earth radius"), (R_LEO, "LEO")]
PAPER_N_ADAM = 10_000
PAPER_N_LBFGS = 0
PAPER_CONVERGENCE_THRESHOLD = 1e-6
ATMOSPHERE_BETA = 0.01  # lumped C_d A / m [m^2 / kg]
ATMOSPHERE_RHO0 = 1.225  # sea-level density [kg / m^3]
ATMOSPHERE_SCALE_HEIGHT_KM = 7.2  # exponential scale height [km]
ATMOSPHERE_LAYER_BREAK_KM = 50.0
ATMOSPHERE_UPPER_SCALE_HEIGHT_KM = 25.0


def _parse_args():
    parser = argparse.ArgumentParser(description="Paper 2D polar descent / landing maneuver from HEO to Earth.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--alpha-final-pi", type=float, default=0.5, help="Initial guess multiplier so alpha_N_init = alpha_final_pi * pi.")
    parser.add_argument("--time-guess-scale", type=float, default=1.0, help="Scale applied to the Hohmann-like descent time guess.")
    parser.add_argument(
        "--fixed-final-angle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fix the terminal polar angle instead of learning a free final angle.",
    )
    parser.add_argument("--baseline-max-iteration", type=int, default=10, help="Outer OpenGoddard iteration budget.")
    parser.add_argument(
        "--atmosphere",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable a simple exponential atmosphere with drag.",
    )
    return parser.parse_args()


def _descent_time_guess(*, time_guess_scale: float) -> float:
    hohmann_like = np.pi * np.sqrt((R_LEO + R_EARTH) ** 3 / (8.0 * GM_EARTH))
    return float(hohmann_like * time_guess_scale)


def _exponential_drag_polar_acceleration(
    *,
    rho: torch.Tensor,
    alpha: torch.Tensor,
    v_rho: torch.Tensor,
    v_alpha: torch.Tensor,
    r_cart: torch.Tensor,
    t: torch.Tensor,
    t_total,
    beta: float,
    rho0: float,
    scale_height_km: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del alpha, r_cart, t, t_total

    altitude_km = torch.clamp(rho - R_EARTH, min=0.0)
    density = rho0 * torch.exp(-altitude_km / scale_height_km)

    speed_km_s = torch.sqrt(v_rho**2 + v_alpha**2 + 1e-12)
    speed_m_s = speed_km_s * 1000.0
    drag_m_s2 = 0.5 * density * beta * speed_m_s**2
    drag_km_s2 = drag_m_s2 / 1000.0

    extra_rho = -drag_km_s2 * (v_rho / speed_km_s)
    extra_alpha = -drag_km_s2 * (v_alpha / speed_km_s)
    return extra_rho, extra_alpha


def _piecewise_exponential_drag_polar_acceleration(
    *,
    rho: torch.Tensor,
    alpha: torch.Tensor,
    v_rho: torch.Tensor,
    v_alpha: torch.Tensor,
    r_cart: torch.Tensor,
    t: torch.Tensor,
    t_total,
    beta: float,
    rho0: float,
    lower_scale_height_km: float,
    upper_scale_height_km: float,
    layer_break_km: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del alpha, r_cart, t, t_total

    altitude_km = torch.clamp(rho - R_EARTH, min=0.0)
    break_density = rho0 * np.exp(-layer_break_km / lower_scale_height_km)
    lower_density = rho0 * torch.exp(-altitude_km / lower_scale_height_km)
    upper_density = break_density * torch.exp(-(altitude_km - layer_break_km) / upper_scale_height_km)
    density = torch.where(altitude_km <= layer_break_km, lower_density, upper_density)

    speed_km_s = torch.sqrt(v_rho**2 + v_alpha**2 + 1e-12)
    speed_m_s = speed_km_s * 1000.0
    drag_m_s2 = 0.5 * density * beta * speed_m_s**2
    drag_km_s2 = drag_m_s2 / 1000.0

    extra_rho = -drag_km_s2 * (v_rho / speed_km_s)
    extra_alpha = -drag_km_s2 * (v_alpha / speed_km_s)
    return extra_rho, extra_alpha


def _drag_density_numpy(altitude_km: np.ndarray, *, drag_model: str | None) -> np.ndarray:
    model = (drag_model or "none").strip().lower()
    altitude = np.clip(np.asarray(altitude_km, dtype=float), a_min=0.0, a_max=None)
    if model in {"none", "vacuum"}:
        return np.zeros_like(altitude)
    if model in {"exponential", "single_scale"}:
        return ATMOSPHERE_RHO0 * np.exp(-altitude / ATMOSPHERE_SCALE_HEIGHT_KM)
    if model in {"piecewise_exponential", "two_layer"}:
        break_density = ATMOSPHERE_RHO0 * np.exp(-ATMOSPHERE_LAYER_BREAK_KM / ATMOSPHERE_SCALE_HEIGHT_KM)
        lower_density = ATMOSPHERE_RHO0 * np.exp(-altitude / ATMOSPHERE_SCALE_HEIGHT_KM)
        upper_density = break_density * np.exp(-(altitude - ATMOSPHERE_LAYER_BREAK_KM) / ATMOSPHERE_UPPER_SCALE_HEIGHT_KM)
        return np.where(altitude <= ATMOSPHERE_LAYER_BREAK_KM, lower_density, upper_density)
    raise ValueError(f"Unsupported drag_model '{drag_model}'.")


def compute_drag_magnitude(result, *, drag_model: str | None) -> np.ndarray:
    r_polar = np.asarray(getattr(result, "r_polar", result.r), dtype=float)
    v_polar = np.asarray(getattr(result, "v_polar", result.v), dtype=float)
    rho = r_polar[:, 0]
    v_rho = v_polar[:, 0]
    v_alpha = v_polar[:, 1]
    altitude_km = np.clip(rho - R_EARTH, a_min=0.0, a_max=None)
    density = _drag_density_numpy(altitude_km, drag_model=drag_model)
    speed_km_s = np.sqrt(v_rho**2 + v_alpha**2 + 1e-12)
    speed_m_s = speed_km_s * 1000.0
    drag_m_s2 = 0.5 * density * ATMOSPHERE_BETA * speed_m_s**2
    return drag_m_s2 / 1000.0


def make_drag_acceleration_fn(drag_model: str | None):
    model = (drag_model or "none").strip().lower()
    if model in {"none", "vacuum"}:
        return None
    if model in {"exponential", "single_scale"}:
        return partial(
            _exponential_drag_polar_acceleration,
            beta=ATMOSPHERE_BETA,
            rho0=ATMOSPHERE_RHO0,
            scale_height_km=ATMOSPHERE_SCALE_HEIGHT_KM,
        )
    if model in {"piecewise_exponential", "two_layer"}:
        return partial(
            _piecewise_exponential_drag_polar_acceleration,
            beta=ATMOSPHERE_BETA,
            rho0=ATMOSPHERE_RHO0,
            lower_scale_height_km=ATMOSPHERE_SCALE_HEIGHT_KM,
            upper_scale_height_km=ATMOSPHERE_UPPER_SCALE_HEIGHT_KM,
            layer_break_km=ATMOSPHERE_LAYER_BREAK_KM,
        )
    raise ValueError(f"Unsupported drag_model '{drag_model}'.")


def build_config(
    *,
    alpha_final_pi: float = 0.5,
    time_guess_scale: float = 1.0,
    fixed_final_angle: bool = False,
    atmosphere: bool = False,
    drag_model: str | None = None,
    label: str | None = None,
    color: str | None = None,
    smoke: bool | None = None,
) -> dict:
    config = deepcopy(circular_ot_kinematic_polar_config)
    alpha_N_initial = float(alpha_final_pi * np.pi)
    t_total_initial = _descent_time_guess(time_guess_scale=time_guess_scale)
    resolved_drag_model = drag_model if drag_model is not None else ("exponential" if atmosphere else "none")

    config["label"] = label or PINN_LABEL
    config["extra_parameters"] = {
        "t_total": torch.nn.Parameter(torch.tensor(t_total_initial, dtype=torch.float32)),
    }
    if fixed_final_angle:
        config["pinn"]["output_transform_fn"] = partial(
            kinematic_polar_positive_radius_landing_fixed_angle_fn,
            x0=torch.tensor([R_LEO, 0.0], dtype=torch.float64),
            xN=torch.tensor([R_EARTH, alpha_N_initial], dtype=torch.float64),
            vt_0=float(Orbit.LEO.V),
            vt_N=0.0,
        )
    else:
        config["extra_parameters"]["alpha_N"] = torch.nn.Parameter(torch.tensor(alpha_N_initial, dtype=torch.float32))
        config["pinn"]["output_transform_fn"] = partial(
            kinematic_polar_positive_radius_landing_fn,
            x0=torch.tensor([R_LEO, 0.0], dtype=torch.float64),
            rho_N=float(R_EARTH),
            vt_0=float(Orbit.LEO.V),
            vt_N=0.0,
        )
    config["optimizer"]["coordinate_system"] = "polar"
    config["optimizer"]["r0"] = torch.tensor([R_LEO, 0.0], dtype=torch.float32)
    config["optimizer"]["rN"] = torch.tensor([R_EARTH, alpha_N_initial], dtype=torch.float32)
    config["optimizer"]["t_total"] = torch.tensor(t_total_initial, dtype=torch.float32)
    config["optimizer"]["n_adam"] = PAPER_N_ADAM
    config["optimizer"]["n_lbfgs"] = PAPER_N_LBFGS
    config["optimizer"]["convergence_threshold"] = PAPER_CONVERGENCE_THRESHOLD
    config["optimizer"]["external_acceleration_fn"] = make_drag_acceleration_fn(resolved_drag_model)
    config["plotting"]["color"] = color or PINN_COLOR
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["plotting"]["quiver_scale"] = 1 / 250
    config["plotting"]["quiver_count"] = 10
    config["scenario"] = {
        "alpha_final_pi": float(alpha_final_pi),
        "alpha_final_rad": float(alpha_N_initial),
        "time_guess_scale": float(time_guess_scale),
        "fixed_final_angle": bool(fixed_final_angle),
        "atmosphere": bool(atmosphere),
        "drag_model": resolved_drag_model,
    }

    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    if smoke_enabled:
        config["optimizer"]["n_adam"] = 1
        config["optimizer"]["n_lbfgs"] = 0
    return config


def build_baseline_entry(
    *,
    time_guess_scale: float = 1.0,
    alpha_final_pi: float = 0.5,
    fixed_final_angle: bool = False,
    baseline_max_iteration: int = 10,
    atmosphere: bool = False,
    drag_model: str | None = None,
    smoke: bool | None = None,
) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    solver_kwargs = paper_baseline_solver_kwargs(smoke_enabled=smoke_enabled)
    solver_kwargs["max_iteration"] = 1 if smoke_enabled else int(baseline_max_iteration)
    resolved_drag_model = drag_model if drag_model is not None else ("exponential" if atmosphere else "none")
    result = kinematic_descent_landing_2d_goddard(
        BASELINE_LABEL,
        **solver_kwargs,
        time_final_guess=_descent_time_guess(time_guess_scale=time_guess_scale),
        alpha_final=(float(alpha_final_pi) * np.pi) if fixed_final_angle else None,
        drag_model=resolved_drag_model,
    )
    result["color"] = BASELINE_COLOR
    result["linestyle"] = "solid"
    result["trajectory_linestyle"] = "solid"
    result["quiver_count"] = 10
    result["zorder"] = 2
    return {
        "label": result["label"],
        "result": result["result"],
        "model": result.get("model"),
        "config": result.get("config"),
        "plotting": {
            key: result[key]
            for key in ("linestyle", "trajectory_linestyle", "color", "quiver_scale", "quiver_count", "zorder")
            if key in result
        },
        "source": "opengoddard",
    }


def plot_orbit_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=ORBIT_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    fig, ax = plt.subplots(figsize=ORBIT_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.r[:, 0],
            result.r[:, 1],
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        r_q, _, T_q = get_quiver_data(
            result,
            step=exp.get("quiver_step", 10),
            count=exp.get("quiver_count"),
        )
        ax.quiver(
            r_q[:, 0],
            r_q[:, 1],
            T_q[:, 0],
            T_q[:, 1],
            color="k",
            scale=exp["quiver_scale"] * 25.0,
            label="_nolegend_",
        )

    reference_result = entries[0]["result"]
    ax.plot(reference_result.r0[0], reference_result.r0[1], "o", color="red", markersize=6, label=r"$r(t=0)$")
    ax.plot(
        reference_result.rN[0],
        reference_result.rN[1],
        "x",
        color="red",
        markersize=8,
        markeredgewidth=1.8,
        label="Initial guess",
    )

    earth = plt.Circle((0, 0), R_EARTH, facecolor="#9ecae1", edgecolor="black", linewidth=plotter.secondary_linewidth, label="Earth")
    ax.add_patch(earth)
    heo_circle = plt.Circle(
        (0, 0),
        R_LEO,
        color="darkgrey",
        fill=False,
        linestyle="dashed",
        linewidth=plotter.secondary_linewidth,
        label="LEO",
    )
    ax.add_patch(heo_circle)
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.set_aspect("equal")
    ax.set_ylim(-1000.0, 9000.0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="k", linewidth=1.8))
    labels.append("Thrust vectors")
    ax.legend(handles, labels, loc="lower center", framealpha=0.95, facecolor="white", edgecolor="0.3", ncol=2)
    ax.margins(x=0.05, y=0.02)
    fig.tight_layout()
    figure_path = plotter._build_figure_path("orbit_traj")
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)


def _time_minutes(result) -> np.ndarray:
    t = np.asarray(result.t, dtype=float).reshape(-1)
    t_total = float(getattr(result, "t_total", 1.0))
    return t * t_total / 60.0


def _set_time_minutes_limits(ax, time_minutes: np.ndarray, *, pad_fraction: float = 0.015) -> None:
    t_min = float(np.min(time_minutes))
    t_max = float(np.max(time_minutes))
    span = max(t_max - t_min, 1e-9)
    pad = span * pad_fraction
    ax.set_xlim(t_min - pad, t_max + pad)


def plot_thrust_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        thrust_mag = np.asarray(result.F_mag, dtype=float).reshape(-1)
        ax.plot(
            _time_minutes(result),
            thrust_mag,
            color=exp["color"],
            linestyle=exp["linestyle"],
            linewidth=plotter.main_linewidth,
            label=label,
        )

    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Thrust magnitude / km s$^{-2}$")
    all_times = np.concatenate([_time_minutes(exp["result"]) for exp in plotter.experiments.values()])
    _set_time_minutes_limits(ax, all_times)
    ax.legend(loc="best", framealpha=0.95, facecolor="white", edgecolor="0.3")
    fig.tight_layout()
    figure_path = plotter._build_figure_path("thrust")
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)


def plot_gravity_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)
    label_to_config = {str(entry.get("label", "")): entry.get("config") for entry in entries}

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        time_minutes = _time_minutes(result)
        a_mag = np.asarray(result.a_mag, dtype=float).reshape(-1)
        g_mag = np.asarray(result.G_mag, dtype=float).reshape(-1)
        color = exp["color"]
        entry_config = label_to_config.get(str(label))
        drag_model = None
        if isinstance(entry_config, dict):
            scenario = entry_config.get("scenario")
            if isinstance(scenario, dict):
                drag_model = scenario.get("drag_model")
            if drag_model is None:
                spaceship = entry_config.get("spaceship")
                if isinstance(spaceship, dict):
                    drag_model = spaceship.get("drag_model")
        drag_mag = compute_drag_magnitude(result, drag_model=drag_model)
        ax.plot(
            time_minutes,
            a_mag,
            color=color,
            linestyle="solid",
            linewidth=plotter.main_linewidth,
            label=f"{label} RFM",
        )
        ax.plot(
            time_minutes,
            g_mag,
            color=color,
            linestyle="dashed",
            linewidth=plotter.secondary_linewidth,
            label=f"{label} Gravity",
        )
        if np.max(drag_mag) > 0.0:
            ax.plot(
                time_minutes,
                drag_mag,
                color=color,
                linestyle="dotted",
                linewidth=plotter.secondary_linewidth,
                label=f"{label} Drag",
            )

    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Gravity / Required Force magnitude / km s$^{-2}$")
    all_times = np.concatenate([_time_minutes(exp["result"]) for exp in plotter.experiments.values()])
    _set_time_minutes_limits(ax, all_times)
    ax.legend(loc="upper center", framealpha=0.95, facecolor="white", edgecolor="0.3", fontsize=9)
    fig.tight_layout()
    figure_path = plotter._build_figure_path("gravity")
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)


def plot_altitude_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        altitude_km = np.asarray(getattr(result, "r_polar", result.r), dtype=float)[:, 0] - R_EARTH
        ax.plot(
            _time_minutes(result),
            altitude_km,
            color=exp["color"],
            linestyle=exp["linestyle"],
            linewidth=plotter.main_linewidth,
            label=label,
        )

    ax.set_xlabel("Time / min")
    ax.set_ylabel("Height above surface / km")
    all_times = np.concatenate([_time_minutes(exp["result"]) for exp in plotter.experiments.values()])
    _set_time_minutes_limits(ax, all_times)
    ax.legend(loc="best", framealpha=0.95, facecolor="white", edgecolor="0.3")
    fig.tight_layout()
    figure_path = plotter._build_figure_path("altitude")
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)


def plot_polar_thrust_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t = np.asarray(result.t, dtype=float).reshape(-1)
        if getattr(result, "F_rho", None) is None or getattr(result, "F_alpha", None) is None:
            continue
        F_rho = np.asarray(result.F_rho, dtype=float).reshape(-1)
        F_alpha = np.asarray(result.F_alpha, dtype=float).reshape(-1)
        color = PINN_COLOR if label == PINN_LABEL else BASELINE_COLOR if label == BASELINE_LABEL else exp["color"]
        ax.plot(
            t,
            F_rho,
            color=color,
            linestyle="solid",
            linewidth=plotter.main_linewidth,
            label=rf"{label} $F_\rho$",
        )
        ax.plot(
            t,
            F_alpha,
            color=color,
            linestyle="dashed",
            linewidth=plotter.secondary_linewidth,
            label=rf"{label} $F_\alpha$",
        )


def _apply_paper_entry_styles(entries: list[dict]) -> list[dict]:
    styled_entries = []
    for entry in entries:
        styled = dict(entry)
        plotting = dict(entry.get("plotting", {}))
        label = entry.get("label")
        if isinstance(label, str) and label.startswith(PINN_LABEL):
            styled["label"] = PINN_LABEL
            label = PINN_LABEL
        if label == PINN_LABEL:
            plotting.setdefault("color", PINN_COLOR)
            plotting.setdefault("linestyle", "solid")
            plotting.setdefault("trajectory_linestyle", "solid")
        elif label == BASELINE_LABEL:
            plotting.setdefault("color", BASELINE_COLOR)
            plotting.setdefault("linestyle", "solid")
            plotting.setdefault("trajectory_linestyle", "solid")
        styled["plotting"] = plotting
        styled.update(plotting)
        styled_entries.append(styled)
    return styled_entries

    ax.set_xlabel("Normalized time")
    ax.set_ylabel(r"Thrust components / km s$^{-2}$")
    ax.set_xlim(0.0, 1.0)
    ax.legend(framealpha=0.95, facecolor="white", edgecolor="0.3")
    fig.tight_layout()
    figure_path = plotter._build_figure_path("thrust_polar")
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)


def main(
    *,
    skip_plots: bool = False,
    print_summary: bool = True,
    alpha_final_pi: float = 0.5,
    time_guess_scale: float = 1.0,
    fixed_final_angle: bool = False,
    atmosphere: bool = False,
    baseline_max_iteration: int = 10,
    smoke: bool | None = None,
):
    config = build_config(
        alpha_final_pi=alpha_final_pi,
        time_guess_scale=time_guess_scale,
        fixed_final_angle=fixed_final_angle,
        atmosphere=atmosphere,
        smoke=smoke,
    )
    additional_entries = []
    additional_entries.append(
        capture_baseline_entry(
            lambda: build_baseline_entry(
                time_guess_scale=time_guess_scale,
                alpha_final_pi=alpha_final_pi,
                fixed_final_angle=fixed_final_angle,
                baseline_max_iteration=baseline_max_iteration,
                atmosphere=atmosphere,
                smoke=smoke,
            ),
            log_filename="baseline_opengoddard.log",
        )
    )
    collection_run = run_experiment_collection(
        configs=[config],
        additional_entries=additional_entries,
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        styled_entries = _apply_paper_entry_styles(collection_run["entries"])
        plotter = TrajectoryPlotter(
            styled_entries,
            dim=2,
            figsize=MAIN_FIGSIZE,
            fig_prefix=FIG_PREFIX,
            output_dir=collection_run["plot_output_dir"],
        )
        plot_thrust_figure(styled_entries, output_dir=collection_run["plot_output_dir"])
        plot_polar_thrust_figure(styled_entries, output_dir=collection_run["plot_output_dir"])
        plot_gravity_figure(styled_entries, output_dir=collection_run["plot_output_dir"])
        plot_altitude_figure(styled_entries, output_dir=collection_run["plot_output_dir"])
        plotter.plot_loss()
        plot_orbit_figure(styled_entries, output_dir=collection_run["plot_output_dir"])

    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        alpha_final_pi=args.alpha_final_pi,
        time_guess_scale=args.time_guess_scale,
        fixed_final_angle=args.fixed_final_angle,
        atmosphere=args.atmosphere,
        baseline_max_iteration=args.baseline_max_iteration,
    )

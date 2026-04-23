from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spacepinn
import torch
from spacepinn.config.config_orbit_transfer import (
    Orbit,
    R_GEO,
    R_HEO,
    R_LEO,
    OrbitalTransferBC,
    circular_ot_kinematic_polar_config,
)
from spacepinn.config.transform_functions import kinematic_polar_fn
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import (
    PAPER_BASELINE_MAX_ITERATION,
    paper_baseline_solver_kwargs,
)
from spacepinn.paper._baseline_summary import print_baseline_delta_v_summary
from spacepinn.opengoddard.circular_orbit_transfer_goddard_no_alpha import (
    kinematic_ot_goddard_free_final_angle_no_alpha,
)
from spacepinn.plotting.helpers import register_plot_artifact_if_possible
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "low_thrust_transfer"
FIG_PREFIX = "low_thrust_transfer"
MAIN_FIGSIZE = (6.0, 6.0)
TIME_AXIS_PADDING_FRACTION = 0.02
TARGET_ORBITS = {
    "heo": Orbit.HEO,
    "geo": Orbit.GEO,
}
PINN_LABEL = "PINN with exact BC"
BASELINE_LABEL = "Baseline (OpenGoddard)"
PINN_COLOR = "#2ca02c"
BASELINE_COLOR = "#4d4d4d"
MAIN_LINEWIDTH = 2.8
SECONDARY_LINEWIDTH = 2.4
OPENGODDARD_MAX_ITERATION = PAPER_BASELINE_MAX_ITERATION
PAPER_N_ADAM = 10_000
PAPER_N_LBFGS = 0
PAPER_CONVERGENCE_THRESHOLD = 1e-5


def _parse_args():
    parser = argparse.ArgumentParser(description="Paper low-thrust free-final-angle orbit transfer.")
    parser.add_argument("--terminal-angle-pi", type=float, default=None)
    parser.add_argument("--time-guess-scale", type=float, default=None)
    parser.add_argument("--extra-turns", type=int, default=None)
    parser.add_argument("--tof-scale", type=float, default=None)
    parser.add_argument("--target-orbit", choices=sorted(TARGET_ORBITS), default="geo")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def _resolve_initial_guess_parameters(
    *,
    target_orbit: str,
    terminal_angle_pi: float | None,
    time_guess_scale: float | None,
    extra_turns: int | None,
    tof_scale: float | None,
) -> tuple[float, float]:
    default_terminal_angle_pi = 1.0
    default_time_guess_scale = 1.0
    if target_orbit == "geo":
        default_terminal_angle_pi = 5.0
        default_time_guess_scale = 2.0
    resolved_terminal_angle_pi = terminal_angle_pi
    if resolved_terminal_angle_pi is None and extra_turns is not None:
        resolved_terminal_angle_pi = 1.0 + 2.0 * float(extra_turns)
    resolved_time_guess_scale = time_guess_scale
    if resolved_time_guess_scale is None and tof_scale is not None:
        resolved_time_guess_scale = float(tof_scale)
    return (
        default_terminal_angle_pi if resolved_terminal_angle_pi is None else float(resolved_terminal_angle_pi),
        default_time_guess_scale if resolved_time_guess_scale is None else float(resolved_time_guess_scale),
    )


def _resolve_baseline_time_guess(*, transfer_bc: OrbitalTransferBC, time_guess_scale: float) -> float:
    return float(transfer_bc.T_hohnmann * time_guess_scale)


def kinematic_polar_free_final_angle_fn(t, x, x0, rho_N, vt_0, vt_N, model):
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    rho_N_tensor = torch.as_tensor(rho_N, device=t.device, dtype=t.dtype).reshape(())
    vt_0_tensor = torch.as_tensor(vt_0, device=t.device, dtype=t.dtype).reshape(())
    vt_N_tensor = torch.as_tensor(vt_N, device=t.device, dtype=t.dtype).reshape(())
    alpha_N_tensor = model.alpha_N.to(device=t.device, dtype=t.dtype).reshape(())
    xN_tensor = torch.stack((rho_N_tensor, alpha_N_tensor))
    v0_tensor = torch.stack((torch.zeros_like(vt_0_tensor), vt_0_tensor))
    vN_tensor = torch.stack((torch.zeros_like(vt_N_tensor), vt_N_tensor))
    return kinematic_polar_fn(
        t=t,
        x=x,
        x0=x0_tensor,
        xN=xN_tensor,
        v0=v0_tensor,
        vN=vN_tensor,
        model=model,
        transform_only_R=False,
    )


def build_config(
    *,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> dict:
    terminal_angle_pi, time_guess_scale = _resolve_initial_guess_parameters(
        target_orbit=target_orbit,
        terminal_angle_pi=terminal_angle_pi,
        time_guess_scale=time_guess_scale,
        extra_turns=extra_turns,
        tof_scale=tof_scale,
    )
    transfer_bc = OrbitalTransferBC(Orbit.LEO, TARGET_ORBITS[target_orbit], alpha_T=np.pi, coordinate_system="polar")
    config = deepcopy(circular_ot_kinematic_polar_config)

    alpha_N_initial = float(np.pi * terminal_angle_pi)
    t_total_initial = float(transfer_bc.T_hohnmann * time_guess_scale)

    config["label"] = PINN_LABEL
    config["extra_parameters"] = {
        "t_total": torch.nn.Parameter(torch.tensor(t_total_initial, dtype=torch.float32)),
        "alpha_N": torch.nn.Parameter(torch.tensor(alpha_N_initial, dtype=torch.float32)),
    }
    config["pinn"]["output_transform_fn"] = partial(
        kinematic_polar_free_final_angle_fn,
        x0=transfer_bc.x0.clone().detach(),
        rho_N=float(transfer_bc.xN[0].item()),
        vt_0=float(transfer_bc.v0[1].item()),
        vt_N=float(transfer_bc.vN[1].item()),
    )
    config["optimizer"]["r0"] = transfer_bc.x0.clone().detach()
    config["optimizer"]["rN"] = torch.tensor([float(transfer_bc.xN[0].item()), alpha_N_initial], dtype=torch.float32)
    config["optimizer"]["t_total"] = torch.tensor(t_total_initial, dtype=torch.float32)
    config["optimizer"]["n_adam"] = PAPER_N_ADAM
    config["optimizer"]["n_lbfgs"] = PAPER_N_LBFGS
    config["optimizer"]["convergence_threshold"] = PAPER_CONVERGENCE_THRESHOLD
    config["plotting"]["color"] = PINN_COLOR
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["plotting"]["quiver_scale"] = 1 / 250

    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    if smoke_enabled:
        config["optimizer"]["n_adam"] = 1
        config["optimizer"]["n_lbfgs"] = 0
    return config


def build_baseline_entry(
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> dict:
    _, time_guess_scale = _resolve_initial_guess_parameters(
        target_orbit=target_orbit,
        terminal_angle_pi=None,
        time_guess_scale=time_guess_scale,
        extra_turns=None,
        tof_scale=tof_scale,
    )
    transfer_bc = OrbitalTransferBC(Orbit.LEO, TARGET_ORBITS[target_orbit], alpha_T=np.pi, coordinate_system="polar")
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    result = kinematic_ot_goddard_free_final_angle_no_alpha(
        BASELINE_LABEL,
        **paper_baseline_solver_kwargs(smoke_enabled=smoke_enabled),
        transfer_bc=transfer_bc,
        time_final_guess=_resolve_baseline_time_guess(transfer_bc=transfer_bc, time_guess_scale=time_guess_scale),
        time_final_upper_bound=None,
    )
    result["color"] = BASELINE_COLOR
    result["linestyle"] = "solid"
    result["trajectory_linestyle"] = "solid"
    result["source"] = "opengoddard"
    result["zorder"] = 2
    return {
        "label": result["label"],
        "result": result["result"],
        "model": result.get("model"),
        "config": result.get("config"),
        "plotting": {
            key: result[key]
            for key in ("linestyle", "trajectory_linestyle", "color", "quiver_scale", "zorder")
            if key in result
        },
        "source": "opengoddard",
    }


def _physical_time_minutes(result) -> tuple[np.ndarray, float]:
    t_total_minutes = float(result.t_total) / 60.0
    t_minutes = np.asarray(result.t, dtype=float).reshape(-1) * t_total_minutes
    t_padding = TIME_AXIS_PADDING_FRACTION * t_total_minutes
    return t_minutes, t_padding


def plot_thrust_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    plotter.main_linewidth = MAIN_LINEWIDTH
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = _physical_time_minutes(result)
        ax.plot(
            t_minutes,
            np.maximum(result.F_mag, 1e-12),
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )

    max_t_minutes = 0.0
    max_t_padding = 0.0
    for exp in plotter.experiments.values():
        _, t_padding = _physical_time_minutes(exp["result"])
        max_t_minutes = max(max_t_minutes, float(exp["result"].t_total) / 60.0)
        max_t_padding = max(max_t_padding, t_padding)
    ax.set_yscale("log")
    ax.set_xlim(-max_t_padding, max_t_minutes + max_t_padding)
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Thrust magnitude / km s$^{-2}$")
    ax.legend(framealpha=0.98, facecolor="white", edgecolor="0.3")
    fig.tight_layout()
    figure_path = plotter._build_figure_path("thrust")
    fig.savefig(figure_path)
    register_plot_artifact_if_possible(figure_path)


def plot_gravity_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = SECONDARY_LINEWIDTH
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        t_minutes, _ = _physical_time_minutes(result)
        ax.plot(
            t_minutes,
            np.maximum(result.a_mag, 1e-12),
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=f"{label} RFM",
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        ax.plot(
            t_minutes,
            np.maximum(result.G_mag, 1e-12),
            linestyle="dashed" if exp["linestyle"] == "solid" else exp["linestyle"],
            color=exp["color"],
            label=f"{label} Gravity",
            linewidth=plotter.secondary_linewidth,
            zorder=exp["zorder"],
        )

    max_t_minutes = 0.0
    max_t_padding = 0.0
    max_force = 0.0
    for exp in plotter.experiments.values():
        _, t_padding = _physical_time_minutes(exp["result"])
        max_t_minutes = max(max_t_minutes, float(exp["result"].t_total) / 60.0)
        max_t_padding = max(max_t_padding, t_padding)
        result = exp["result"]
        max_force = max(
            max_force,
            float(np.max(np.asarray(result.a_mag, dtype=float))),
            float(np.max(np.asarray(result.G_mag, dtype=float))),
        )
    ax.set_yscale("log")
    ax.set_xlim(-max_t_padding, max_t_minutes + max_t_padding)
    ax.set_ylim(top=max_force * 2.0 if max_force > 0 else None)
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Gravity / Required Force magnitude / km s$^{-2}$")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        ncol=2,
        fontsize=8,
        columnspacing=1.0,
        handlelength=1.6,
        labelspacing=0.35,
        borderaxespad=0.35,
        framealpha=0.98,
        facecolor="white",
        edgecolor="0.3",
    )
    fig.tight_layout()
    figure_path = plotter._build_figure_path("gravity")
    fig.savefig(figure_path)
    register_plot_artifact_if_possible(figure_path)


def plot_orbit_figure(entries: list[dict], *, output_dir: str, target_orbit: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    plotter.main_linewidth = MAIN_LINEWIDTH
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

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

    reference_entry = entries[0]
    reference_result = reference_entry["result"]
    optimizer_cfg = (reference_entry.get("config") or {}).get("optimizer", {})
    initial_terminal_polar = optimizer_cfg.get("rN")
    if isinstance(initial_terminal_polar, dict):
        initial_terminal_polar = initial_terminal_polar.get("value")
    initial_terminal_guess_xy = None
    if initial_terminal_polar is not None:
        initial_terminal_polar = np.asarray(initial_terminal_polar, dtype=float).reshape(-1)
        if initial_terminal_polar.size >= 2:
            radius_guess = float(initial_terminal_polar[0])
            alpha_guess = float(initial_terminal_polar[1])
            initial_terminal_guess_xy = (
                radius_guess * np.cos(alpha_guess),
                radius_guess * np.sin(alpha_guess),
            )

    ax.plot(reference_result.r0[0], reference_result.r0[1], "o", color="red", markersize=6, label=r"$r(t=0)$")
    if initial_terminal_guess_xy is not None:
        ax.plot(
            initial_terminal_guess_xy[0],
            initial_terminal_guess_xy[1],
            "x",
            color="red",
            markersize=8,
            markeredgewidth=1.8,
            label="Initial terminal-angle guess",
        )

    for idx, (radius, name) in enumerate(((R_LEO, "LEO"), (R_HEO if target_orbit == "heo" else R_GEO, target_orbit.upper()))):
        ax.add_patch(
            plt.Circle(
                (0, 0),
                radius,
                color="black" if idx == 0 else "darkgrey",
                fill=False,
                linestyle="dashed",
                linewidth=plotter.secondary_linewidth,
                label=name,
            )
        )

    ax.scatter(0, 0, color="green", marker="o", s=100, label="Earth")
    ax.set_xlabel("x / km")
    ax.set_ylabel("y / km")
    ax.set_aspect("equal")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        bbox_transform=ax.transAxes,
        ncol=2,
        fontsize=8,
        framealpha=0.98,
        facecolor="white",
        edgecolor="0.3",
    )
    fig.tight_layout()
    figure_path = plotter._build_figure_path("orbit_traj")
    fig.savefig(figure_path)
    register_plot_artifact_if_possible(figure_path)


def main(
    *,
    skip_plots: bool = False,
    print_summary: bool = True,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
):
    config = build_config(
        target_orbit=target_orbit,
        terminal_angle_pi=terminal_angle_pi,
        time_guess_scale=time_guess_scale,
        extra_turns=extra_turns,
        tof_scale=tof_scale,
        smoke=smoke,
    )
    baseline_entry = capture_baseline_entry(
        lambda: build_baseline_entry(
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
        ),
        log_filename="baseline_opengoddard.log",
    )
    collection_run = run_experiment_collection(
        configs=[config],
        additional_entries=[baseline_entry],
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
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
        plotter.plot_loss()
        plot_thrust_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
        plot_gravity_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
        plot_orbit_figure(
            collection_run["entries"],
            output_dir=collection_run["plot_output_dir"],
            target_orbit=target_orbit,
        )

    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        target_orbit=args.target_orbit,
        terminal_angle_pi=args.terminal_angle_pi,
        time_guess_scale=args.time_guess_scale,
        extra_turns=args.extra_turns,
        tof_scale=args.tof_scale,
    )

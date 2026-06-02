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
    H_GEO,
    H_HEO,
    H_LEO,
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
from spacepinn.paper._aggregate_summary import persist_paper_monte_carlo_aggregate_summary
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
from spacepinn.paper._plot_style import MAIN_AXES_RECT, MAIN_FIGSIZE as PAPER_MAIN_FIGSIZE, configure_paper_plotter
from spacepinn.opengoddard.circular_orbit_transfer_goddard_no_alpha import (
    kinematic_ot_goddard_free_final_angle_no_alpha,
)
from spacepinn.plotting.helpers import register_plot_artifact_if_possible
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "orbit_transfer_free_angle"
FIG_PREFIX = "free_terminal_angle"
MAIN_FIGSIZE = PAPER_MAIN_FIGSIZE
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
REPRESENTATIVE_SEED = 5099
MC_SEED_START = 5000
MC_NUM_SEEDS = 100


def _parse_args():
    parser = argparse.ArgumentParser(description="Paper circular orbit transfer with free terminal angle.")
    add_single_mc_arguments(parser, default_mode="single")
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
        default_terminal_angle_pi = 2.10
        default_time_guess_scale = 1.09
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
    seed: int | None = None,
    label_seed: bool = False,
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

    if seed is not None:
        config["seed"] = int(seed)
    config["label"] = label_with_seed(PINN_LABEL, seed) if label_seed and seed is not None else PINN_LABEL
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


def _target_orbit_label(target_orbit: str) -> str:
    if target_orbit == "heo":
        return f"MEO ({H_HEO:.0f} km above Earth)"
    return f"GEO ({H_GEO:.0f} km above Earth)"


def _target_radius_symbol(target_orbit: str) -> str:
    if target_orbit == "heo":
        return r"$R_{\mathrm{MEO}}$"
    return r"$R_{\mathrm{GEO}}$"


def plot_thrust_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    plotter.main_linewidth = MAIN_LINEWIDTH
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

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
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(loc="upper center", framealpha=0.98, facecolor="white", edgecolor="0.3")
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("thrust")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)


def plot_gravity_figure(entries: list[dict], *, output_dir: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = SECONDARY_LINEWIDTH
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

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
    ax.set_ylim(top=max_force * 8.0 if max_force > 0 else None)
    ax.set_xlabel("Time / min")
    ax.set_ylabel(r"Gravity / Required Force magnitude / km s$^{-2}$")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        columnspacing=1.0,
        handlelength=1.6,
        labelspacing=0.28,
        borderaxespad=0.35,
        framealpha=0.98,
        facecolor="white",
        edgecolor="0.3",
    )
    plotter.style_legend(legend)
    figure_path = plotter._build_figure_path("gravity")
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)


def plot_orbit_figure(entries: list[dict], *, output_dir: str, target_orbit: str) -> None:
    plotter = TrajectoryPlotter(entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    configure_paper_plotter(plotter)
    plotter.main_linewidth = MAIN_LINEWIDTH
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

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
    ax.plot(
        reference_result.r0[0],
        reference_result.r0[1],
        "o",
        color="red",
        markersize=6,
        label=r"$\mathbf{r}(t_0)=(R_{\mathrm{LEO}},0)$",
    )
    terminal_point = np.asarray(reference_result.r[-1], dtype=float).reshape(-1)
    ax.plot(
        terminal_point[0],
        terminal_point[1],
        "x",
        color="red",
        markersize=8,
        markeredgewidth=1.8,
        label=rf"$\mathbf{{r}}(T)$ = ({_target_radius_symbol(target_orbit)}, free terminal angle)",
    )

    for idx, (radius, name) in enumerate(
        (
            (R_LEO, f"LEO ({H_LEO:.0f} km above Earth)"),
            (R_HEO if target_orbit == "heo" else R_GEO, _target_orbit_label(target_orbit)),
        )
    ):
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

    ax.scatter(0, 0, color="green", marker="o", s=100, label="_nolegend_", zorder=5)
    ax.annotate(
        "Earth",
        (0, 0),
        xytext=(0, -7),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=plotter.legend_fontsize + 2.0,
        color="black",
        zorder=6,
    )
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


def monte_carlo_group_key(entry: dict) -> str | None:
    return single_group_key(entry, base_label=PINN_LABEL)


def _baseline_entry(
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> dict:
    return capture_baseline_entry(
        lambda: build_baseline_entry(
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
        ),
        log_filename="baseline_opengoddard.log",
    )


def _plot_representative(entries: list[dict], *, output_dir: str, target_orbit: str) -> None:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = SECONDARY_LINEWIDTH
    plotter.plot_loss()
    plot_thrust_figure(entries, output_dir=output_dir)
    plot_gravity_figure(entries, output_dir=output_dir)
    plot_orbit_figure(entries, output_dir=output_dir, target_orbit=target_orbit)


def run_single(
    *,
    representative_seed: int = REPRESENTATIVE_SEED,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> dict:
    config = build_config(
        target_orbit=target_orbit,
        terminal_angle_pi=terminal_angle_pi,
        time_guess_scale=time_guess_scale,
        extra_turns=extra_turns,
        tof_scale=tof_scale,
        seed=representative_seed,
        label_seed=False,
        smoke=smoke,
    )
    return run_experiment_collection(
        configs=[config],
        additional_entries=[
            _baseline_entry(
                target_orbit=target_orbit,
                time_guess_scale=time_guess_scale,
                tof_scale=tof_scale,
                smoke=smoke,
            )
        ],
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
    )


def run_monte_carlo(
    *,
    seed_start: int = MC_SEED_START,
    num_seeds: int = MC_NUM_SEEDS,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> dict:
    seeds = seed_sequence(start=seed_start, count=num_seeds, smoke=smoke)
    configs = [
        build_config(
            target_orbit=target_orbit,
            terminal_angle_pi=terminal_angle_pi,
            time_guess_scale=time_guess_scale,
            extra_turns=extra_turns,
            tof_scale=tof_scale,
            seed=seed,
            label_seed=True,
            smoke=smoke,
        )
        for seed in seeds
    ]
    return run_experiment_collection(
        configs=configs,
        additional_entries=[
            _baseline_entry(
                target_orbit=target_orbit,
                time_guess_scale=time_guess_scale,
                tof_scale=tof_scale,
                smoke=smoke,
            )
        ],
        label=f"{COLLECTION_LABEL}_monte_carlo",
        run_root=str(RUN_ROOT),
    )


def main(
    *,
    mode: str = "single",
    skip_plots: bool = False,
    print_summary: bool = True,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
    representative_seed: int | None = None,
    seed_start: int | None = None,
    num_seeds: int | None = None,
):
    representative_seed = REPRESENTATIVE_SEED if representative_seed is None else int(representative_seed)
    if mode == "mc":
        collection_run = run_monte_carlo(
            seed_start=MC_SEED_START if seed_start is None else int(seed_start),
            num_seeds=MC_NUM_SEEDS if num_seeds is None else int(num_seeds),
            target_orbit=target_orbit,
            terminal_angle_pi=terminal_angle_pi,
            time_guess_scale=time_guess_scale,
            extra_turns=extra_turns,
            tof_scale=tof_scale,
            smoke=smoke,
        )
        persist_paper_monte_carlo_aggregate_summary(
            collection_run,
            title=collection_run["label"],
            baseline_labels=(BASELINE_LABEL,),
            group_key=monte_carlo_group_key,
        )
    else:
        collection_run = run_single(
            representative_seed=representative_seed,
            target_orbit=target_orbit,
            terminal_angle_pi=terminal_angle_pi,
            time_guess_scale=time_guess_scale,
            extra_turns=extra_turns,
            tof_scale=tof_scale,
            smoke=smoke,
        )

    if print_summary:
        print_collection_run_summary(collection_run)
        if mode == "mc":
            grouped = {PINN_LABEL: [entry for entry in collection_run["entries"] if entry.get("source") == "pinn"]}
            print_monte_carlo_summary(grouped, title=collection_run["label"])
        print_baseline_delta_v_summary(
            collection_run["entries"],
            title=collection_run["label"],
            baseline_labels=(BASELINE_LABEL,),
            group_key=monte_carlo_group_key if mode == "mc" else None,
            include_variance=mode == "mc",
        )

    if not skip_plots:
        plot_entries = representative_entries(
            collection_run["entries"],
            representative_seed=representative_seed,
            base_label=PINN_LABEL,
        )
        _plot_representative(plot_entries, output_dir=collection_run["plot_output_dir"], target_orbit=target_orbit)
        if mode == "mc":
            plot_single_group_boxplots(
                collection_run["entries"],
                output_dir=collection_run["plot_output_dir"],
                fig_prefix=FIG_PREFIX,
                base_label=PINN_LABEL,
                baseline_labels=(BASELINE_LABEL,),
            )

    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode=resolve_mode(args),
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        target_orbit=args.target_orbit,
        terminal_angle_pi=args.terminal_angle_pi,
        time_guess_scale=args.time_guess_scale,
        extra_turns=args.extra_turns,
        tof_scale=args.tof_scale,
        representative_seed=args.representative_seed,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
    )

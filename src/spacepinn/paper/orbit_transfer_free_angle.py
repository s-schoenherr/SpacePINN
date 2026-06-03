from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import spacepinn
import torch
from spacepinn.config.config_orbit_transfer import (
    Orbit,
    OrbitalTransferBC,
    circular_ot_kinematic_polar_config,
)
from spacepinn.config.transform_functions import kinematic_polar_fn
from spacepinn.paper.baseline import (
    PAPER_BASELINE_MAX_ITERATION,
    capture_baseline_entry,
    paper_baseline_solver_kwargs,
)
from spacepinn.paper.monte_carlo import (
    add_single_mc_arguments,
    label_with_seed,
    resolve_mode,
    single_group_key,
)
from spacepinn.paper.plots.orbit_transfer import plot_free_orbit_transfer
from spacepinn.paper.runtime import smoke_mode_enabled
from spacepinn.paper.suite import ExperimentSuite, run_experiment_suite
from spacepinn.opengoddard.circular_orbit_transfer_goddard_no_alpha import (
    kinematic_ot_goddard_free_final_angle_no_alpha,
)

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "orbit_transfer_free_angle"
FIG_PREFIX = "free_terminal_angle"
TARGET_ORBITS = {
    "heo": Orbit.HEO,
    "geo": Orbit.GEO,
}
PINN_LABEL = "PINN with exact BC"
BASELINE_LABEL = "Baseline (OpenGoddard)"
PINN_COLOR = "#2ca02c"
BASELINE_COLOR = "#4d4d4d"
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


def _baseline_entries(
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> list[dict]:
    return [
        _baseline_entry(
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
        )
    ]


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
    suite = ExperimentSuite(
        label=COLLECTION_LABEL,
        run_root=RUN_ROOT,
        representative_seed=REPRESENTATIVE_SEED,
        mc_seed_start=MC_SEED_START,
        mc_num_seeds=MC_NUM_SEEDS,
        build_config=lambda seed, label_seed, smoke: build_config(
            target_orbit=target_orbit,
            terminal_angle_pi=terminal_angle_pi,
            time_guess_scale=time_guess_scale,
            extra_turns=extra_turns,
            tof_scale=tof_scale,
            seed=seed,
            label_seed=label_seed,
            smoke=smoke,
        ),
        build_baseline_entries=lambda smoke: _baseline_entries(
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
        ),
        plot_representative=lambda entries, output_dir: plot_free_orbit_transfer(
            entries,
            output_dir=output_dir,
            target_orbit=target_orbit,
        ),
        group_key=monte_carlo_group_key,
        base_label=PINN_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        fig_prefix=FIG_PREFIX,
    )
    return run_experiment_suite(
        suite,
        mode=mode,
        skip_plots=skip_plots,
        print_summary=print_summary,
        smoke=smoke,
        representative_seed=representative_seed,
        seed_start=seed_start,
        num_seeds=num_seeds,
    )


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

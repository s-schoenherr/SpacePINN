from __future__ import annotations

import argparse
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import spacepinn
import torch

from spacepinn.config.config_orbit_transfer import circular_ot_kinematic_polar_config
from spacepinn.config.transform_functions import kinematic_rendezvous_hold_point_eci_polar_fn
from spacepinn.opengoddard.rendezvous_hold_point_eci_goddard import (
    kinematic_rendezvous_hold_point_eci_goddard,
)
from spacepinn.paper.baseline import (
    PAPER_BASELINE_MAX_ITERATION,
    capture_baseline_entry,
    print_baseline_delta_v_summary,
)
from spacepinn.paper.monte_carlo import (
    add_single_mc_arguments,
    label_with_seed,
    persist_paper_monte_carlo_aggregate_summary,
    plot_single_group_boxplots,
    representative_entries,
    resolve_mode,
    seed_sequence,
    single_group_key,
)
from spacepinn.paper.plots.rendezvous import plot_results
from spacepinn.paper.runtime import smoke_mode_enabled
from spacepinn.paper.suite import run_entry_collection
from spacepinn.problems.rendezvous_hold_point_eci import (
    DEFAULT_T_FINAL_SECONDS,
    build_scenario,
    target_state_eci,
)
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.runner import execute_single_experiment, print_collection_run_summary

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "rendezvous_hold_point_eci"
FIG_PREFIX = "rendezvous"
PINN_LABEL = "PINN with exact BC"
BASELINE_LABEL = "OpenGoddard"
WARMSTART_BASELINE_LABEL = "OpenGoddard (PINN initial guess)"
PINN_COLOR = "#2ca02c"
BASELINE_COLOR = "#4d4d4d"
WARMSTART_BASELINE_COLOR = "#1f77b4"
PAPER_N_ADAM = 100_000
PAPER_N_LBFGS = 0
PAPER_CONVERGENCE_THRESHOLD = 1e-7
BASELINE_MAX_ITERATION = PAPER_BASELINE_MAX_ITERATION
BASELINE_FTOL = 1e-11
BASELINE_SLSQP_MAXITER = 25
REPRESENTATIVE_SEED = 9058
MC_SEED_START = 9000
MC_NUM_SEEDS = 100


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


def monte_carlo_group_key(entry: dict) -> str | None:
    return single_group_key(entry, base_label=PINN_LABEL)


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

    collection_entries: list[dict] = []
    for config in configs:
        pinn_model, pinn_result = execute_single_experiment(config)
        _sync_dynamic_terminal_reference(pinn_result, scenario=config["scenario"])
        plotting = dict(config.get("plotting", {}))
        collection_entries.append(
            {
                "label": config["label"],
                "source": "pinn",
                "result": pinn_result,
                "config": config,
                "model": pinn_model,
                "plotting": plotting,
                **plotting,
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
    _sync_dynamic_terminal_reference(cold_entry["result"], scenario=scenario)
    collection_entries.append(cold_entry)

    warm_start_result = _select_representative_result(
        collection_entries,
        representative_seed=None,
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
    collection_entries.append(warm_entry)

    collection_label = f"{COLLECTION_LABEL}_monte_carlo" if mode == "mc" else COLLECTION_LABEL
    collection_run = run_entry_collection(
        entries=collection_entries,
        label=collection_label,
        run_root=RUN_ROOT,
    )
    collection_run["scenario"] = scenario
    return collection_run


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

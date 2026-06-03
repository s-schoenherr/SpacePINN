from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import spacepinn
from spacepinn.config.config_orbit_transfer import (
    circular_ot_kinematic_polar_config,
)
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
from spacepinn.paper.plots.orbit_transfer import plot_fixed_orbit_transfer
from spacepinn.paper.runtime import smoke_mode_enabled
from spacepinn.paper.suite import ExperimentSuite, run_experiment_suite
from spacepinn.opengoddard.circular_orbit_transfer_goddard import (
    kinematic_ot_goddard,
)
from spacepinn.plotting.style import PALETTE

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "orbit_transfer_fixed_angle"
FIG_PREFIX = "fixed_terminal_angle"
BASELINE_LABEL = "Baseline (OpenGoddard)"
KINEMATIC_LABEL = "PINN with exact BC"
DIRECT_COLLOCATION_COLOR = PALETTE["opengoddard"]
PAPER_N_ADAM = 10_000
PAPER_N_LBFGS = 0
PAPER_CONVERGENCE_THRESHOLD = 1e-5
OPENGODDARD_MAX_ITERATION = PAPER_BASELINE_MAX_ITERATION
REPRESENTATIVE_SEED = 4047
MC_SEED_START = 4000
MC_NUM_SEEDS = 100


def _parse_args():
    parser = argparse.ArgumentParser(description="Paper circular orbit transfer with fixed terminal angle.")
    add_single_mc_arguments(parser, default_mode="single")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def build_config(*, seed: int | None = None, label_seed: bool = False, smoke: bool | None = None) -> dict:
    config = deepcopy(circular_ot_kinematic_polar_config)
    if seed is not None:
        config["seed"] = int(seed)
    config["label"] = label_with_seed(KINEMATIC_LABEL, seed) if label_seed and seed is not None else KINEMATIC_LABEL
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


def monte_carlo_group_key(entry: dict) -> str | None:
    return single_group_key(entry, base_label=KINEMATIC_LABEL)


def _baseline_entry(*, smoke: bool | None = None) -> dict:
    return capture_baseline_entry(
        lambda: build_baseline_entry(smoke=smoke),
        log_filename="baseline_opengoddard.log",
    )


def _baseline_entries(*, smoke: bool | None = None) -> list[dict]:
    return [_baseline_entry(smoke=smoke)]


SUITE = ExperimentSuite(
    label=COLLECTION_LABEL,
    run_root=RUN_ROOT,
    representative_seed=REPRESENTATIVE_SEED,
    mc_seed_start=MC_SEED_START,
    mc_num_seeds=MC_NUM_SEEDS,
    build_config=build_config,
    build_baseline_entries=_baseline_entries,
    plot_representative=lambda entries, output_dir: plot_fixed_orbit_transfer(entries, output_dir=output_dir),
    group_key=monte_carlo_group_key,
    base_label=KINEMATIC_LABEL,
    baseline_labels=(BASELINE_LABEL,),
    fig_prefix=FIG_PREFIX,
)


def main(
    *,
    mode: str = "single",
    skip_plots: bool = False,
    print_summary: bool = True,
    smoke: bool | None = None,
    representative_seed: int | None = None,
    seed_start: int | None = None,
    num_seeds: int | None = None,
):
    return run_experiment_suite(
        SUITE,
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
        representative_seed=args.representative_seed,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
    )

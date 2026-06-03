from __future__ import annotations

import argparse
from copy import deepcopy
import functools
from functools import partial
import inspect
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

import spacepinn
from spacepinn.config.config_3d import geometric_3d_config, kinematic_3d_config, ordinary_3d_config
from spacepinn.config.shared_parameters import x0_3d, xN_3d
from spacepinn.config.transform_functions import kinematic_fn
from spacepinn.experiment import build_pretrained_model
from spacepinn.opengoddard.geometric_3d_goddard import geometric_3d_opengoddard
from spacepinn.paper.baseline import (
    capture_baseline_entry,
    get_baseline_entries,
    paper_baseline_solver_kwargs,
    print_baseline_delta_v_summary,
)
from spacepinn.paper.monte_carlo import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper.plots.swingby_3d import (
    plot_loss_figure,
    plot_monte_carlo_boxplots_paper,
    plot_monte_carlo_gravity_paper,
    plot_monte_carlo_thrust_paper,
    plot_monte_carlo_traj_2d_paper,
    plot_monte_carlo_traj_3d_paper,
)
from spacepinn.paper.runtime import smoke_mode_enabled
from spacepinn.paper.suite import run_entry_collection
from spacepinn.pretraining.kinematic_to_geometric_pretraining_3d import (
    PLANE_VELOCITY,
)
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotting.style import PALETTE
from spacepinn.runner import load_run, print_collection_run_summary
from spacepinn.runner.context import RunCollectionContext
from spacepinn.runner.execution import execute_single_experiment
from spacepinn.runner.runtime import _prepare_runtime_config

DTYPE = "float32"
RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_3d"
MC_COLLECTION_LABEL = f"{COLLECTION_LABEL}_monte_carlo"
FIG_PREFIX = "swingby_3d"
BASELINE_LABEL = "Baseline (OpenGoddard)"
GEOMETRIC_LABEL = "PINN with exact BC"
ORDINARY_LABEL = "PINN with soft BC"
LEGACY_ORDINARY_LABEL = "PINN without exact BC"
PRETRAINED_LABEL = "PINN with exact BC and pre-conditioning"
NUM_SEEDS = 100
SEEDS = [2000 + index for index in range(NUM_SEEDS)]
SMOKE_NUM_SEEDS = 2
REPRESENTATIVE_SEEDS = {
    GEOMETRIC_LABEL: 2076,
    ORDINARY_LABEL: 2040,
    PRETRAINED_LABEL: 2092,
}
COLORS = {
    GEOMETRIC_LABEL: PALETTE["position"],
    ORDINARY_LABEL: PALETTE["vanilla"],
    PRETRAINED_LABEL: PALETTE["kinematic"],
}
ORDINARY_LAMBDA_BC = 0.42133217438472903


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper swingby 3D experiment.")
    parser.add_argument("--mode", choices=("single", "mc"), default="single")
    parser.add_argument("--mc", action="store_true", help="Shortcut for --mode mc.")
    parser.add_argument(
        "--from-run",
        default=None,
        help="Optional saved collection run to replot without retraining.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write plots to. Defaults to the run's artifacts/plots directory.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Run the collection without plotting.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Suppress the printed collection summary.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Optional multiprocessing worker count for the seed runs.",
    )
    return parser.parse_args()


def get_seeds(*, smoke: bool | None = None) -> list[int]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    return SEEDS[:SMOKE_NUM_SEEDS] if smoke_enabled else SEEDS


def build_configs(*, smoke: bool | None = None) -> list[dict]:
    configs: list[dict] = []
    for seed in get_seeds(smoke=smoke):
        configs.extend(
            [
                _build_geometric_config(seed, smoke=smoke),
                _build_ordinary_config(seed, smoke=smoke),
                _build_kinematic_pretrain_config(seed, smoke=smoke),
            ]
        )
    return configs


def _build_geometric_config(seed: int, *, smoke: bool | None = None) -> dict:
    config = deepcopy(geometric_3d_config)
    config["label"] = f"{GEOMETRIC_LABEL} | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    config["optimizer"]["n_adam"] = 1 if (smoke_mode_enabled() if smoke is None else smoke) else 2_000
    if smoke_mode_enabled() if smoke is None else smoke:
        config["optimizer"]["n_lbfgs"] = 0
    return config


def _build_ordinary_config(seed: int, *, smoke: bool | None = None) -> dict:
    config = deepcopy(ordinary_3d_config)
    config["label"] = f"{ORDINARY_LABEL} | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    config["optimizer"]["w_bc"] = ORDINARY_LAMBDA_BC
    config["optimizer"]["n_adam"] = 1 if (smoke_mode_enabled() if smoke is None else smoke) else 2_000
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    if smoke_mode_enabled() if smoke is None else smoke:
        config["optimizer"]["n_lbfgs"] = 0
    return config


def _build_kinematic_pretrain_config(seed: int, *, smoke: bool | None = None) -> dict:
    config = deepcopy(kinematic_3d_config)
    config["label"] = f"Kinematic pretrain | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    config["optimizer"]["n_adam"] = 1 if (smoke_mode_enabled() if smoke is None else smoke) else 2_000
    config["optimizer"]["n_lbfgs"] = 0
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["pinn"]["output_transform_fn"] = partial(
        kinematic_fn,
        x0=x0_3d,
        xN=xN_3d,
        v0=PLANE_VELOCITY,
        vN=PLANE_VELOCITY,
    )
    return config


def _build_pretrained_finetune_config(seed: int, *, initial_t_total: float, smoke: bool | None = None) -> dict:
    config = deepcopy(geometric_3d_config)
    config["label"] = f"{PRETRAINED_LABEL} | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    config["optimizer"]["n_adam"] = 1 if (smoke_mode_enabled() if smoke is None else smoke) else 2_000
    config["extra_parameters"]["t_total"] = torch.nn.Parameter(
        torch.tensor(float(initial_t_total), requires_grad=True)
    )
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["plotting"]["color"] = PALETTE["kinematic"]
    if smoke_mode_enabled() if smoke is None else smoke:
        config["optimizer"]["n_lbfgs"] = 0
    return config


def group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    grouped = {GEOMETRIC_LABEL: [], ORDINARY_LABEL: [], PRETRAINED_LABEL: []}
    for entry in entries:
        label = str(entry.get("label", ""))
        canonical = _canonical_group_label(label)
        if canonical is not None:
            grouped[canonical].append(entry)
    return grouped


def build_baseline_entry(*, smoke: bool | None = None) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    baseline_entry = geometric_3d_opengoddard(
        BASELINE_LABEL,
        **paper_baseline_solver_kwargs(smoke_enabled=smoke_enabled),
    )
    baseline_entry["source"] = "opengoddard"
    return baseline_entry


def _baseline_entries(*, smoke: bool | None = None) -> list[dict]:
    return [
        capture_baseline_entry(
            lambda: build_baseline_entry(smoke=smoke),
            log_filename="baseline_opengoddard.log",
        )
    ]


def get_baseline_entry(collection_run: dict) -> dict:
    try:
        baselines = get_baseline_entries(collection_run.get("entries", []), baseline_labels=(BASELINE_LABEL,))
    except ValueError:
        return build_baseline_entry()
    return baselines[0]


def _canonical_group_label(label: str) -> str | None:
    if label.startswith(PRETRAINED_LABEL):
        return PRETRAINED_LABEL
    if label.startswith(GEOMETRIC_LABEL):
        return GEOMETRIC_LABEL
    if label.startswith(ORDINARY_LABEL) or label.startswith(LEGACY_ORDINARY_LABEL):
        return ORDINARY_LABEL
    return None


def select_median_entry(entries: list[dict]) -> dict:
    sorted_entries = sorted(entries, key=lambda entry: float(entry["result"].delta_v))
    return sorted_entries[len(sorted_entries) // 2]


def select_best_entry(entries: list[dict]) -> dict:
    return min(entries, key=lambda entry: float(entry["result"].delta_v))


def _representative_entries(collection_run: dict) -> list[dict]:
    grouped_entries = group_entries(collection_run["entries"])
    representative_entries: list[dict] = []
    for group_name in (GEOMETRIC_LABEL, ORDINARY_LABEL, PRETRAINED_LABEL):
        entries = grouped_entries.get(group_name, [])
        if not entries:
            continue
        best_entry = select_best_entry(entries)
        plotting = dict(best_entry.get("plotting", {}))
        representative_entries.append(
            {
                "label": group_name,
                "result": best_entry["result"],
                "model": None,
                "config": best_entry.get("config"),
                "source": best_entry.get("source", "pinn"),
                "color": plotting.get("color", COLORS[group_name]),
                "linestyle": plotting.get("linestyle", "solid"),
                "trajectory_linestyle": plotting.get("trajectory_linestyle", plotting.get("linestyle", "solid")),
                "quiver_scale": plotting.get("quiver_scale", 20),
                "quiver_count": plotting.get("quiver_count"),
            }
        )

    baseline_entry = get_baseline_entry(collection_run)
    baseline_plotting = dict(baseline_entry.get("plotting", {}))
    representative_entries.append(
        {
            "label": BASELINE_LABEL,
            "result": baseline_entry["result"],
            "model": None,
            "config": baseline_entry.get("config"),
            "source": baseline_entry.get("source", "opengoddard"),
            "color": baseline_plotting.get("color", PALETTE["opengoddard"]),
            "linestyle": baseline_plotting.get("linestyle", "solid"),
            "trajectory_linestyle": "dashdot",
            "quiver_scale": baseline_plotting.get("quiver_scale", 20),
            "quiver_count": baseline_plotting.get("quiver_count"),
        }
    )
    return representative_entries


def monte_carlo_group_key(entry: dict) -> str | None:
    return _canonical_group_label(str(entry.get("label", "")))


def _make_config_multiprocessing_safe(value):
    if isinstance(value, torch.nn.Parameter):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.numpy().copy()
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.ndim == 0 else detached.numpy().copy()
    if isinstance(value, dict):
        return {key: _make_config_multiprocessing_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_make_config_multiprocessing_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_make_config_multiprocessing_safe(item) for item in value)
    if isinstance(value, functools.partial):
        return functools.partial(
            value.func,
            *(_make_config_multiprocessing_safe(arg) for arg in value.args),
            **{key: _make_config_multiprocessing_safe(val) for key, val in (value.keywords or {}).items()},
        )
    return value


def _execute_config(config: dict, *, model=None) -> tuple[dict, object, object]:
    config_runtime = _prepare_runtime_config(deepcopy(config))
    model, result = execute_single_experiment(config_runtime, model=model)
    return config_runtime, model, result


def _prepend_pretraining_history(*, pretrain_result, finetune_result) -> None:
    finetune_result.history.loss = [*pretrain_result.loss, *finetune_result.loss]
    finetune_result.history.loss_physics = [*pretrain_result.loss_physics, *finetune_result.loss_physics]
    finetune_result.history.loss_bc = [*pretrain_result.loss_bc, *finetune_result.loss_bc]
    finetune_result._sync_legacy_attributes()


def _entry_payload(*, label: str, result, config: dict, source: str = "pinn") -> dict:
    return {
        "label": label,
        "result": result,
        "config": _make_config_multiprocessing_safe(config),
        "plotting": dict(config.get("plotting", {})),
        "source": source,
    }


def _add_collection_entry(collection_context, **kwargs):
    supported = inspect.signature(collection_context.add_entry).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return collection_context.add_entry(**filtered_kwargs)


def _run_seed(seed: int, smoke: bool = False) -> list[dict]:
    exact_config = _build_geometric_config(seed, smoke=smoke)
    soft_config = _build_ordinary_config(seed, smoke=smoke)
    pretrain_config = _build_kinematic_pretrain_config(seed, smoke=smoke)

    exact_runtime, _exact_model, exact_result = _execute_config(exact_config)
    soft_runtime, _soft_model, soft_result = _execute_config(soft_config)
    pretrain_runtime, pretrain_model, pretrain_result = _execute_config(pretrain_config)

    finetune_config = _build_pretrained_finetune_config(seed, initial_t_total=float(pretrain_result.t_total), smoke=smoke)
    finetune_runtime = _prepare_runtime_config(deepcopy(finetune_config))
    finetune_model = build_pretrained_model(finetune_runtime, pretrain_model)
    finetune_model, finetune_result = execute_single_experiment(finetune_runtime, model=finetune_model)
    _prepend_pretraining_history(pretrain_result=pretrain_result, finetune_result=finetune_result)

    return [
        _entry_payload(label=exact_runtime["label"], result=exact_result, config=exact_runtime),
        _entry_payload(label=soft_runtime["label"], result=soft_result, config=soft_runtime),
        _entry_payload(label=finetune_runtime["label"], result=finetune_result, config=finetune_runtime),
    ]


def plot_collection_run(
    collection_run: dict,
    *,
    output_dir: str | Path | None = None,
    include_boxplots: bool = True,
) -> dict[str, list[dict]]:
    target_dir = Path(output_dir) if output_dir is not None else Path(collection_run["plot_output_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    grouped_entries = group_entries(collection_run["entries"])
    baseline_entry = get_baseline_entry(collection_run)
    representative_entries = _representative_entries(collection_run)
    summary_entries = list(collection_run["entries"])
    if baseline_entry not in summary_entries:
        summary_entries.append(baseline_entry)
    print_monte_carlo_summary(grouped_entries, title="geometric vs ordinary vs pre-trained 3D Monte Carlo")
    print_baseline_delta_v_summary(
        summary_entries,
        title=COLLECTION_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        group_key=monte_carlo_group_key,
        include_variance=True,
    )

    plot_monte_carlo_traj_2d_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_traj2d",
        baseline_entry=baseline_entry,
    )
    plot_monte_carlo_traj_3d_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_traj3d",
        baseline_entry=baseline_entry,
    )

    if include_boxplots:
        plot_monte_carlo_boxplots_paper(
            grouped_entries,
            colors=COLORS,
            output_dir=target_dir,
            fig_name=f"{FIG_PREFIX}_boxplots",
            baseline_entry=baseline_entry,
        )
    plot_loss_figure(representative_entries, output_dir=target_dir)
    plot_monte_carlo_thrust_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_thrust",
        baseline_entry=baseline_entry,
    )
    plot_monte_carlo_gravity_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_gravity",
        baseline_entry=baseline_entry,
    )
    return grouped_entries


def replot_saved_run(run_dir: str | Path, *, output_dir: str | Path | None = None, print_summary: bool = True) -> dict:
    collection_run = load_run(run_dir)
    collection_run["plot_output_dir"] = str(Path(output_dir) if output_dir is not None else Path(collection_run["run_dir"]) / "artifacts" / "plots")
    persist_paper_monte_carlo_aggregate_summary(
        collection_run,
        title=COLLECTION_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        group_key=monte_carlo_group_key,
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    plot_collection_run(collection_run, output_dir=output_dir)
    return collection_run


def run_collection(*, smoke: bool | None = None, workers: int = 1, label: str = MC_COLLECTION_LABEL) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    seeds = get_seeds(smoke=smoke_enabled)

    if workers <= 1:
        entries = []
        for seed in seeds:
            entries.extend(_run_seed(seed, smoke=smoke_enabled))
        return _finalize_collection_entries(entries, smoke=smoke_enabled, label=label)

    collection_context = RunCollectionContext(label=label, run_root=str(RUN_ROOT))
    collection_context.start()
    collection_results = []

    try:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            completed_by_seed = pool.starmap(_run_seed, [(seed, smoke_enabled) for seed in seeds])

        for seed_entries in completed_by_seed:
            for entry in seed_entries:
                _add_collection_entry(
                    collection_context,
                    label=entry["label"],
                    result=entry["result"],
                    config=entry["config"],
                    model=None,
                    source=entry["source"],
                )
                collection_results.append(
                    {
                        "label": entry["label"],
                        "source": entry["source"],
                        "result": entry["result"],
                        **entry["plotting"],
                        "model": None,
                        "run_id": collection_context.run_id,
                        "run_dir": str(collection_context.run_dir),
                        "plot_output_dir": str(collection_context.plot_dir),
                        "summary_path": str(collection_context.summary_path),
                    }
                )

        baseline_entry = _baseline_entries(smoke=smoke_enabled)[0]
        _add_collection_entry(
            collection_context,
            label=baseline_entry["label"],
            result=baseline_entry["result"],
            config=baseline_entry.get("config"),
            model=None,
            source=baseline_entry["source"],
            log_text=baseline_entry.get("log_text"),
            log_filename=baseline_entry.get("log_filename"),
        )
        collection_results.append(
            {
                "label": baseline_entry["label"],
                "source": baseline_entry["source"],
                "result": baseline_entry["result"],
                "model": None,
                "run_id": collection_context.run_id,
                "run_dir": str(collection_context.run_dir),
                "plot_output_dir": str(collection_context.plot_dir),
                "summary_path": str(collection_context.summary_path),
            }
        )

        collection_context.finalize_success()

        return {
            "label": label,
            "entries": collection_results,
            "run_id": collection_context.run_id,
            "run_dir": str(collection_context.run_dir),
            "plot_output_dir": str(collection_context.plot_dir),
            "summary_path": str(collection_context.summary_path),
            "manifest_path": str(collection_context.manifest_path),
            "config_path": str(collection_context.config_path),
        }
    except Exception as error:
        collection_context.finalize_failure(error)
        raise


def _finalize_collection_entries(entries: list[dict], *, smoke: bool | None = None, label: str = MC_COLLECTION_LABEL) -> dict:
    return run_entry_collection(
        entries=entries,
        label=label,
        run_root=RUN_ROOT,
        baseline_entries=_baseline_entries(smoke=smoke),
    )


def _run_representative_entries(*, smoke: bool | None = None) -> list[dict]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    entries: list[dict] = []

    exact_config = _build_geometric_config(REPRESENTATIVE_SEEDS[GEOMETRIC_LABEL], smoke=smoke_enabled)
    exact_runtime, _exact_model, exact_result = _execute_config(exact_config)
    entries.append(_entry_payload(label=GEOMETRIC_LABEL, result=exact_result, config=exact_runtime))

    soft_config = _build_ordinary_config(REPRESENTATIVE_SEEDS[ORDINARY_LABEL], smoke=smoke_enabled)
    soft_runtime, _soft_model, soft_result = _execute_config(soft_config)
    entries.append(_entry_payload(label=ORDINARY_LABEL, result=soft_result, config=soft_runtime))

    pretrain_seed = REPRESENTATIVE_SEEDS[PRETRAINED_LABEL]
    pretrain_config = _build_kinematic_pretrain_config(pretrain_seed, smoke=smoke_enabled)
    pretrain_runtime, pretrain_model, pretrain_result = _execute_config(pretrain_config)
    finetune_config = _build_pretrained_finetune_config(
        pretrain_seed,
        initial_t_total=float(pretrain_result.t_total),
        smoke=smoke_enabled,
    )
    finetune_runtime = _prepare_runtime_config(deepcopy(finetune_config))
    finetune_model = build_pretrained_model(finetune_runtime, pretrain_model)
    _finetune_model, finetune_result = execute_single_experiment(finetune_runtime, model=finetune_model)
    _prepend_pretraining_history(pretrain_result=pretrain_result, finetune_result=finetune_result)
    entries.append(_entry_payload(label=PRETRAINED_LABEL, result=finetune_result, config=finetune_runtime))

    return entries


def main(
    *,
    mode: str = "single",
    skip_plots: bool = False,
    print_summary: bool = True,
    smoke: bool | None = None,
    from_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    workers: int = 1,
):
    if from_run is not None:
        return replot_saved_run(from_run, output_dir=output_dir, print_summary=print_summary)

    if mode == "mc":
        collection_run = run_collection(smoke=smoke, workers=workers, label=MC_COLLECTION_LABEL)
        persist_paper_monte_carlo_aggregate_summary(
            collection_run,
            title=MC_COLLECTION_LABEL,
            baseline_labels=(BASELINE_LABEL,),
            group_key=monte_carlo_group_key,
        )
    else:
        collection_run = _finalize_collection_entries(
            _run_representative_entries(smoke=smoke),
            smoke=smoke,
            label=COLLECTION_LABEL,
        )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        plot_collection_run(collection_run, output_dir=output_dir, include_boxplots=mode == "mc")

    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        mode="mc" if args.mc else args.mode,
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        from_run=args.from_run,
        output_dir=args.output_dir,
        workers=args.workers,
    )

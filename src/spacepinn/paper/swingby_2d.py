from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import multiprocessing as mp
from pathlib import Path

import numpy as np

import spacepinn
from spacepinn.config.config_2d import geometric_2d_config, ordinary_2d_config
from spacepinn.paper.baseline import (
    capture_baseline_entry,
    get_baseline_entries,
    paper_baseline_solver_kwargs,
    print_baseline_delta_v_summary,
)
from spacepinn.paper.monte_carlo import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper.plots.swingby_2d import (
    plot_gravity_figure,
    plot_loss_figure,
    plot_monte_carlo_boxplots_paper,
    plot_thrust_figure,
    plot_traj_figure,
)
from spacepinn.paper.runtime import smoke_mode_enabled
from spacepinn.paper.suite import run_entry_collection
from spacepinn.opengoddard.geometric_2d_goddard import geometric_2d_opengoddard
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotting.style import PALETTE
from spacepinn.runner import load_run, print_collection_run_summary
from spacepinn.runner.context import RunCollectionContext
from spacepinn.runner.execution import execute_single_experiment
from spacepinn.runner.runtime import _prepare_runtime_config

DTYPE = "float32"
RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_2d"
MC_COLLECTION_LABEL = f"{COLLECTION_LABEL}_monte_carlo"
FIG_PREFIX = "swingby_2d"
GEOMETRIC_LABEL = "PINN with exact BC"
ORDINARY_LABEL = "PINN with soft BC"
NUM_SEEDS = 100
SEEDS = [1000 + index for index in range(NUM_SEEDS)]
SMOKE_NUM_SEEDS = 2
REPRESENTATIVE_SEEDS = {
    GEOMETRIC_LABEL: 1079,
    ORDINARY_LABEL: 1056,
}
QUIVER_COUNT = 10
NOMINAL_BRANCH_PATH_LENGTH_FACTOR = 1.05
COLORS = {
    GEOMETRIC_LABEL: PALETTE["position"],
    ORDINARY_LABEL: PALETTE["vanilla"],
}
BASELINE_LABEL = "Baseline (OpenGoddard)"
ORDINARY_LAMBDA_BC = 0.32397426295281967


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper swingby 2D experiment.")
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
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    seeds = get_seeds(smoke=smoke)

    configs: list[dict] = []
    for seed in seeds:
        geometric_config = _build_geometric_config(seed)
        ordinary_config = _build_ordinary_config(seed)

        configs.extend([geometric_config, ordinary_config])

    if smoke_enabled:
        for config in configs:
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0

    return configs


def _build_geometric_config(seed: int) -> dict:
    config = deepcopy(geometric_2d_config)
    config["label"] = f"{GEOMETRIC_LABEL} | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    config["plotting"]["color"] = COLORS[GEOMETRIC_LABEL]
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["plotting"]["quiver_count"] = QUIVER_COUNT
    return config


def _build_ordinary_config(seed: int) -> dict:
    config = deepcopy(ordinary_2d_config)
    config["label"] = f"{ORDINARY_LABEL} | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    config["optimizer"]["w_bc"] = ORDINARY_LAMBDA_BC
    config["plotting"]["color"] = COLORS[ORDINARY_LABEL]
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
    config["plotting"]["quiver_count"] = QUIVER_COUNT
    return config


def group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    grouped = {GEOMETRIC_LABEL: [], ORDINARY_LABEL: []}
    for entry in entries:
        label = entry.get("label", "")
        if label.startswith(GEOMETRIC_LABEL):
            grouped[GEOMETRIC_LABEL].append(entry)
        elif label.startswith(ORDINARY_LABEL):
            grouped[ORDINARY_LABEL].append(entry)
    return grouped


def build_baseline_entry(*, smoke: bool | None = None) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    baseline_entry = geometric_2d_opengoddard(
        BASELINE_LABEL,
        **paper_baseline_solver_kwargs(smoke_enabled=smoke_enabled),
    )
    baseline_entry["source"] = "opengoddard"
    baseline_entry["plotting"] = dict(baseline_entry.get("plotting", {}))
    baseline_entry["plotting"]["color"] = PALETTE["opengoddard"]
    baseline_entry["plotting"]["linestyle"] = "solid"
    baseline_entry["plotting"]["trajectory_linestyle"] = "dashed"
    baseline_entry["plotting"]["quiver_count"] = QUIVER_COUNT
    baseline_entry["color"] = PALETTE["opengoddard"]
    baseline_entry["linestyle"] = "solid"
    baseline_entry["trajectory_linestyle"] = "dashed"
    baseline_entry["quiver_count"] = QUIVER_COUNT
    return baseline_entry


def _baseline_entries(*, smoke: bool | None = None) -> list[dict]:
    return [
        capture_baseline_entry(
            lambda: build_baseline_entry(smoke=smoke),
            log_filename="baseline_opengoddard.log",
        )
    ]


def is_baseline_entry(entry: dict) -> bool:
    return entry.get("source") == "opengoddard" or entry.get("label") == BASELINE_LABEL


def get_baseline_entry(collection_run: dict) -> dict:
    entries = collection_run.get("entries", [])
    try:
        baselines = get_baseline_entries(entries, baseline_labels=(BASELINE_LABEL,))
    except ValueError:
        return build_baseline_entry()
    return baselines[0]


def select_median_entry(entries: list[dict]) -> dict:
    sorted_entries = sorted(entries, key=lambda entry: float(entry["result"].delta_v))
    return sorted_entries[len(sorted_entries) // 2]


def select_best_entry(entries: list[dict]) -> dict:
    return min(entries, key=lambda entry: float(entry["result"].delta_v))


def trajectory_path_length(entry: dict) -> float:
    r = np.asarray(entry["result"].r, dtype=float)
    if len(r) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(r, axis=0), axis=1)))


def select_best_nominal_branch_entry(entries: list[dict], *, reference_entry: dict | None = None) -> dict:
    if reference_entry is None:
        return select_best_entry(entries)

    reference_length = trajectory_path_length(reference_entry)
    if reference_length <= 0:
        return select_best_entry(entries)

    max_path_length = NOMINAL_BRANCH_PATH_LENGTH_FACTOR * reference_length
    nominal_entries = [entry for entry in entries if trajectory_path_length(entry) <= max_path_length]
    if not nominal_entries:
        return select_best_entry(entries)

    return select_best_entry(nominal_entries)


def _execute_config(config: dict) -> tuple[dict, object, object]:
    config_runtime = _prepare_runtime_config(deepcopy(config))
    model, result = execute_single_experiment(config_runtime)
    return config_runtime, model, result


def _entry_payload(*, label: str, result, config: dict, source: str = "pinn") -> dict:
    return {
        "label": label,
        "result": result,
        "config": config,
        "plotting": dict(config.get("plotting", {})),
        "source": source,
    }


def _add_collection_entry(collection_context, **kwargs):
    supported = inspect.signature(collection_context.add_entry).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return collection_context.add_entry(**filtered_kwargs)


def _run_seed(seed: int, smoke: bool = False) -> list[dict]:
    geometric_config = _build_geometric_config(seed)
    ordinary_config = _build_ordinary_config(seed)

    if smoke:
        geometric_config["optimizer"]["n_adam"] = 1
        geometric_config["optimizer"]["n_lbfgs"] = 0
        ordinary_config["optimizer"]["n_adam"] = 1
        ordinary_config["optimizer"]["n_lbfgs"] = 0

    geometric_runtime, _geometric_model, geometric_result = _execute_config(geometric_config)
    ordinary_runtime, _ordinary_model, ordinary_result = _execute_config(ordinary_config)

    return [
        _entry_payload(label=geometric_runtime["label"], result=geometric_result, config=geometric_runtime),
        _entry_payload(label=ordinary_runtime["label"], result=ordinary_result, config=ordinary_runtime),
    ]


def monte_carlo_group_key(entry: dict) -> str | None:
    label = str(entry.get("label", ""))
    if label.startswith(GEOMETRIC_LABEL):
        return GEOMETRIC_LABEL
    if label.startswith(ORDINARY_LABEL):
        return ORDINARY_LABEL
    return None


def _representative_entries(collection_run: dict, *, include_baseline: bool = True) -> list[dict]:
    grouped_entries = group_entries(collection_run["entries"])
    baseline_entry = get_baseline_entry(collection_run)
    entries: list[dict] = []
    for group_name in (GEOMETRIC_LABEL, ORDINARY_LABEL):
        method_entries = grouped_entries.get(group_name, [])
        if not method_entries:
            continue
        best_entry = select_best_nominal_branch_entry(method_entries, reference_entry=baseline_entry)
        plotting = dict(best_entry.get("plotting", {}))
        plotting["color"] = COLORS[group_name]
        plotting["linestyle"] = "solid"
        plotting["trajectory_linestyle"] = "solid"
        plotting["quiver_count"] = QUIVER_COUNT
        entries.append(
            {
                "label": group_name,
                "result": best_entry["result"],
                "model": None,
                "config": best_entry.get("config"),
                "source": best_entry.get("source", "pinn"),
                "plotting": plotting,
                **plotting,
            }
        )
    if include_baseline:
        plotting = dict(baseline_entry.get("plotting", {}))
        plotting["color"] = PALETTE["opengoddard"]
        plotting["linestyle"] = "solid"
        plotting["trajectory_linestyle"] = "dashed"
        plotting["quiver_count"] = QUIVER_COUNT
        entries.append(
            {
                "label": BASELINE_LABEL,
                "result": baseline_entry["result"],
                "model": baseline_entry.get("model"),
                "config": baseline_entry.get("config"),
                "source": baseline_entry.get("source", "opengoddard"),
                "plotting": plotting,
                **plotting,
            }
        )
    return entries


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
    summary_entries = list(collection_run["entries"])
    if baseline_entry not in summary_entries:
        summary_entries.append(baseline_entry)
    print_monte_carlo_summary(grouped_entries, title="geometric vs ordinary 2D Monte Carlo")
    print_baseline_delta_v_summary(
        summary_entries,
        title=COLLECTION_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        group_key=monte_carlo_group_key,
        include_variance=True,
    )

    representative_entries = _representative_entries(collection_run, include_baseline=True)
    plot_traj_figure(representative_entries, output_dir=target_dir)
    plot_loss_figure(representative_entries, output_dir=target_dir)
    if include_boxplots:
        plot_monte_carlo_boxplots_paper(
            grouped_entries,
            colors=COLORS,
            output_dir=target_dir,
            fig_name=f"{FIG_PREFIX}_boxplots",
            baseline_entry=baseline_entry,
        )
    plot_thrust_figure(representative_entries, output_dir=target_dir)
    plot_gravity_figure(representative_entries, output_dir=target_dir)
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


def _run_representative_entries(*, smoke: bool | None = None) -> list[dict]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    configs = [
        _build_geometric_config(REPRESENTATIVE_SEEDS[GEOMETRIC_LABEL]),
        _build_ordinary_config(REPRESENTATIVE_SEEDS[ORDINARY_LABEL]),
    ]
    entries: list[dict] = []
    for config in configs:
        if smoke_enabled:
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0
        runtime_config, _model, result = _execute_config(config)
        label = GEOMETRIC_LABEL if runtime_config["label"].startswith(GEOMETRIC_LABEL) else ORDINARY_LABEL
        entries.append(_entry_payload(label=label, result=result, config=runtime_config))
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
        collection_run = run_entry_collection(
            entries=_run_representative_entries(smoke=smoke),
            label=COLLECTION_LABEL,
            run_root=RUN_ROOT,
            baseline_entries=_baseline_entries(smoke=smoke),
        )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        plot_collection_run(collection_run, output_dir=output_dir, include_boxplots=mode == "mc")

    return collection_run


def run_collection(*, smoke: bool | None = None, workers: int = 1, label: str = MC_COLLECTION_LABEL) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    seeds = get_seeds(smoke=smoke_enabled)

    if workers <= 1:
        entries = []
        for seed in seeds:
            entries.extend(_run_seed(seed, smoke=smoke_enabled))
        return run_entry_collection(
            entries=entries,
            label=label,
            run_root=RUN_ROOT,
            baseline_entries=_baseline_entries(smoke=smoke_enabled),
        )

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

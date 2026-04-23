from __future__ import annotations

import argparse
import functools
from copy import deepcopy
import inspect
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spacepinn
import torch

from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._aggregate_summary import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper._baseline_summary import (
    get_baseline_entries,
    print_baseline_delta_v_summary,
)
from spacepinn.paper.rendezvous_hold_point_eci import (
    BASELINE_COLOR,
    BASELINE_LABEL,
    DEFAULT_T_FINAL_SECONDS,
    FIG_PREFIX as SINGLE_FIG_PREFIX,
    MAIN_FIGSIZE,
    PAPER_CONVERGENCE_THRESHOLD,
    PINN_COLOR,
    PINN_LABEL,
    WARMSTART_BASELINE_COLOR,
    WARMSTART_BASELINE_LABEL,
    build_config as build_single_config,
    plot_results,
)
from spacepinn.plotting.helpers import register_plot_artifact_if_possible
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.runner import load_run, print_collection_run_summary, run_experiment_collection
from spacepinn.runner.context import RunCollectionContext
from spacepinn.runner.execution import execute_single_experiment
from spacepinn.runner.runtime import _prepare_runtime_config

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "rendezvous_hold_point_eci_monte_carlo"
FIG_PREFIX = "rendezvous_hold_point_eci_monte_carlo"
NUM_SEEDS = 100
SEEDS = [9000 + index for index in range(NUM_SEEDS)]
SMOKE_NUM_SEEDS = 2
BOXPLOT_FIGSIZE = (17.2, 4.8)
COLORS = {PINN_LABEL: PINN_COLOR}
BASELINE_LABELS = (BASELINE_LABEL, WARMSTART_BASELINE_LABEL)
DEFAULT_BASELINE_RUN = RUN_ROOT / "2026" / "04" / "20260420_095242_rendezvous_hold_point_eci"
MONTE_CARLO_CONVERGENCE_THRESHOLD = PAPER_CONVERGENCE_THRESHOLD


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Monte Carlo for rendezvous hold-point transfer.")
    parser.add_argument("--t-final-seconds", type=float, default=DEFAULT_T_FINAL_SECONDS)
    parser.add_argument("--baseline-run", default=str(DEFAULT_BASELINE_RUN))
    parser.add_argument("--from-run", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def get_seeds(*, smoke: bool | None = None) -> list[int]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    return SEEDS[:SMOKE_NUM_SEEDS] if smoke_enabled else SEEDS


def _build_seed_config(
    seed: int,
    *,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    smoke: bool | None = None,
) -> dict:
    config = deepcopy(build_single_config(t_final_seconds=t_final_seconds, smoke=smoke))
    config["label"] = f"{PINN_LABEL} | seed={seed}"
    config["seed"] = seed
    config["optimizer"]["convergence_threshold"] = MONTE_CARLO_CONVERGENCE_THRESHOLD
    return config


def build_configs(
    *,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    smoke: bool | None = None,
) -> list[dict]:
    return [_build_seed_config(seed, t_final_seconds=t_final_seconds, smoke=smoke) for seed in get_seeds(smoke=smoke)]


def group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    grouped = {PINN_LABEL: []}
    for entry in entries:
        label = str(entry.get("label", ""))
        if label.startswith(PINN_LABEL):
            grouped[PINN_LABEL].append(entry)
    return grouped


def monte_carlo_group_key(entry: dict) -> str | None:
    label = str(entry.get("label", ""))
    if label.startswith(PINN_LABEL):
        return PINN_LABEL
    return None


def _make_config_multiprocessing_safe(value):
    if isinstance(value, torch.nn.Parameter):
        return value.detach().cpu().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
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


def _normalize_baseline_entry_style(entry: dict) -> dict:
    normalized = dict(entry)
    plotting = dict(normalized.get("plotting", {}))
    label = str(normalized.get("label", ""))
    if label == BASELINE_LABEL:
        plotting["color"] = BASELINE_COLOR
    elif label == WARMSTART_BASELINE_LABEL:
        plotting["color"] = WARMSTART_BASELINE_COLOR
    plotting.setdefault("linestyle", "solid")
    plotting.setdefault("trajectory_linestyle", "solid")
    plotting.setdefault("zorder", 2)
    normalized["plotting"] = plotting
    normalized["color"] = plotting["color"]
    normalized["linestyle"] = plotting["linestyle"]
    normalized["trajectory_linestyle"] = plotting["trajectory_linestyle"]
    normalized["zorder"] = plotting["zorder"]
    normalized["source"] = "opengoddard"
    return normalized


def load_reused_baseline_entries(run_dir: str | Path) -> list[dict]:
    loaded_run = load_run(run_dir)
    baseline_entries = get_baseline_entries(loaded_run.get("entries", []), baseline_labels=BASELINE_LABELS)
    reused_entries: list[dict] = []
    for baseline_entry in baseline_entries:
        reused_entry = {
            "label": baseline_entry["label"],
            "result": baseline_entry["result"],
            "model": None,
            "config": baseline_entry.get("config"),
            "plotting": dict(baseline_entry.get("plotting", {})),
            "source": baseline_entry.get("source", "opengoddard"),
        }
        log_path = baseline_entry.get("paths", {}).get("log_file")
        if log_path is not None:
            log_file = Path(str(log_path))
            if log_file.exists():
                reused_entry["log_text"] = log_file.read_text(encoding="utf-8")
                reused_entry["log_filename"] = log_file.name
        reused_entries.append(_normalize_baseline_entry_style(reused_entry))
    return reused_entries


def select_median_entry(entries: list[dict]) -> dict:
    sorted_entries = sorted(entries, key=lambda entry: float(entry["result"].delta_v))
    return sorted_entries[len(sorted_entries) // 2]


def _representative_entries(collection_run: dict, *, baseline_run: str | Path) -> list[dict]:
    grouped_entries = group_entries(collection_run["entries"])
    median_entry = select_median_entry(grouped_entries[PINN_LABEL])
    baselines = load_reused_baseline_entries(baseline_run)
    return [{**median_entry, "label": PINN_LABEL}] + baselines


def _resolve_scenario(
    collection_run: dict,
    *,
    baseline_run: str | Path,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
) -> dict:
    for entry in collection_run.get("entries", []):
        config = entry.get("config")
        if isinstance(config, dict) and "scenario" in config:
            return config["scenario"]

    for entry in load_reused_baseline_entries(baseline_run):
        config = entry.get("config")
        if isinstance(config, dict) and "scenario" in config:
            return config["scenario"]

    return build_single_config(t_final_seconds=t_final_seconds)["scenario"]


def plot_monte_carlo_boxplots_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entries: list[dict],
    figsize: tuple[float, float] = BOXPLOT_FIGSIZE,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    metric_specs = [
        ("delta_v", r"$\Delta v$"),
        ("t_total", "Time of Flight"),
        ("iterations_to_convergence", "Iterations to Convergence"),
    ]
    display_labels = list(colors.keys())

    for ax, (metric_name, ylabel) in zip(axes, metric_specs):
        values = []
        for group_name in colors:
            group_values = []
            for entry in grouped_entries[group_name]:
                result = entry["result"]
                if metric_name == "iterations_to_convergence":
                    group_values.append(len(result.loss))
                else:
                    group_values.append(getattr(result, metric_name))
            values.append(group_values)

        boxplot = ax.boxplot(values, patch_artist=True, tick_labels=display_labels)
        for patch, group_name in zip(boxplot["boxes"], colors):
            patch.set_facecolor(colors[group_name])
            patch.set_alpha(0.45)

        for index, group_name in enumerate(colors, start=1):
            group_values = np.array(values[index - 1], dtype=float)
            if group_values.size == 0:
                continue
            x_positions = np.full(group_values.shape, float(index), dtype=float)
            jitter = np.linspace(-0.08, 0.08, group_values.size) if group_values.size > 1 else np.array([0.0])
            ax.scatter(x_positions + jitter, group_values, color=colors[group_name], s=20, alpha=0.7)

        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=12)
        if metric_name == "delta_v":
            baseline_lines = [
                f"{entry['label']}: {float(entry['result'].delta_v):.4g}"
                for entry in baseline_entries
            ]
            ax.text(
                0.03,
                0.96,
                "\n".join(baseline_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "black",
                    "alpha": 0.95,
                    "boxstyle": "round,pad=0.25",
                },
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)


def plot_collection_run(collection_run: dict, *, baseline_run: str | Path, output_dir: str | Path | None = None) -> dict[str, list[dict]]:
    target_dir = Path(output_dir) if output_dir is not None else Path(collection_run["plot_output_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    grouped_entries = group_entries(collection_run["entries"])
    baseline_entries = load_reused_baseline_entries(baseline_run)
    summary_entries = list(collection_run["entries"])
    summary_entries.extend(entry for entry in baseline_entries if entry not in summary_entries)

    print_monte_carlo_summary(grouped_entries, title="Rendezvous hold-point Monte Carlo")
    print_baseline_delta_v_summary(
        summary_entries,
        title=COLLECTION_LABEL,
        baseline_labels=BASELINE_LABELS,
        group_key=monte_carlo_group_key,
        include_variance=True,
    )

    representative_entries = _representative_entries(collection_run, baseline_run=baseline_run)
    scenario = _resolve_scenario(collection_run, baseline_run=baseline_run)
    plot_results(representative_entries, output_dir=str(target_dir), scenario=scenario)
    plot_monte_carlo_boxplots_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_boxplots",
        baseline_entries=baseline_entries,
    )
    return grouped_entries


def replot_saved_run(
    run_dir: str | Path,
    *,
    baseline_run: str | Path = DEFAULT_BASELINE_RUN,
    output_dir: str | Path | None = None,
    print_summary: bool = True,
) -> dict:
    collection_run = load_run(run_dir)
    collection_run["plot_output_dir"] = str(
        Path(output_dir) if output_dir is not None else Path(collection_run["run_dir"]) / "artifacts" / "plots"
    )
    persist_paper_monte_carlo_aggregate_summary(
        collection_run,
        title=COLLECTION_LABEL,
        baseline_labels=BASELINE_LABELS,
        group_key=monte_carlo_group_key,
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    plot_collection_run(collection_run, baseline_run=baseline_run, output_dir=output_dir)
    return collection_run


def _run_seed(seed: int, t_final_seconds: float = DEFAULT_T_FINAL_SECONDS, smoke: bool = False) -> dict:
    config = _build_seed_config(seed, t_final_seconds=t_final_seconds, smoke=smoke)
    config_runtime = _prepare_runtime_config(deepcopy(config))
    model, result = execute_single_experiment(config_runtime)
    return _entry_payload(label=config_runtime["label"], result=result, config=_make_config_multiprocessing_safe(config_runtime))


def _finalize_collection_entries(entries: list[dict], *, baseline_run: str | Path) -> dict:
    baseline_entries = load_reused_baseline_entries(baseline_run)
    return run_experiment_collection(
        configs=[],
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
        additional_entries=[
            {
                "label": entry["label"],
                "result": entry["result"],
                "config": entry["config"],
                "model": None,
                "plotting": entry["plotting"],
                "source": entry["source"],
            }
            for entry in entries
        ]
        + baseline_entries,
    )


def run_collection(
    *,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    smoke: bool | None = None,
    workers: int = 1,
    baseline_run: str | Path = DEFAULT_BASELINE_RUN,
) -> dict:
    seeds = get_seeds(smoke=smoke)

    if workers <= 1:
        entries = [_run_seed(seed, t_final_seconds=t_final_seconds, smoke=bool(smoke)) for seed in seeds]
        return _finalize_collection_entries(entries, baseline_run=baseline_run)

    collection_context = RunCollectionContext(label=COLLECTION_LABEL, run_root=str(RUN_ROOT))
    collection_context.start()
    collection_results = []

    try:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            completed_entries = pool.starmap(
                _run_seed,
                [(seed, t_final_seconds, bool(smoke)) for seed in seeds],
            )

        for entry in completed_entries:
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

        baseline_entries = load_reused_baseline_entries(baseline_run)
        for baseline_entry in baseline_entries:
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
                    **baseline_entry.get("plotting", {}),
                    "model": None,
                    "run_id": collection_context.run_id,
                    "run_dir": str(collection_context.run_dir),
                    "plot_output_dir": str(collection_context.plot_dir),
                    "summary_path": str(collection_context.summary_path),
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
        }
    except Exception as error:
        collection_context.finalize_failure(error)
        raise


def main(
    *,
    skip_plots: bool = False,
    print_summary: bool = True,
    t_final_seconds: float = DEFAULT_T_FINAL_SECONDS,
    smoke: bool | None = None,
    from_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    workers: int = 1,
    baseline_run: str | Path = DEFAULT_BASELINE_RUN,
):
    if from_run is not None:
        return replot_saved_run(from_run, baseline_run=baseline_run, output_dir=output_dir, print_summary=print_summary)

    collection_run = run_collection(
        t_final_seconds=t_final_seconds,
        smoke=smoke,
        workers=workers,
        baseline_run=baseline_run,
    )
    persist_paper_monte_carlo_aggregate_summary(
        collection_run,
        title=COLLECTION_LABEL,
        baseline_labels=BASELINE_LABELS,
        group_key=monte_carlo_group_key,
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        plot_collection_run(collection_run, baseline_run=baseline_run, output_dir=output_dir)

    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        t_final_seconds=args.t_final_seconds,
        baseline_run=args.baseline_run,
        from_run=args.from_run,
        output_dir=args.output_dir,
        workers=args.workers,
    )

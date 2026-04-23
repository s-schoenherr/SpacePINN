from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import spacepinn
from spacepinn.config.config_2d import exact_bc_2d_config, soft_bc_2d_config
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._aggregate_summary import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import paper_baseline_solver_kwargs
from spacepinn.paper._baseline_summary import (
    get_baseline_entries,
    print_baseline_delta_v_summary,
)
from spacepinn.opengoddard.geometric_2d_goddard import geometric_2d_opengoddard
from spacepinn.plotting.helpers import (
    get_gravity_sources,
    plot_masses_2d,
    register_plot_artifact_if_possible,
)
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotting.style import PALETTE
from spacepinn.runner import load_run, print_collection_run_summary, run_experiment_collection
from spacepinn.runner.context import RunCollectionContext
from spacepinn.runner.execution import execute_single_experiment
from spacepinn.runner.runtime import _prepare_runtime_config

DTYPE = "float32"
RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_2d_monte_carlo"
FIG_PREFIX = "swingby_2d_monte_carlo"
GEOMETRIC_LABEL = "PINN with exact BC"
ORDINARY_LABEL = "PINN with soft BC"
NUM_SEEDS = 100
SEEDS = [1000 + index for index in range(NUM_SEEDS)]
SMOKE_NUM_SEEDS = 2
MAIN_LINEWIDTH = 2.4
SECONDARY_LINEWIDTH = 2.0
BOXPLOT_FIGSIZE = (17.2, 4.8)
COLORS = {
    GEOMETRIC_LABEL: PALETTE["position"],
    ORDINARY_LABEL: PALETTE["vanilla"],
}
BASELINE_LABEL = "Baseline (OpenGoddard)"
ORDINARY_LAMBDA_BC = 0.32397426295281967

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.unicode_minus": True,
        "font.size": 11,
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Monte Carlo for geometric vs ordinary 2D.")
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
    config = deepcopy(exact_bc_2d_config)
    config["label"] = f"{GEOMETRIC_LABEL} | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    return config


def _build_ordinary_config(seed: int) -> dict:
    config = deepcopy(soft_bc_2d_config)
    config["label"] = f"{ORDINARY_LABEL} | seed={seed}"
    config["seed"] = seed
    config["numeric_dtype"] = DTYPE
    config["optimizer"]["w_bc"] = ORDINARY_LAMBDA_BC
    config["plotting"]["linestyle"] = "solid"
    config["plotting"]["trajectory_linestyle"] = "solid"
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
    return baseline_entry


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


def plot_monte_carlo_traj_2d_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize: tuple[float, float] = (7, 7),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    any_entry = next(entry for entries in grouped_entries.values() for entry in entries)

    for group_name, entries in grouped_entries.items():
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(best_result.r[:, 0], best_result.r[:, 1], color=color, linewidth=MAIN_LINEWIDTH, label=group_name)

    ax.plot(any_entry["result"].r0[0], any_entry["result"].r0[1], "o", color="red", label=r"$r(t=0)$")
    ax.plot(any_entry["result"].rN[0], any_entry["result"].rN[1], "x", color="red", label=r"$r(t=1)$")
    plot_masses_2d(ax, get_gravity_sources(any_entry["result"]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_thrust_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize: tuple[float, float] = (7, 4.5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, entries in grouped_entries.items():
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(best_result.t, best_result.F_mag, color=color, linewidth=MAIN_LINEWIDTH, label=group_name)

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Thrust magnitude")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_gravity_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize: tuple[float, float] = (7, 4.5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, entries in grouped_entries.items():
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(best_result.t, best_result.a_mag, color=color, linewidth=MAIN_LINEWIDTH, label=f"{group_name} RFM")
        ax.plot(
            best_result.t,
            best_result.G_mag,
            color=color,
            linewidth=SECONDARY_LINEWIDTH,
            linestyle="--",
            label=f"{group_name} Gravity",
        )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Gravity / Required Force magnitude")
    ax.set_xlim(0, 1)
    ax.legend(ncol=2)
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_boxplots_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = BOXPLOT_FIGSIZE,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    metric_specs = [
        ("delta_v", r"$\Delta V$"),
        ("t_total", "Time of Flight"),
        ("iterations_to_convergence", "Iterations to Convergence"),
    ]
    display_labels = [
        r"$\mathrm{Exact\ BC}$",
        r"$\mathrm{Soft\ BC}$",
    ]

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
            ax.scatter(x_positions + jitter, group_values, color=colors[group_name], s=18, alpha=0.65)

        ylabel_size = 16 if metric_name == "delta_v" else 14
        ylabel_weight = "semibold" if metric_name == "delta_v" else "normal"
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight=ylabel_weight)
        ax.tick_params(axis="x", labelrotation=12, labelsize=14)
        if metric_name == "delta_v" and baseline_entry is not None:
            baseline_delta_v = float(baseline_entry["result"].delta_v)
            baseline_legend = Line2D(
                [],
                [],
                linestyle="None",
                marker=None,
                linewidth=0.0,
                label=f"{BASELINE_LABEL}: {baseline_delta_v:.4g}",
            )
            ax.legend(
                handles=[baseline_legend],
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.95,
                handlelength=0.0,
                handletextpad=0.0,
                borderpad=0.35,
                fontsize=12,
                labelcolor="black",
                prop={"weight": "bold", "size": 12},
            )

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.16, top=0.95, wspace=0.32)
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_collection_run(collection_run: dict, *, output_dir: str | Path | None = None) -> dict[str, list[dict]]:
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

    plot_monte_carlo_traj_2d_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_traj2d",
    )
    plot_monte_carlo_boxplots_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_boxplots",
        baseline_entry=baseline_entry,
    )
    plot_monte_carlo_thrust_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_thrust",
    )
    plot_monte_carlo_gravity_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_gravity",
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


def main(
    *,
    skip_plots: bool = False,
    print_summary: bool = True,
    smoke: bool | None = None,
    from_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    workers: int = 1,
):
    if from_run is not None:
        return replot_saved_run(from_run, output_dir=output_dir, print_summary=print_summary)

    collection_run = run_collection(smoke=smoke, workers=workers)
    persist_paper_monte_carlo_aggregate_summary(
        collection_run,
        title=COLLECTION_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        group_key=monte_carlo_group_key,
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        plot_collection_run(collection_run, output_dir=output_dir)

    return collection_run


def run_collection(*, smoke: bool | None = None, workers: int = 1) -> dict:
    seeds = get_seeds(smoke=smoke)

    if workers <= 1:
        entries = []
        for seed in seeds:
            entries.extend(_run_seed(seed, smoke=bool(smoke)))
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
            + [
                capture_baseline_entry(
                    lambda: build_baseline_entry(smoke=smoke),
                    log_filename="baseline_opengoddard.log",
                )
            ],
        )

    collection_context = RunCollectionContext(label=COLLECTION_LABEL, run_root=str(RUN_ROOT))
    collection_context.start()
    collection_results = []

    try:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            completed_by_seed = pool.starmap(_run_seed, [(seed, bool(smoke)) for seed in seeds])

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

        baseline_entry = capture_baseline_entry(
            lambda: build_baseline_entry(smoke=smoke),
            log_filename="baseline_opengoddard.log",
        )
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


if __name__ == "__main__":
    args = _parse_args()
    main(
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        from_run=args.from_run,
        output_dir=args.output_dir,
        workers=args.workers,
    )

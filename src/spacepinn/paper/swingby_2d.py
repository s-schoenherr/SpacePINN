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
from spacepinn.config.config_2d import geometric_2d_config, ordinary_2d_config
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._aggregate_summary import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import paper_baseline_solver_kwargs
from spacepinn.paper._baseline_summary import (
    get_baseline_entries,
    print_baseline_delta_v_summary,
)
from spacepinn.paper._plot_style import (
    MAIN_AXES_RECT,
    MAIN_FIGSIZE,
    MAIN_LINEWIDTH,
    SECONDARY_LINEWIDTH,
    TRAJECTORY_AXES_RECT,
    TRAJECTORY_FIGSIZE,
    configure_paper_plotter,
)
from spacepinn.opengoddard.geometric_2d_goddard import geometric_2d_opengoddard
from spacepinn.plotting.helpers import get_gravity_sources, get_quiver_data, register_plot_artifact_if_possible, set_time_axis_labels
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import load_run, print_collection_run_summary, run_experiment_collection
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
BOXPLOT_FIGSIZE = (17.2, 4.8)
BOXPLOT_AXIS_LABEL_FONTSIZE = 18
BOXPLOT_DELTA_V_LABEL_FONTSIZE = 20
BOXPLOT_TICK_LABEL_FONTSIZE = 16
BOXPLOT_BASELINE_LEGEND_FONTSIZE = 13
BOXPLOT_SUBPLOT_ADJUST = {
    "left": 0.08,
    "right": 0.985,
    "bottom": 0.20,
    "top": 0.94,
    "wspace": 0.34,
}
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


def _draw_boxplot_panel(ax, values: list[list[float]], *, display_labels: list[str], colors: dict[str, str], scatter_size: float) -> None:
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
        ax.scatter(x_positions + jitter, group_values, color=colors[group_name], s=scatter_size, alpha=0.65)


def _apply_boxplot_ylim(ax, values: list[list[float]], *, extra_top_fraction: float = 0.16) -> None:
    flattened = [float(value) for group_values in values for value in group_values]
    if not flattened:
        return
    ymin = min(flattened)
    ymax = max(flattened)
    span = ymax - ymin
    lower_pad = max(0.0, 0.04 * span)
    upper_pad = max(1.0e-3, extra_top_fraction * max(span, ymax))
    ax.set_ylim(max(0.0, ymin - lower_pad), ymax + upper_pad)


def _trim_soft_bc_display_outliers(values: list[list[float]]) -> list[list[float]]:
    trimmed = [list(group_values) for group_values in values]
    soft_bc_index = 1
    if soft_bc_index >= len(trimmed):
        return trimmed

    soft_bc_values = [float(value) for value in trimmed[soft_bc_index]]
    if len(soft_bc_values) < 2:
        return trimmed

    sorted_values = sorted(soft_bc_values)
    largest = sorted_values[-1]
    second_largest = sorted_values[-2]
    if second_largest <= 0:
        return trimmed

    if largest <= 10.0 * second_largest:
        return trimmed

    trimmed[soft_bc_index] = [value for value in soft_bc_values if value != largest]
    return trimmed


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
    baseline_entry["plotting"]["trajectory_linestyle"] = "solid"
    baseline_entry["plotting"]["quiver_count"] = QUIVER_COUNT
    baseline_entry["color"] = PALETTE["opengoddard"]
    baseline_entry["linestyle"] = "solid"
    baseline_entry["trajectory_linestyle"] = "solid"
    baseline_entry["quiver_count"] = QUIVER_COUNT
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


def _build_plotter(entries: list[dict], *, output_dir: str | Path) -> TrajectoryPlotter:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    return configure_paper_plotter(plotter)


def _representative_entries(collection_run: dict, *, include_baseline: bool = True) -> list[dict]:
    grouped_entries = group_entries(collection_run["entries"])
    entries: list[dict] = []
    for group_name in (GEOMETRIC_LABEL, ORDINARY_LABEL):
        method_entries = grouped_entries.get(group_name, [])
        if not method_entries:
            continue
        best_entry = select_best_entry(method_entries)
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
        baseline_entry = get_baseline_entry(collection_run)
        plotting = dict(baseline_entry.get("plotting", {}))
        plotting["color"] = PALETTE["opengoddard"]
        plotting["linestyle"] = "solid"
        plotting["trajectory_linestyle"] = "solid"
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


def plot_traj_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    fig = plt.figure(figsize=TRAJECTORY_FIGSIZE)
    ax = fig.add_axes(TRAJECTORY_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.r[:, 0],
            result.r[:, 1],
            linestyle=exp.get("trajectory_linestyle", exp["linestyle"]),
            color=exp["color"],
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        r_q, G_q, _ = get_quiver_data(
            result,
            step=exp.get("quiver_step", 10),
            count=exp.get("quiver_count"),
        )
        ax.quiver(
            r_q[:, 0],
            r_q[:, 1],
            G_q[:, 0],
            G_q[:, 1],
            color=exp["color"],
            scale=exp["quiver_scale"],
            label="_nolegend_",
        )

    reference_result = entries[0]["result"]
    ax.plot(reference_result.r0[0], reference_result.r0[1], "o", color="red", label=r"$\mathbf{r}(t_0)$")
    ax.plot(
        reference_result.rN[0],
        reference_result.rN[1],
        "x",
        color="red",
        markersize=7,
        markeredgewidth=1.5,
        label=r"$\mathbf{r}(T)$",
    )
    gravity_sources = get_gravity_sources(reference_result)
    mass_colors = ["#006400", "#228B22", "#6B8E23"]
    mass_labels = [r"$GM_1 = 0.5$", r"$GM_2 = 1.0$", r"$GM_3 = 0.5$"]
    mass_marker_sizes = [170, 300, 170]
    annotation_offsets = [(12, 10), (12, 2), None]
    for index, (x, y, _gm) in enumerate(gravity_sources):
        ax.scatter(x, y, s=mass_marker_sizes[index], color=mass_colors[index], marker="o", zorder=4)
        if index < 2:
            ax.annotate(
                mass_labels[index],
                (x, y),
                xytext=annotation_offsets[index],
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=plotter.legend_fontsize + 3.0,
                color="black",
            )
        else:
            ax.text(
                0.7,
                0.2,
                mass_labels[index],
                ha="left",
                va="center",
                fontsize=plotter.legend_fontsize + 3.0,
                color="black",
            )
    ax.set_xlabel("x / normalized units", labelpad=10)
    ax.set_ylabel("y / normalized units")
    ax.set_box_aspect(1)
    ax.set_aspect("equal")
    plotter.style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="0.3",
        columnspacing=plotter.legend_columnspacing,
        handlelength=plotter.legend_handlelength,
        labelspacing=0.30,
        borderaxespad=0.35,
    )
    plotter.style_legend(legend)
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_traj2d.pdf"
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_thrust_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.t,
            result.F_mag,
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )

    set_time_axis_labels(ax, "Thrust magnitude", plot_legend=True)
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    plotter.style_legend(ax.get_legend())
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_thrust.pdf"
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_gravity_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    ymax = 0.0
    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.t,
            result.a_mag,
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=f"{label} RFM",
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        ax.plot(
            result.t,
            result.G_mag,
            linestyle="dashed" if exp["linestyle"] == "solid" else exp["linestyle"],
            color=exp["color"],
            label=f"{label} Gravity",
            linewidth=plotter.secondary_linewidth,
            zorder=exp["zorder"],
        )
        ymax = max(ymax, max(result.a_mag), max(result.G_mag))

    set_time_axis_labels(ax, "Gravity / Required Force magnitude", plot_legend=False)
    ax.set_box_aspect(1)
    ymin = min(min(min(exp["result"].a_mag), min(exp["result"].G_mag)) for exp in plotter.experiments.values())
    lower_margin = max(0.06, 0.08 * (ymax - ymin))
    upper_margin = max(0.30, 0.36 * ymax)
    ax.set_ylim(max(0.0, ymin - lower_margin), ymax + upper_margin)
    plotter.style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="0.3",
        columnspacing=plotter.legend_columnspacing,
        handlelength=plotter.legend_handlelength,
        labelspacing=0.28,
        borderaxespad=0.35,
    )
    plotter.style_legend(legend)
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_gravity.pdf"
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(str(figure_path))
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
        "Exact BC",
        "Soft BC",
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
        display_values = _trim_soft_bc_display_outliers(values)
        _draw_boxplot_panel(ax, display_values, display_labels=display_labels, colors=colors, scatter_size=18)

        if metric_name == "delta_v":
            _apply_boxplot_ylim(ax, display_values, extra_top_fraction=0.22)

        ylabel_size = BOXPLOT_DELTA_V_LABEL_FONTSIZE if metric_name == "delta_v" else BOXPLOT_AXIS_LABEL_FONTSIZE
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight="normal")
        ax.tick_params(axis="x", labelrotation=0, labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
        if metric_name == "delta_v" and baseline_entry is not None:
            baseline_delta_v = float(baseline_entry["result"].delta_v)
            baseline_legend = Line2D(
                [],
                [],
                linestyle="None",
                marker=None,
                linewidth=0.0,
                label=f"{BASELINE_LABEL}: {baseline_delta_v:.3g}",
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
                borderpad=0.25,
                labelcolor="black",
                prop={"weight": "bold", "size": BOXPLOT_BASELINE_LEGEND_FONTSIZE},
            )

    fig.subplots_adjust(**BOXPLOT_SUBPLOT_ADJUST)
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


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

    representative_entries = _representative_entries(collection_run, include_baseline=False)
    plot_traj_figure(representative_entries, output_dir=target_dir)
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
        collection_run = run_experiment_collection(
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
                for entry in _run_representative_entries(smoke=smoke)
            ]
            + [
                capture_baseline_entry(
                    lambda: build_baseline_entry(smoke=smoke),
                    log_filename="baseline_opengoddard.log",
                )
            ],
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
        return run_experiment_collection(
            configs=[],
            label=label,
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
                    lambda: build_baseline_entry(smoke=smoke_enabled),
                    log_filename="baseline_opengoddard.log",
                )
            ],
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

        baseline_entry = capture_baseline_entry(
            lambda: build_baseline_entry(smoke=smoke_enabled),
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

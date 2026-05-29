from __future__ import annotations

import argparse
from copy import deepcopy
import functools
from functools import partial
import inspect
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import blended_transform_factory

import spacepinn
from spacepinn.config.config_3d import geometric_3d_config, kinematic_3d_config, ordinary_3d_config
from spacepinn.config.shared_parameters import x0_3d, xN_3d
from spacepinn.config.transform_functions import kinematic_fn
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._aggregate_summary import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import paper_baseline_solver_kwargs
from spacepinn.paper._plot_style import (
    LOSS_AXES_RECT,
    LOSS_FIGSIZE,
    MAIN_AXES_RECT,
    MAIN_FIGSIZE,
    MAIN_LINEWIDTH,
    SECONDARY_LINEWIDTH,
    TRAJECTORY_AXES_RECT,
    TRAJECTORY_FIGSIZE,
)
from spacepinn.paper._baseline_summary import (
    get_baseline_entries,
    print_baseline_delta_v_summary,
)
from spacepinn.experiment import build_pretrained_model
from spacepinn.pretraining.kinematic_to_geometric_pretraining_3d import (
    PLANE_VELOCITY,
)
from spacepinn.opengoddard.geometric_3d_goddard import geometric_3d_opengoddard
from spacepinn.plotting.helpers import get_gravity_sources, register_plot_artifact_if_possible
from spacepinn.plotting.paper_style import PAPER_STYLE
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import load_run, print_collection_run_summary, run_experiment_collection
from spacepinn.runner.context import RunCollectionContext
from spacepinn.runner.execution import execute_single_experiment
from spacepinn.runner.runtime import _prepare_runtime_config

DTYPE = "float32"
RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_3d_monte_carlo"
FIG_PREFIX = "swingby_3d_monte_carlo"
BASELINE_LABEL = "Baseline (OpenGoddard)"
GEOMETRIC_LABEL = "PINN with exact BC"
ORDINARY_LABEL = "PINN with soft BC"
LEGACY_ORDINARY_LABEL = "PINN without exact BC"
PRETRAINED_LABEL = "PINN with exact BC and pre-conditioning"
NUM_SEEDS = 100
SEEDS = [2000 + index for index in range(NUM_SEEDS)]
SMOKE_NUM_SEEDS = 2
BOXPLOT_FIGSIZE = (17.2, 4.8)
BOXPLOT_AXIS_LABEL_FONTSIZE = 18
BOXPLOT_DELTA_V_LABEL_FONTSIZE = 20
BOXPLOT_TICK_LABEL_FONTSIZE = 16
BOXPLOT_BASELINE_LEGEND_FONTSIZE = 13
BOXPLOT_INSET_TICK_LABEL_FONTSIZE = 10
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
    PRETRAINED_LABEL: PALETTE["kinematic"],
}
ORDINARY_LAMBDA_BC = 0.42133217438472903

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.unicode_minus": True,
        "font.size": 11,
    }
)


def _style_axes(ax, *, is_3d: bool = False) -> None:
    ax.xaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    ax.yaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    if is_3d and hasattr(ax, "zaxis"):
        ax.zaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=PAPER_STYLE.tick_label_fontsize)
    if is_3d and hasattr(ax, "zaxis"):
        ax.tick_params(axis="z", which="both", labelsize=PAPER_STYLE.tick_label_fontsize)


def _style_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(PAPER_STYLE.legend_fontsize)
    frame = legend.get_frame()
    if frame is not None:
        frame.set_alpha(PAPER_STYLE.legend_framealpha)
        frame.set_edgecolor("black")
        frame.set_linewidth(1.0)
        frame.set_facecolor("white")


def _style_boxplot_axis(ax, *, axis_label_fontsize: float, tick_label_fontsize: float) -> None:
    _style_axes(ax)
    ax.xaxis.label.set_size(axis_label_fontsize)
    ax.yaxis.label.set_size(axis_label_fontsize)
    for label in ax.get_xticklabels():
        label.set_fontsize(tick_label_fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_fontsize)


def _save_figure(fig, figure_path: Path) -> None:
    save_kwargs = {"pad_inches": PAPER_STYLE.save_pad_inches}
    if PAPER_STYLE.save_bbox_inches is not None:
        save_kwargs["bbox_inches"] = PAPER_STYLE.save_bbox_inches
    fig.savefig(figure_path, **save_kwargs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Monte Carlo for geometric vs ordinary 3D.")
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


def _quartiles(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(sorted(float(value) for value in values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def _delta_v_zoom_upper(values: list[list[float]]) -> float:
    flattened = [float(value) for group_values in values for value in group_values]
    if not flattened:
        return 1.0

    visible_values: list[float] = []
    for index, group_values in enumerate(values):
        if not group_values:
            continue
        numeric_values = [float(value) for value in group_values]
        if index == 2:
            q1, q3 = _quartiles(numeric_values)
            iqr = q3 - q1
            group_upper = q3 + 1.5 * iqr
            filtered = [value for value in numeric_values if value <= group_upper]
            visible_values.extend(filtered if filtered else numeric_values)
        else:
            visible_values.extend(numeric_values)

    if not visible_values:
        visible_values = list(flattened)

    max_visible = max(visible_values)
    min_visible = min(visible_values)
    padding = max(2.0e-3, 0.55 * (max_visible - min_visible))
    return max_visible + padding


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


def _loss_label(group_name: str, suffix: str) -> str:
    replacements = {
        GEOMETRIC_LABEL: "Exact BC",
        ORDINARY_LABEL: "Soft BC",
        PRETRAINED_LABEL: "Exact BC + pre-cond.",
    }
    short_group = replacements.get(group_name, group_name.replace("pre-conditioning", "pre-cond."))
    return f"{short_group} {suffix}"


def _projection_label(group_name: str) -> str:
    replacements = {
        GEOMETRIC_LABEL: "Exact BC",
        ORDINARY_LABEL: "Soft BC",
        PRETRAINED_LABEL: "Exact BC + pre-cond.",
        BASELINE_LABEL: "Baseline",
    }
    return replacements.get(group_name, group_name)


def _projected_mass_handles(result) -> list[Line2D]:
    gravity_sources = get_gravity_sources(result)
    if gravity_sources is None:
        return []
    colors = ["#006400", "#228B22", "#6B8E23"]
    sizes = [10, 14, 10]
    handles = []
    for index, source in enumerate(gravity_sources):
        mass = float(source[-1])
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                color=colors[index % len(colors)],
                markersize=sizes[index % len(sizes)],
                label=rf"$GM_{index + 1} = {mass:.1f}$",
            )
        )
    return handles


def _plot_projected_masses(ax, result, *, dims: tuple[int, int]) -> None:
    gravity_sources = get_gravity_sources(result)
    if gravity_sources is None:
        return
    colors = ["#006400", "#228B22", "#6B8E23"]
    sizes = [120, 220, 120]
    for index, source in enumerate(gravity_sources):
        ax.scatter(
            source[dims[0]],
            source[dims[1]],
            s=sizes[index % len(sizes)],
            color=colors[index % len(colors)],
            marker="o",
            zorder=4,
        )


def _set_projection_limits(ax, result, *, dims: tuple[int, int]) -> None:
    values = [result.r[:, dims[0]], result.r[:, dims[1]], result.r0[list(dims)], result.rN[list(dims)]]
    gravity_sources = get_gravity_sources(result)
    if gravity_sources is not None:
        for source in gravity_sources:
            values.append(np.asarray([source[dims[0]], source[dims[1]]], dtype=float))
    flattened = np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values])
    finite = flattened[np.isfinite(flattened)]
    if finite.size == 0:
        lower, upper = -1.0, 1.0
    else:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        padding = max(0.08, 0.05 * (upper - lower))
        lower -= padding
        upper += padding
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)


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


def plot_monte_carlo_traj_2d_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.9))
    any_entry = next(entry for entries in grouped_entries.values() for entry in entries if entries)
    reference_result = any_entry["result"]
    projection_specs = [
        (axes[0, 0], (0, 1), "x / normalized units", "y / normalized units"),
        (axes[0, 1], (0, 2), "x / normalized units", "z / normalized units"),
        (axes[1, 0], (1, 2), "y / normalized units", "z / normalized units"),
    ]
    line_handles: list[Line2D] = []

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        best_result = best_entry["result"]
        handle = None
        for ax, dims, _xlabel, _ylabel in projection_specs:
            (line,) = ax.plot(
                best_result.r[:, dims[0]],
                best_result.r[:, dims[1]],
                color=color,
                linewidth=MAIN_LINEWIDTH,
                label=_projection_label(group_name),
            )
            handle = line
        if handle is not None:
            line_handles.append(handle)

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        baseline_handle = None
        for ax, dims, _xlabel, _ylabel in projection_specs:
            (line,) = ax.plot(
                baseline_result.r[:, dims[0]],
                baseline_result.r[:, dims[1]],
                color=PALETTE["opengoddard"],
                linewidth=MAIN_LINEWIDTH,
                linestyle="dashed",
                label=_projection_label(BASELINE_LABEL),
            )
            baseline_handle = line
        if baseline_handle is not None:
            line_handles.append(baseline_handle)

    start_handle = Line2D([], [], linestyle="None", marker="o", color="red", label=r"$\mathbf{r}(t_0)$")
    end_handle = Line2D(
        [],
        [],
        linestyle="None",
        marker="x",
        color="red",
        markeredgewidth=1.5,
        markersize=7,
        label=r"$\mathbf{r}(T)$",
    )
    for ax, dims, xlabel, ylabel in projection_specs:
        ax.plot(reference_result.r0[dims[0]], reference_result.r0[dims[1]], "o", color="red")
        ax.plot(
            reference_result.rN[dims[0]],
            reference_result.rN[dims[1]],
            "x",
            color="red",
            markersize=7,
            markeredgewidth=1.5,
        )
        _plot_projected_masses(ax, reference_result, dims=dims)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _set_projection_limits(ax, reference_result, dims=dims)
        ax.set_box_aspect(1)
        _style_axes(ax)

    axes[1, 1].set_frame_on(False)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    legend = axes[1, 1].legend(
        handles=[*line_handles, start_handle, end_handle, *_projected_mass_handles(reference_result)],
        loc="center",
        frameon=True,
        borderpad=0.35,
        handlelength=PAPER_STYLE.legend_handlelength,
        labelspacing=0.22,
    )
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.985, wspace=0.26, hspace=0.34)
    fig.savefig(figure_path, pad_inches=PAPER_STYLE.save_pad_inches)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_traj_3d_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig = plt.figure(figsize=TRAJECTORY_FIGSIZE)
    ax = fig.add_axes(TRAJECTORY_AXES_RECT, projection="3d")
    any_entry = next(entry for entries in grouped_entries.values() for entry in entries if entries)

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(
            best_result.r[:, 0],
            best_result.r[:, 1],
            best_result.r[:, 2],
            color=color,
            linewidth=MAIN_LINEWIDTH,
            label=group_name,
        )

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        ax.plot(
            baseline_result.r[:, 0],
            baseline_result.r[:, 1],
            baseline_result.r[:, 2],
            color=PALETTE["opengoddard"],
            linewidth=MAIN_LINEWIDTH,
            linestyle="dashdot",
            label=BASELINE_LABEL,
        )

    ax.scatter(*any_entry["result"].r0, marker="o", color="red", label=r"$r(t=0)$")
    ax.scatter(*any_entry["result"].rN, marker="x", color="red", label=r"$r(t=1)$")
    ax.set_xlabel("x / normalized units")
    ax.set_ylabel("y / normalized units")
    ax.set_zlabel("z / normalized units")
    _style_axes(ax, is_3d=True)
    legend = ax.legend(loc="upper left", ncol=1, frameon=True)
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_loss_figure(entries: list[dict], *, output_dir: str | Path, fig_name: str = f"{FIG_PREFIX}_loss") -> None:
    loss_entries = [entry for entry in entries if entry["label"] != BASELINE_LABEL]
    fig = plt.figure(figsize=LOSS_FIGSIZE)
    ax = fig.add_axes(LOSS_AXES_RECT)

    for index, entry in enumerate(loss_entries):
        label = entry["label"]
        result = entry["result"]
        color = entry.get("color", COLORS.get(label, PALETTE["position"]))
        zorder = len(loss_entries) - index
        ax.plot(
            result.loss,
            linestyle="solid",
            color=color,
            label=_loss_label(label, "Total Loss"),
            linewidth=MAIN_LINEWIDTH,
            zorder=zorder,
        )
        if result.loss_bc:
            ax.plot(
                result.loss_bc,
                linestyle="--",
                color=color,
                label=_loss_label(label, r"$\lambda_{BC}L_{BC}$"),
                linewidth=SECONDARY_LINEWIDTH,
                zorder=zorder,
            )
        if result.loss_physics:
            ax.plot(
                result.loss_physics,
                linestyle="-.",
                color=color,
                label=_loss_label(label, r"$\lambda_{P}L_{P}$"),
                linewidth=SECONDARY_LINEWIDTH,
                zorder=zorder,
            )

    visible_lengths = [len(entry["result"].loss) for entry in loss_entries]
    if visible_lengths:
        ax.set_xlim(0, max(visible_lengths) * 1.18)
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    _style_axes(ax)
    legend = ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        columnspacing=PAPER_STYLE.legend_columnspacing,
        handlelength=PAPER_STYLE.legend_handlelength,
        labelspacing=0.28,
        borderaxespad=0.35,
    )
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
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
        "Exact BC",
        "Soft BC",
        "Exact +\npre-cond.",
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

        _draw_boxplot_panel(ax, values, display_labels=display_labels, colors=colors, scatter_size=18)

        if metric_name == "delta_v":
            _apply_boxplot_ylim(ax, values, extra_top_fraction=0.18)

        ylabel_size = BOXPLOT_DELTA_V_LABEL_FONTSIZE if metric_name == "delta_v" else BOXPLOT_AXIS_LABEL_FONTSIZE
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight="normal")
        ax.tick_params(axis="x", labelrotation=0, labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        _style_boxplot_axis(
            ax,
            axis_label_fontsize=ylabel_size,
            tick_label_fontsize=BOXPLOT_TICK_LABEL_FONTSIZE,
        )
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

            inset = inset_axes(
                ax,
                width="57%",
                height="58%",
                loc="upper left",
                bbox_to_anchor=(0.19, 0.00, 0.78, 0.80),
                bbox_transform=ax.transAxes,
                borderpad=0.0,
            )
            _draw_boxplot_panel(inset, values, display_labels=display_labels, colors=colors, scatter_size=10)
            inset.set_ylim(0.0, 0.009)
            inset.tick_params(axis="x", bottom=False, labelbottom=False)
            inset.tick_params(axis="y", labelsize=BOXPLOT_INSET_TICK_LABEL_FONTSIZE, pad=1)
            inset.set_xticks([])

            cluster_box_transform = blended_transform_factory(ax.transData, ax.transAxes)
            source_box_top = 0.06
            source_rect = Rectangle(
                (0.78, 0.0),
                2.46,
                source_box_top,
                fill=False,
                edgecolor="0.35",
                linewidth=1.1,
                alpha=0.9,
                transform=cluster_box_transform,
            )
            ax.add_patch(source_rect)
            connector = ConnectionPatch(
                xyA=(0.78, source_box_top),
                coordsA=cluster_box_transform,
                xyB=(0.0, 0.0),
                coordsB=inset.transAxes,
                color="0.35",
                linewidth=1.1,
                alpha=0.9,
            )
            ax.add_artist(connector)

    fig.subplots_adjust(**BOXPLOT_SUBPLOT_ADJUST)
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_thrust_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(best_result.t, best_result.F_mag, color=color, linewidth=MAIN_LINEWIDTH, label=group_name)

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        ax.plot(
            baseline_result.t,
            baseline_result.F_mag,
            color=PALETTE["opengoddard"],
            linewidth=MAIN_LINEWIDTH,
            linestyle="dashdot",
            label=BASELINE_LABEL,
        )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Thrust magnitude")
    ax.set_xlim(0, 1)
    ax.set_box_aspect(1)
    _style_axes(ax)
    legend = ax.legend(loc="upper left", ncol=1, frameon=True)
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_gravity_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    def _gravity_label(group_name: str, quantity: str) -> str:
        short_group = group_name.replace("pre-conditioning", "pre-cond.")
        short_quantity = "Grav." if quantity == "Gravity" else quantity
        return f"{short_group} {short_quantity}"

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(
            best_result.t,
            best_result.a_mag,
            color=color,
            linewidth=MAIN_LINEWIDTH,
            label=_gravity_label(group_name, "RFM"),
        )
        ax.plot(
            best_result.t,
            best_result.G_mag,
            color=color,
            linewidth=SECONDARY_LINEWIDTH,
            linestyle="--",
            label=_gravity_label(group_name, "Gravity"),
        )

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        ax.plot(
            baseline_result.t,
            baseline_result.a_mag,
            color=PALETTE["opengoddard"],
            linewidth=MAIN_LINEWIDTH,
            linestyle="dashdot",
            label=_gravity_label(BASELINE_LABEL, "RFM"),
        )
        ax.plot(
            baseline_result.t,
            baseline_result.G_mag,
            color=PALETTE["opengoddard"],
            linewidth=SECONDARY_LINEWIDTH,
            linestyle=(0, (5, 2, 1, 2)),
            label=_gravity_label(BASELINE_LABEL, "Gravity"),
        )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Gravity / Required Force magnitude")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.0)
    ax.set_box_aspect(1)
    _style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        columnspacing=0.8,
        handlelength=1.4,
        labelspacing=0.28,
        borderaxespad=0.35,
    )
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_collection_run(collection_run: dict, *, output_dir: str | Path | None = None) -> dict[str, list[dict]]:
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


def run_collection(*, smoke: bool | None = None, workers: int = 1) -> dict:
    seeds = get_seeds(smoke=smoke)

    if workers <= 1:
        entries = []
        for seed in seeds:
            entries.extend(_run_seed(seed, smoke=bool(smoke)))
        return _finalize_collection_entries(entries, smoke=smoke)

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


def _finalize_collection_entries(entries: list[dict], *, smoke: bool | None = None) -> dict:
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
            for entry in entries
        ]
        + [
            capture_baseline_entry(
                lambda: build_baseline_entry(smoke=smoke),
                log_filename="baseline_opengoddard.log",
            )
        ],
    )
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


if __name__ == "__main__":
    args = _parse_args()
    main(
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        from_run=args.from_run,
        output_dir=args.output_dir,
        workers=args.workers,
    )

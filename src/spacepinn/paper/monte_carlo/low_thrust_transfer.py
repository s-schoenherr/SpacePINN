from __future__ import annotations

import argparse
import functools
from copy import deepcopy
import inspect
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import spacepinn
import torch
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._aggregate_summary import persist_paper_monte_carlo_aggregate_summary
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_summary import (
    get_baseline_entries,
    print_baseline_delta_v_summary,
)
from spacepinn.paper.low_thrust_transfer import (
    BASELINE_COLOR,
    BASELINE_LABEL,
    MAIN_FIGSIZE,
    PINN_COLOR,
    PINN_LABEL,
    TARGET_ORBITS,
    build_baseline_entry,
    build_config as build_single_config,
    plot_gravity_figure,
    plot_orbit_figure,
    plot_thrust_figure,
)
from spacepinn.plotting.helpers import register_plot_artifact_if_possible
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import load_run, print_collection_run_summary, run_experiment_collection
from spacepinn.runner.context import RunCollectionContext
from spacepinn.runner.execution import execute_single_experiment
from spacepinn.runner.runtime import _prepare_runtime_config

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "low_thrust_transfer_monte_carlo"
FIG_PREFIX = "low_thrust_transfer_monte_carlo"
NUM_SEEDS = 100
SEEDS = [5000 + index for index in range(NUM_SEEDS)]
SMOKE_NUM_SEEDS = 2
BOXPLOT_FIGSIZE = (17.2, 4.8)
COLORS = {
    PINN_LABEL: PINN_COLOR,
}
MONTE_CARLO_CONVERGENCE_THRESHOLD = 1e-6


def plot_loss_figure(entries: list[dict], *, output_dir: str) -> None:
    loss_entries = [entry for entry in entries if entry["label"] != BASELINE_LABEL]
    plotter = TrajectoryPlotter(loss_entries, dim=2, figsize=MAIN_FIGSIZE, fig_prefix=FIG_PREFIX, output_dir=output_dir)
    plotter.main_linewidth = 2.8
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    positive_losses: list[np.ndarray] = []
    for label, exp in plotter.experiments.items():
        result = exp["result"]
        loss_values = np.maximum(np.asarray(result.loss, dtype=float), 1e-12)
        positive_losses.append(loss_values)
        ax.plot(
            loss_values,
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=label + " Total Loss",
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )

    visible_lengths = [len(entry["result"].loss) for entry in loss_entries]
    if visible_lengths:
        ax.set_xlim(0, max(visible_lengths))

    if positive_losses:
        all_losses = np.concatenate(positive_losses)
        loss_min = float(np.min(all_losses))
        loss_max = float(np.max(all_losses))
        ax.set_ylim(loss_min / 1.35, loss_max * 1.35)

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right", framealpha=0.98, facecolor="white", edgecolor="0.3")
    fig.tight_layout()
    figure_path = plotter._build_figure_path("loss")
    fig.savefig(figure_path)
    register_plot_artifact_if_possible(figure_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper Monte Carlo for low-thrust transfer.")
    parser.add_argument("--terminal-angle-pi", type=float, default=None)
    parser.add_argument("--time-guess-scale", type=float, default=None)
    parser.add_argument("--extra-turns", type=int, default=None)
    parser.add_argument("--tof-scale", type=float, default=None)
    parser.add_argument("--target-orbit", choices=sorted(TARGET_ORBITS), default="geo")
    parser.add_argument("--baseline-run", default=None, help="Optional saved run to reuse the baseline entry from.")
    parser.add_argument("--from-run", default=None, help="Optional saved collection run to replot without retraining.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write plots to. Defaults to the run's artifacts/plots directory.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Run the collection without plotting.")
    parser.add_argument("--skip-summary", action="store_true", help="Suppress the printed collection summary.")
    parser.add_argument("--workers", type=int, default=1, help="Optional multiprocessing worker count for the seed runs.")
    return parser.parse_args()


def get_seeds(*, smoke: bool | None = None) -> list[int]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    return SEEDS[:SMOKE_NUM_SEEDS] if smoke_enabled else SEEDS


def build_configs(
    *,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> list[dict]:
    return [
        _build_seed_config(
            seed,
            target_orbit=target_orbit,
            terminal_angle_pi=terminal_angle_pi,
            time_guess_scale=time_guess_scale,
            extra_turns=extra_turns,
            tof_scale=tof_scale,
            smoke=smoke,
        )
        for seed in get_seeds(smoke=smoke)
    ]


def _build_seed_config(
    seed: int,
    *,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
) -> dict:
    config = deepcopy(
        build_single_config(
            target_orbit=target_orbit,
            terminal_angle_pi=terminal_angle_pi,
            time_guess_scale=time_guess_scale,
            extra_turns=extra_turns,
            tof_scale=tof_scale,
            smoke=smoke,
        )
    )
    config["label"] = f"{PINN_LABEL} | seed={seed}"
    config["seed"] = seed
    config["optimizer"]["convergence_threshold"] = MONTE_CARLO_CONVERGENCE_THRESHOLD
    return config


def group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    grouped = {PINN_LABEL: []}
    for entry in entries:
        label = str(entry.get("label", ""))
        if label.startswith(PINN_LABEL):
            grouped[PINN_LABEL].append(entry)
    return grouped


def _normalize_baseline_entry_style(entry: dict) -> dict:
    normalized = dict(entry)
    plotting = dict(normalized.get("plotting", {}))
    plotting["color"] = BASELINE_COLOR
    plotting["linestyle"] = "solid"
    plotting["trajectory_linestyle"] = "solid"
    plotting["zorder"] = plotting.get("zorder", 2)
    normalized["plotting"] = plotting
    normalized["color"] = BASELINE_COLOR
    normalized["linestyle"] = "solid"
    normalized["trajectory_linestyle"] = "solid"
    normalized["zorder"] = plotting["zorder"]
    normalized["source"] = "opengoddard"
    return normalized


def get_baseline_entry(
    collection_run: dict,
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
    baseline_run: str | Path | None = None,
) -> dict:
    if baseline_run is not None:
        baseline_collection = load_run(baseline_run)
        baselines = get_baseline_entries(baseline_collection.get("entries", []), baseline_labels=(BASELINE_LABEL,))
        return _normalize_baseline_entry_style(baselines[0])
    try:
        baselines = get_baseline_entries(collection_run.get("entries", []), baseline_labels=(BASELINE_LABEL,))
    except ValueError:
        return build_baseline_entry(
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
        )
    return _normalize_baseline_entry_style(baselines[0])


def select_median_entry(entries: list[dict]) -> dict:
    sorted_entries = sorted(entries, key=lambda entry: float(entry["result"].delta_v))
    return sorted_entries[len(sorted_entries) // 2]


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


def _run_seed(
    seed: int,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool = False,
) -> dict:
    config = _build_seed_config(
        seed,
        target_orbit=target_orbit,
        terminal_angle_pi=terminal_angle_pi,
        time_guess_scale=time_guess_scale,
        extra_turns=extra_turns,
        tof_scale=tof_scale,
        smoke=smoke,
    )
    config_runtime = _prepare_runtime_config(deepcopy(config))
    model, result = execute_single_experiment(config_runtime)
    return _entry_payload(
        label=config_runtime["label"],
        result=result,
        config=_make_config_multiprocessing_safe(config_runtime),
    )


def _representative_entries(
    collection_run: dict,
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
    baseline_run: str | Path | None = None,
) -> list[dict]:
    grouped_entries = group_entries(collection_run["entries"])
    median_entry = select_median_entry(grouped_entries[PINN_LABEL])
    baseline_entry = get_baseline_entry(
        collection_run,
        target_orbit=target_orbit,
        time_guess_scale=time_guess_scale,
        tof_scale=tof_scale,
        smoke=smoke,
        baseline_run=baseline_run,
    )
    return [
        {
            **median_entry,
            "label": PINN_LABEL,
        },
        {
            **baseline_entry,
            "plotting": {
                **baseline_entry.get("plotting", {}),
                "color": baseline_entry.get("plotting", {}).get("color", BASELINE_COLOR),
            },
        },
    ]


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
        ("delta_v", r"$\Delta\mathrm{V}$"),
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

        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis="x", labelrotation=12)
        if metric_name == "delta_v" and baseline_entry is not None:
            baseline_delta_v = float(baseline_entry["result"].delta_v)
            ax.text(
                0.03,
                0.96,
                f"{BASELINE_LABEL}: {baseline_delta_v:.4g}",
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
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)


def plot_collection_run(
    collection_run: dict,
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
    output_dir: str | Path | None = None,
    baseline_run: str | Path | None = None,
) -> dict[str, list[dict]]:
    target_dir = Path(output_dir) if output_dir is not None else Path(collection_run["plot_output_dir"])
    target_dir.mkdir(parents=True, exist_ok=True)

    grouped_entries = group_entries(collection_run["entries"])
    baseline_entry = get_baseline_entry(
        collection_run,
        target_orbit=target_orbit,
        time_guess_scale=time_guess_scale,
        tof_scale=tof_scale,
        smoke=smoke,
        baseline_run=baseline_run,
    )
    summary_entries = list(collection_run["entries"])
    if baseline_entry not in summary_entries:
        summary_entries.append(baseline_entry)
    print_monte_carlo_summary(grouped_entries, title="Low-thrust transfer Monte Carlo")
    print_baseline_delta_v_summary(
        summary_entries,
        title=COLLECTION_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        group_key=monte_carlo_group_key,
        include_variance=True,
    )

    representative_entries = _representative_entries(
        collection_run,
        target_orbit=target_orbit,
        time_guess_scale=time_guess_scale,
        tof_scale=tof_scale,
        smoke=smoke,
        baseline_run=baseline_run,
    )
    plotter = TrajectoryPlotter(
        representative_entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=str(target_dir),
    )
    plot_loss_figure(representative_entries, output_dir=str(target_dir))
    plot_thrust_figure(representative_entries, output_dir=str(target_dir))
    plot_gravity_figure(representative_entries, output_dir=str(target_dir))
    plot_orbit_figure(representative_entries, output_dir=str(target_dir), target_orbit=target_orbit)
    plot_monte_carlo_boxplots_paper(
        grouped_entries,
        colors=COLORS,
        output_dir=target_dir,
        fig_name=f"{FIG_PREFIX}_boxplots",
        baseline_entry=baseline_entry,
    )
    return grouped_entries


def replot_saved_run(
    run_dir: str | Path,
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    output_dir: str | Path | None = None,
    print_summary: bool = True,
    baseline_run: str | Path | None = None,
) -> dict:
    collection_run = load_run(run_dir)
    collection_run["plot_output_dir"] = str(
        Path(output_dir) if output_dir is not None else Path(collection_run["run_dir"]) / "artifacts" / "plots"
    )
    persist_paper_monte_carlo_aggregate_summary(
        collection_run,
        title=COLLECTION_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        group_key=monte_carlo_group_key,
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    plot_collection_run(
        collection_run,
        target_orbit=target_orbit,
        time_guess_scale=time_guess_scale,
        tof_scale=tof_scale,
        output_dir=output_dir,
        baseline_run=baseline_run,
    )
    return collection_run


def _finalize_collection_entries(
    entries: list[dict],
    *,
    target_orbit: str = "geo",
    time_guess_scale: float | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
    baseline_run: str | Path | None = None,
) -> dict:
    baseline_entry = (
        get_baseline_entry(
            {"entries": []},
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
            baseline_run=baseline_run,
        )
        if baseline_run is not None
        else capture_baseline_entry(
            lambda: build_baseline_entry(
                target_orbit=target_orbit,
                time_guess_scale=time_guess_scale,
                tof_scale=tof_scale,
                smoke=smoke,
            ),
            log_filename="baseline_opengoddard.log",
        )
    )
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
        + [baseline_entry],
    )


def run_collection(
    *,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
    workers: int = 1,
    baseline_run: str | Path | None = None,
) -> dict:
    seeds = get_seeds(smoke=smoke)

    if workers <= 1:
        entries = [
            _run_seed(
                seed,
                target_orbit=target_orbit,
                terminal_angle_pi=terminal_angle_pi,
                time_guess_scale=time_guess_scale,
                extra_turns=extra_turns,
                tof_scale=tof_scale,
                smoke=bool(smoke),
            )
            for seed in seeds
        ]
        return _finalize_collection_entries(
            entries,
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
            baseline_run=baseline_run,
        )

    collection_context = RunCollectionContext(label=COLLECTION_LABEL, run_root=str(RUN_ROOT))
    collection_context.start()
    collection_results = []

    try:
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            completed_entries = pool.starmap(
                _run_seed,
                [
                    (seed, target_orbit, terminal_angle_pi, time_guess_scale, extra_turns, tof_scale, bool(smoke))
                    for seed in seeds
                ],
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

        baseline_entry = (
            get_baseline_entry(
                {"entries": []},
                target_orbit=target_orbit,
                time_guess_scale=time_guess_scale,
                tof_scale=tof_scale,
                smoke=smoke,
                baseline_run=baseline_run,
            )
            if baseline_run is not None
            else capture_baseline_entry(
                lambda: build_baseline_entry(
                    target_orbit=target_orbit,
                    time_guess_scale=time_guess_scale,
                    tof_scale=tof_scale,
                    smoke=smoke,
                ),
                log_filename="baseline_opengoddard.log",
            )
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


def main(
    *,
    skip_plots: bool = False,
    print_summary: bool = True,
    target_orbit: str = "geo",
    terminal_angle_pi: float | None = None,
    time_guess_scale: float | None = None,
    extra_turns: int | None = None,
    tof_scale: float | None = None,
    smoke: bool | None = None,
    from_run: str | Path | None = None,
    output_dir: str | Path | None = None,
    workers: int = 1,
    baseline_run: str | Path | None = None,
):
    if from_run is not None:
        return replot_saved_run(
            from_run,
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            output_dir=output_dir,
            print_summary=print_summary,
            baseline_run=baseline_run,
        )

    collection_run = run_collection(
        target_orbit=target_orbit,
        terminal_angle_pi=terminal_angle_pi,
        time_guess_scale=time_guess_scale,
        extra_turns=extra_turns,
        tof_scale=tof_scale,
        smoke=smoke,
        workers=workers,
        baseline_run=baseline_run,
    )
    persist_paper_monte_carlo_aggregate_summary(
        collection_run,
        title=COLLECTION_LABEL,
        baseline_labels=(BASELINE_LABEL,),
        group_key=monte_carlo_group_key,
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        plot_collection_run(
            collection_run,
            target_orbit=target_orbit,
            time_guess_scale=time_guess_scale,
            tof_scale=tof_scale,
            smoke=smoke,
            output_dir=output_dir,
            baseline_run=baseline_run,
        )

    return collection_run


if __name__ == "__main__":
    args = _parse_args()
    main(
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
        target_orbit=args.target_orbit,
        terminal_angle_pi=args.terminal_angle_pi,
        time_guess_scale=args.time_guess_scale,
        extra_turns=args.extra_turns,
        tof_scale=args.tof_scale,
        baseline_run=args.baseline_run,
        from_run=args.from_run,
        output_dir=args.output_dir,
        workers=args.workers,
    )

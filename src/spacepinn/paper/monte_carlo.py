from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from spacepinn.paper.baseline import get_baseline_entries, group_nonbaseline_entries
from spacepinn.paper.runtime import smoke_mode_enabled
from spacepinn.paper.style import MAIN_FIGSIZE, configure_paper_plotter
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.plotting.helpers import register_plot_artifact_if_possible


def add_single_mc_arguments(parser: argparse.ArgumentParser, *, default_mode: str = "single") -> None:
    parser.add_argument("--mode", choices=("single", "mc"), default=default_mode)
    parser.add_argument("--mc", action="store_true", help="Shortcut for --mode mc.")
    parser.add_argument("--representative-seed", type=int, default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)


def resolve_mode(args) -> str:
    return "mc" if getattr(args, "mc", False) else getattr(args, "mode", "single")


def seed_sequence(*, start: int, count: int, smoke: bool | None = None) -> list[int]:
    if smoke_mode_enabled() if smoke is None else smoke:
        return [int(start)]
    return list(range(int(start), int(start) + int(count)))


def label_with_seed(base_label: str, seed: int) -> str:
    return f"{base_label} | seed={int(seed)}"


def strip_seed(label: str) -> str:
    return re.sub(r"\s*\|\s*seed=\d+\s*$", "", str(label))


def single_group_key(entry: dict, *, base_label: str) -> str | None:
    if entry.get("source") != "pinn":
        return None
    label = strip_seed(str(entry.get("label", "")))
    return base_label if label == base_label else label


def representative_entries(entries: Iterable[dict], *, representative_seed: int | None, base_label: str) -> list[dict]:
    entries = list(entries)
    selected: list[dict] = []
    pinn_entries = [entry for entry in entries if entry.get("source") == "pinn"]
    if representative_seed is not None:
        suffix = f"seed={int(representative_seed)}"
        selected.extend(entry for entry in pinn_entries if suffix in str(entry.get("label", "")))
    if not selected and pinn_entries:
        selected.append(min(pinn_entries, key=lambda entry: float(entry["result"].delta_v)))
    selected.extend(entry for entry in entries if entry.get("source") != "pinn")
    copied = [dict(entry) for entry in selected]
    for entry in copied:
        if entry.get("source") == "pinn":
            entry["label"] = base_label
    return copied


def _group_metric_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "std_population": None,
            "variance_population": None,
            "std_sample": None,
            "variance_sample": None,
            "min": None,
            "max": None,
        }
    array = np.asarray([float(value) for value in values], dtype=float)
    sample_variance = float(np.var(array, ddof=1)) if array.size > 1 else 0.0
    sample_std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std_population": float(np.std(array, ddof=0)),
        "variance_population": float(np.var(array, ddof=0)),
        "std_sample": sample_std,
        "variance_sample": sample_variance,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_paper_monte_carlo_aggregate_summary(
    entries: list[dict[str, Any]],
    *,
    title: str,
    baseline_labels: tuple[str, ...],
    group_key,
    baseline_sources: tuple[str, ...] = ("opengoddard",),
) -> dict[str, Any]:
    grouped = group_nonbaseline_entries(
        entries,
        baseline_labels=baseline_labels,
        group_key=group_key,
        baseline_sources=baseline_sources,
    )
    baselines = get_baseline_entries(
        entries,
        baseline_labels=baseline_labels,
        baseline_sources=baseline_sources,
    )

    groups_payload: list[dict[str, Any]] = []
    for group_name, group_entries in grouped.items():
        delta_v_values = [float(entry["result"].delta_v) for entry in group_entries]
        t_total_values = [float(entry["result"].t_total) for entry in group_entries]
        runtime_values = [
            float(entry["result"].runtime_seconds)
            for entry in group_entries
            if getattr(entry["result"], "runtime_seconds", None) is not None
        ]
        groups_payload.append(
            {
                "label": group_name,
                "n": len(group_entries),
                "delta_v": _group_metric_stats(delta_v_values),
                "t_total": _group_metric_stats(t_total_values),
                "runtime_seconds": _group_metric_stats(runtime_values),
            }
        )

    baselines_payload: list[dict[str, Any]] = []
    for baseline_entry in baselines:
        baseline_result = baseline_entry["result"]
        baseline_delta_v = float(baseline_result.delta_v)
        comparisons = []
        for group_name, group_entries in grouped.items():
            delta_v_values = [float(entry["result"].delta_v) for entry in group_entries]
            runtime_values = [
                float(entry["result"].runtime_seconds)
                for entry in group_entries
                if getattr(entry["result"], "runtime_seconds", None) is not None
            ]
            delta_v_array = np.asarray(delta_v_values, dtype=float)
            comparisons.append(
                {
                    "label": group_name,
                    "n": len(group_entries),
                    "delta_v_mean": float(delta_v_array.mean()),
                    "delta_v_variance_sample": float(np.var(delta_v_array, ddof=1)) if delta_v_array.size > 1 else 0.0,
                    "relative_mean_improvement_vs_baseline": float((baseline_delta_v - delta_v_array.mean()) / baseline_delta_v),
                    "runtime_seconds_mean": float(np.mean(runtime_values)) if runtime_values else None,
                }
            )
        baselines_payload.append(
            {
                "label": baseline_entry["label"],
                "delta_v": baseline_delta_v,
                "runtime_seconds": (
                    float(baseline_result.runtime_seconds)
                    if getattr(baseline_result, "runtime_seconds", None) is not None
                    else None
                ),
                "solver": getattr(baseline_result, "solver_metadata", None),
                "comparisons": comparisons,
            }
        )

    return {
        "title": title,
        "num_entries": len(entries),
        "monte_carlo_summary": {"groups": groups_payload},
        "baseline_comparison": {"baselines": baselines_payload},
    }


def persist_paper_monte_carlo_aggregate_summary(
    collection_run: dict[str, Any],
    *,
    title: str,
    baseline_labels: tuple[str, ...],
    group_key,
    baseline_sources: tuple[str, ...] = ("opengoddard",),
) -> dict[str, Any]:
    payload = build_paper_monte_carlo_aggregate_summary(
        collection_run.get("entries", []),
        title=title,
        baseline_labels=baseline_labels,
        group_key=group_key,
        baseline_sources=baseline_sources,
    )
    payload["label"] = collection_run.get("label")
    payload["run_id"] = collection_run.get("run_id")
    payload["run_dir"] = collection_run.get("run_dir")

    run_dir = Path(str(collection_run["run_dir"]))
    aggregate_summary_path = run_dir / "aggregate_summary.json"
    aggregate_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("paths", {})["aggregate_summary"] = str(aggregate_summary_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    collection_run["aggregate_summary"] = payload
    collection_run["aggregate_summary_path"] = str(aggregate_summary_path)
    return payload


def plot_single_group_boxplots(
    entries: list[dict],
    *,
    output_dir: str | Path,
    fig_prefix: str,
    base_label: str,
    baseline_labels: tuple[str, ...] = (),
) -> None:
    pinn_entries = [entry for entry in entries if entry.get("source") == "pinn"]
    if not pinn_entries:
        return

    metrics = [
        ("delta_v", r"$\Delta V$"),
        ("t_total", "Time of Flight"),
        ("epochs", "Training iterations"),
    ]
    values = {
        "delta_v": [float(entry["result"].delta_v) for entry in pinn_entries],
        "t_total": [float(entry["result"].t_total) for entry in pinn_entries],
        "epochs": [len(getattr(entry["result"], "loss", []) or []) for entry in pinn_entries],
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    color = "#2ca02c"
    for ax, (metric, ylabel) in zip(axes, metrics):
        data = values[metric]
        ax.boxplot(
            [data],
            patch_artist=True,
            widths=0.42,
            boxprops={"facecolor": color, "alpha": 0.35, "edgecolor": color, "linewidth": 1.4},
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": "black", "linewidth": 1.1},
            capprops={"color": "black", "linewidth": 1.1},
            flierprops={"marker": "o", "markerfacecolor": "none", "markeredgecolor": "black", "markersize": 4},
        )
        x = np.ones(len(data), dtype=float) + np.linspace(-0.055, 0.055, len(data))
        ax.scatter(x, data, color=color, s=18, alpha=0.65, edgecolors="none")
        ax.set_xticks([1])
        ax.set_xticklabels([base_label])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=8)
        ax.set_box_aspect(0.75)

        if metric == "delta_v":
            for baseline in entries:
                if baseline.get("label") in baseline_labels:
                    ax.axhline(float(baseline["result"].delta_v), color="0.25", linestyle="-.", linewidth=1.5)
                    ax.text(
                        0.03,
                        0.94,
                        f"{baseline['label']}: {float(baseline['result'].delta_v):.3g}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        fontweight="bold",
                        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.18", "alpha": 0.95},
                    )
                    break

    plotter = TrajectoryPlotter([], figsize=MAIN_FIGSIZE)
    configure_paper_plotter(plotter)
    for ax in axes:
        plotter.style_axes(ax)
    fig.tight_layout(w_pad=1.2)
    figure_path = Path(output_dir) / f"{fig_prefix}_boxplots.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spacepinn.paper.baseline import print_baseline_delta_v_summary
from spacepinn.paper.monte_carlo import (
    persist_paper_monte_carlo_aggregate_summary,
    plot_single_group_boxplots,
    representative_entries,
    seed_sequence,
)
from spacepinn.plotting.monte_carlo import print_monte_carlo_summary
from spacepinn.runner import print_collection_run_summary, run_experiment_collection


Entry = dict[str, Any]
CollectionRun = dict[str, Any]


@dataclass(frozen=True)
class ExperimentSuite:
    label: str
    run_root: str | Path
    representative_seed: int
    mc_seed_start: int
    mc_num_seeds: int
    build_config: Callable[..., dict[str, Any]]
    build_baseline_entries: Callable[..., list[Entry]]
    plot_representative: Callable[[list[Entry], str], None]
    group_key: Callable[[Entry], str | None]
    base_label: str
    baseline_labels: tuple[str, ...]
    fig_prefix: str
    boxplots_in_mc: bool = True


def run_experiment_suite(
    suite: ExperimentSuite,
    *,
    mode: str = "single",
    skip_plots: bool = False,
    print_summary: bool = True,
    smoke: bool | None = None,
    representative_seed: int | None = None,
    seed_start: int | None = None,
    num_seeds: int | None = None,
) -> CollectionRun:
    representative_seed = suite.representative_seed if representative_seed is None else int(representative_seed)
    seed_start = suite.mc_seed_start if seed_start is None else int(seed_start)
    num_seeds = suite.mc_num_seeds if num_seeds is None else int(num_seeds)

    if mode == "mc":
        seeds = seed_sequence(start=seed_start, count=num_seeds, smoke=smoke)
        configs = [suite.build_config(seed=seed, label_seed=True, smoke=smoke) for seed in seeds]
        collection_label = f"{suite.label}_monte_carlo"
    else:
        configs = [suite.build_config(seed=representative_seed, label_seed=False, smoke=smoke)]
        collection_label = suite.label

    collection_run = run_experiment_collection(
        configs=configs,
        label=collection_label,
        run_root=str(suite.run_root),
        additional_entries=suite.build_baseline_entries(smoke=smoke),
    )

    if mode == "mc":
        persist_paper_monte_carlo_aggregate_summary(
            collection_run,
            title=collection_run["label"],
            baseline_labels=suite.baseline_labels,
            group_key=suite.group_key,
        )

    if print_summary:
        print_collection_run_summary(collection_run)
        if mode == "mc":
            grouped = {
                suite.base_label: [
                    entry for entry in collection_run["entries"] if entry.get("source") == "pinn"
                ]
            }
            print_monte_carlo_summary(grouped, title=collection_run["label"])
        print_baseline_delta_v_summary(
            collection_run["entries"],
            title=collection_run["label"],
            baseline_labels=suite.baseline_labels,
            group_key=suite.group_key if mode == "mc" else None,
            include_variance=mode == "mc",
        )

    if not skip_plots:
        plot_entries = representative_entries(
            collection_run["entries"],
            representative_seed=representative_seed,
            base_label=suite.base_label,
        )
        suite.plot_representative(plot_entries, collection_run["plot_output_dir"])
        if mode == "mc" and suite.boxplots_in_mc:
            plot_single_group_boxplots(
                collection_run["entries"],
                output_dir=collection_run["plot_output_dir"],
                fig_prefix=suite.fig_prefix,
                base_label=suite.base_label,
                baseline_labels=suite.baseline_labels,
            )

    return collection_run


def as_collection_entry(entry: Entry) -> Entry:
    payload = {
        "label": entry["label"],
        "result": entry["result"],
        "config": entry.get("config"),
        "model": entry.get("model"),
        "plotting": dict(entry.get("plotting", {})),
        "source": entry.get("source", "pinn"),
    }
    if "log_text" in entry:
        payload["log_text"] = entry["log_text"]
    if "log_filename" in entry:
        payload["log_filename"] = entry["log_filename"]
    return payload


def run_entry_collection(
    *,
    entries: list[Entry],
    label: str,
    run_root: str | Path,
    baseline_entries: list[Entry] | None = None,
) -> CollectionRun:
    return run_experiment_collection(
        configs=[],
        label=label,
        run_root=str(run_root),
        additional_entries=[as_collection_entry(entry) for entry in entries] + list(baseline_entries or []),
    )

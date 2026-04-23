from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional

import numpy as np


Entry = dict
GroupKeyFn = Callable[[Entry], Optional[str]]


def is_baseline_entry(entry: Entry, *, baseline_labels: tuple[str, ...], baseline_sources: tuple[str, ...]) -> bool:
    label = str(entry.get("label", ""))
    source = str(entry.get("source", ""))
    return label in baseline_labels or source in baseline_sources


def get_baseline_entries(
    entries: list[Entry],
    *,
    baseline_labels: tuple[str, ...],
    baseline_sources: tuple[str, ...] = ("opengoddard",),
) -> list[Entry]:
    matched = [
        entry
        for entry in entries
        if is_baseline_entry(entry, baseline_labels=baseline_labels, baseline_sources=baseline_sources)
    ]
    if not matched:
        raise ValueError(f"No baseline entry found for labels={baseline_labels} and sources={baseline_sources}.")

    ordered: "OrderedDict[str, Entry]" = OrderedDict()
    for baseline_label in baseline_labels:
        for entry in matched:
            if entry.get("label") == baseline_label:
                ordered.setdefault(baseline_label, entry)
    for entry in matched:
        ordered.setdefault(str(entry.get("label", "baseline")), entry)
    return list(ordered.values())


def group_nonbaseline_entries(
    entries: list[Entry],
    *,
    baseline_labels: tuple[str, ...],
    group_key: GroupKeyFn | None = None,
    baseline_sources: tuple[str, ...] = ("opengoddard",),
) -> "OrderedDict[str, list[Entry]]":
    grouped: "OrderedDict[str, list[Entry]]" = OrderedDict()
    for entry in entries:
        if is_baseline_entry(entry, baseline_labels=baseline_labels, baseline_sources=baseline_sources):
            continue
        key = group_key(entry) if group_key is not None else str(entry.get("label", "unknown"))
        if key is None:
            continue
        grouped.setdefault(str(key), []).append(entry)
    return grouped


def print_baseline_delta_v_summary(
    entries: list[Entry],
    *,
    title: str,
    baseline_labels: tuple[str, ...],
    group_key: GroupKeyFn | None = None,
    include_variance: bool = False,
    baseline_sources: tuple[str, ...] = ("opengoddard",),
) -> None:
    baselines = get_baseline_entries(
        entries,
        baseline_labels=baseline_labels,
        baseline_sources=baseline_sources,
    )
    grouped = group_nonbaseline_entries(
        entries,
        baseline_labels=baseline_labels,
        group_key=group_key,
        baseline_sources=baseline_sources,
    )

    print()
    print("*" * 92)
    print(f"[SWINGBY] Baseline Comparison: {title}")
    for baseline_entry in baselines:
        baseline_result = baseline_entry["result"]
        baseline_delta_v = float(baseline_result.delta_v)
        baseline_runtime = getattr(baseline_result, "runtime_seconds", None)
        print(
            f"{baseline_entry['label']}: delta_v={baseline_delta_v:.6g}"
            + (f" | cpu_time={float(baseline_runtime):.6g}s" if baseline_runtime is not None else "")
        )
        for group_name, group_entries in grouped.items():
            delta_v = np.array([float(entry["result"].delta_v) for entry in group_entries], dtype=float)
            runtime_values = np.array(
                [
                    float(entry["result"].runtime_seconds)
                    for entry in group_entries
                    if getattr(entry["result"], "runtime_seconds", None) is not None
                ],
                dtype=float,
            )
            runtime_summary = f" | cpu_time mean={runtime_values.mean():.6g}s" if runtime_values.size else ""
            rel_improvement = float((baseline_delta_v - delta_v.mean()) / baseline_delta_v)
            if include_variance:
                variance = float(np.var(delta_v, ddof=1)) if delta_v.size > 1 else 0.0
                print(
                    f"{group_name}: n={len(group_entries)} | delta_v mean={delta_v.mean():.6g} variance={variance:.6g}"
                    f" | rel_mean_improvement_vs_{baseline_entry['label']}={rel_improvement:.6g}{runtime_summary}"
                )
            else:
                print(
                    f"{group_name}: delta_v={delta_v.mean():.6g}"
                    f" | rel_improvement_vs_{baseline_entry['label']}={rel_improvement:.6g}{runtime_summary}"
                )
    print("*" * 92)

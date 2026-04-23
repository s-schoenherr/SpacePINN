from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._baseline_summary import get_baseline_entries, group_nonbaseline_entries


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

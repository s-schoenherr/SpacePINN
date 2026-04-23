from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spacepinn
from matplotlib.colors import LogNorm

from spacepinn.experiment import CollectionSpec, PinnEntrySpec, finalize_collection, run_pinn_entry
from spacepinn.paper.low_thrust_transfer import (
    build_config as build_low_thrust_config,
    TARGET_ORBITS,
    OrbitalTransferBC,
    Orbit,
)
from spacepinn.plotting.helpers import register_plot_artifact_if_possible

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "low_thrust_boundary_sweep"
DEFAULT_TOF_SCALES = tuple(np.linspace(1.2, 3.0, 10))
DEFAULT_ALPHA_PI_MULTIPLIERS = tuple(np.linspace(2.5, 7.0, 10))
SMOKE_TOF_SCALES = (1.5, 2.0)
SMOKE_ALPHA_PI_MULTIPLIERS = (4.0, 5.0)
DEFAULT_REFINED_TOF_CENTER = 1.2
DEFAULT_REFINED_ALPHA_CENTER = 2.5
DEFAULT_REFINED_TOF_SPAN = 0.6
DEFAULT_REFINED_ALPHA_SPAN = 1.0
DEFAULT_REFINED_GRID_SIZE = 10
DEFAULT_REFINED_N_ADAM = 50_000


def _parse_float_list(raw: str) -> tuple[float, ...]:
    values = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return tuple(values)


def _parse_args():
    parser = argparse.ArgumentParser(description="Sweep free-final-angle low-thrust settings.")
    parser.add_argument("--target-orbit", choices=sorted(TARGET_ORBITS), default="geo")
    parser.add_argument(
        "--tof-scales",
        type=_parse_float_list,
        default=None,
        help="Comma-separated tof_scale values, for example 1.5,2.0,2.5",
    )
    parser.add_argument(
        "--alpha-pi-multipliers",
        type=_parse_float_list,
        default=None,
        help="Comma-separated multipliers so alpha_N_init = multiplier * pi, for example 3,4,5,6",
    )
    parser.add_argument("--n-adam", type=int, default=None, help="Override Adam iterations for every PINN run.")
    parser.add_argument("--grid-size", type=int, default=DEFAULT_REFINED_GRID_SIZE, help="Grid size per axis for neighborhood sweeps.")
    parser.add_argument("--tof-center", type=float, default=None, help="Center tof_scale for a local neighborhood sweep.")
    parser.add_argument("--tof-span", type=float, default=DEFAULT_REFINED_TOF_SPAN, help="Total tof_scale span around the center.")
    parser.add_argument("--alpha-center", type=float, default=None, help="Center alpha/pi value for a local neighborhood sweep.")
    parser.add_argument("--alpha-span", type=float, default=DEFAULT_REFINED_ALPHA_SPAN, help="Total alpha/pi span around the center.")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def _build_neighborhood(center: float, span: float, grid_size: int) -> tuple[float, ...]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    half_span = 0.5 * float(span)
    return tuple(np.linspace(float(center) - half_span, float(center) + half_span, int(grid_size)))


def _resolve_tof_scales(values: tuple[float, ...] | None) -> tuple[float, ...]:
    if values is not None:
        return tuple(float(value) for value in values)
    if os.getenv("FAST_SMOKE", "0") == "1":
        return SMOKE_TOF_SCALES
    return DEFAULT_TOF_SCALES


def _resolve_alpha_pi_multipliers(values: tuple[float, ...] | None) -> tuple[float, ...]:
    if values is not None:
        return tuple(float(value) for value in values)
    if os.getenv("FAST_SMOKE", "0") == "1":
        return SMOKE_ALPHA_PI_MULTIPLIERS
    return DEFAULT_ALPHA_PI_MULTIPLIERS


def _build_sweep_config(
    *,
    target_orbit: str,
    tof_scale: float,
    alpha_pi_multiplier: float,
    n_adam: int | None = None,
) -> dict:
    alpha_N_initial = float(alpha_pi_multiplier * math.pi)
    config = build_low_thrust_config(
        target_orbit=target_orbit,
        terminal_angle_pi=alpha_pi_multiplier,
        time_guess_scale=tof_scale,
        smoke=False,
    )
    config["label"] = f"PINN with exact BC | tof={tof_scale:.2f} | alpha={alpha_pi_multiplier:.2f}pi"
    config.setdefault("scenario", {})
    config["scenario"]["target_orbit"] = target_orbit
    config["scenario"]["tof_scale"] = float(tof_scale)
    config["scenario"]["alpha_pi_multiplier"] = float(alpha_pi_multiplier)
    config["scenario"]["alpha_N_initial_guess"] = alpha_N_initial
    if n_adam is not None:
        config["optimizer"]["n_adam"] = int(n_adam)
        config["optimizer"]["n_lbfgs"] = 0
    return config


def _final_loss(result) -> float:
    loss = getattr(result, "loss", None) or []
    if len(loss) == 0:
        return float("nan")
    return float(loss[-1])


def _epochs_total(result) -> float:
    value = getattr(result, "epochs_total", None)
    return float("nan") if value is None else float(value)


def _entry_field(entry, field: str, default=None):
    if isinstance(entry, dict):
        return entry.get(field, default)
    return getattr(entry, field, default)


def _result_table(entries: list) -> list[dict]:
    rows = []
    for entry in entries:
        config = _entry_field(entry, "config", {}) or {}
        scenario = config.get("scenario", {})
        result = _entry_field(entry, "result")
        label = _entry_field(entry, "label", "")
        rows.append(
            {
                "label": label,
                "tof_scale": float(scenario.get("tof_scale", float("nan"))),
                "alpha_pi_multiplier": float(scenario.get("alpha_pi_multiplier", float("nan"))),
                "delta_v": float(getattr(result, "delta_v", float("nan"))),
                "final_loss": _final_loss(result),
                "epochs_total": _epochs_total(result),
            }
        )
    return rows


def _summary_lookup(collection_run: dict) -> dict[str, dict]:
    summary = collection_run.get("summary") or {}
    entries = summary.get("entries") or []
    return {entry.get("entry_id"): entry for entry in entries if entry.get("entry_id") is not None}


def _merge_summary_metrics(rows: list[dict], entries: list, summary_by_entry_id: dict[str, dict]) -> list[dict]:
    merged = []
    for row, entry in zip(rows, entries):
        entry_id = _entry_field(entry, "entry_id")
        summary_entry = summary_by_entry_id.get(entry_id, {})
        merged_row = dict(row)
        if math.isnan(merged_row["epochs_total"]):
            summary_epochs = summary_entry.get("epochs_total")
            if summary_epochs is not None:
                merged_row["epochs_total"] = float(summary_epochs)
        if math.isnan(merged_row["final_loss"]):
            summary_final_loss = summary_entry.get("final_loss")
            if summary_final_loss is not None:
                merged_row["final_loss"] = float(summary_final_loss)
        if math.isnan(merged_row["delta_v"]):
            summary_delta_v = summary_entry.get("delta_v")
            if summary_delta_v is not None:
                merged_row["delta_v"] = float(summary_delta_v)
        merged.append(merged_row)
    return merged


def _write_results_csv(rows: list[dict], output_dir: str) -> None:
    output_path = Path(output_dir) / "low_thrust_boundary_sweep.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "tof_scale", "alpha_pi_multiplier", "delta_v", "final_loss", "epochs_total"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _grid_from_rows(rows: list[dict], *, tof_scales: tuple[float, ...], alpha_pi_multipliers: tuple[float, ...], key: str):
    grid = np.full((len(alpha_pi_multipliers), len(tof_scales)), np.nan, dtype=float)
    tof_values = np.asarray(tof_scales, dtype=float)
    alpha_values = np.asarray(alpha_pi_multipliers, dtype=float)
    for row in rows:
        alpha_value = float(row["alpha_pi_multiplier"])
        tof_value = float(row["tof_scale"])
        i = int(np.argmin(np.abs(alpha_values - alpha_value)))
        j = int(np.argmin(np.abs(tof_values - tof_value)))
        grid[i, j] = float(row[key])
    return grid


def _plot_heatmaps(collection_run: dict, *, tof_scales: tuple[float, ...], alpha_pi_multipliers: tuple[float, ...]) -> None:
    output_dir = collection_run["plot_output_dir"]
    rows = _result_table(collection_run["entries"])
    rows = _merge_summary_metrics(rows, collection_run["entries"], _summary_lookup(collection_run))
    _write_results_csv(rows, output_dir)

    figure_path = Path(output_dir) / "low_thrust_boundary_sweep_heatmaps.pdf"
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6))
    metrics = [
        ("final_loss", "Final Loss", "viridis", True),
        ("delta_v", r"$\Delta v$", "magma_r", False),
        ("epochs_total", "Epochs", "plasma_r", False),
    ]

    for ax, (metric_key, title, cmap, use_log) in zip(axes, metrics):
        grid = _grid_from_rows(
            rows,
            tof_scales=tof_scales,
            alpha_pi_multipliers=alpha_pi_multipliers,
            key=metric_key,
        )
        if use_log:
            positive = grid[np.isfinite(grid) & (grid > 0)]
            image = ax.imshow(
                grid,
                aspect="auto",
                origin="lower",
                cmap=cmap,
                norm=LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive))),
            )
        else:
            image = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("tof_scale")
        ax.set_ylabel(r"Initial $\alpha_N / \pi$")
        ax.set_xticks(range(len(tof_scales)))
        ax.set_xticklabels([f"{value:.2f}" for value in tof_scales])
        ax.set_yticks(range(len(alpha_pi_multipliers)))
        ax.set_yticklabels([f"{value:.2f}" for value in alpha_pi_multipliers])
        finite = grid[np.isfinite(grid)]
        grid_min = float(np.min(finite)) if finite.size else 0.0
        grid_max = float(np.max(finite)) if finite.size else 1.0
        for row_idx in range(grid.shape[0]):
            for col_idx in range(grid.shape[1]):
                value = grid[row_idx, col_idx]
                if np.isnan(value):
                    continue
                text = f"{value:.2e}" if metric_key == "final_loss" else f"{value:.1f}"
                normalized = 0.5 if grid_max == grid_min else (float(value) - grid_min) / (grid_max - grid_min)
                text_color = "black" if normalized < 0.35 or normalized > 0.75 else "white"
                ax.text(col_idx, row_idx, text, ha="center", va="center", color=text_color, fontsize=8, fontweight="semibold")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    register_plot_artifact_if_possible(figure_path)
    plt.close(fig)


def _print_summary(collection_run: dict) -> None:
    rows = _result_table(collection_run["entries"])
    rows = _merge_summary_metrics(rows, collection_run["entries"], _summary_lookup(collection_run))
    rows = sorted(rows, key=lambda row: (row["final_loss"], row["delta_v"]))
    print()
    print("[SWINGBY] Best sweep settings by final loss:")
    for row in rows[:5]:
        print(
            "  "
            f"tof_scale={row['tof_scale']:.2f} | "
            f"alpha={row['alpha_pi_multiplier']:.2f}pi | "
            f"final_loss={row['final_loss']:.3e} | "
            f"delta_v={row['delta_v']:.6f} | "
            f"epochs={row['epochs_total']:.0f}"
        )


def main(
    *,
    target_orbit: str = "geo",
    tof_scales: tuple[float, ...] | None = None,
    alpha_pi_multipliers: tuple[float, ...] | None = None,
    n_adam: int | None = None,
    tof_center: float | None = None,
    tof_span: float = DEFAULT_REFINED_TOF_SPAN,
    alpha_center: float | None = None,
    alpha_span: float = DEFAULT_REFINED_ALPHA_SPAN,
    grid_size: int = DEFAULT_REFINED_GRID_SIZE,
    skip_plots: bool = False,
    print_summary: bool = True,
):
    if tof_center is not None:
        resolved_tof_scales = _build_neighborhood(tof_center, tof_span, grid_size)
    else:
        resolved_tof_scales = _resolve_tof_scales(tof_scales)
    if alpha_center is not None:
        resolved_alpha_pi_multipliers = _build_neighborhood(alpha_center, alpha_span, grid_size)
    else:
        resolved_alpha_pi_multipliers = _resolve_alpha_pi_multipliers(alpha_pi_multipliers)

    entries = []
    for alpha_pi_multiplier in resolved_alpha_pi_multipliers:
        for tof_scale in resolved_tof_scales:
            entries.append(
                run_pinn_entry(
                    PinnEntrySpec(
                        config_builder=lambda target_orbit_value=target_orbit, tof_value=float(tof_scale), alpha_value=float(
                            alpha_pi_multiplier
                        ), n_adam_value=n_adam: _build_sweep_config(
                            target_orbit=target_orbit_value,
                            tof_scale=tof_value,
                            alpha_pi_multiplier=alpha_value,
                            n_adam=n_adam_value,
                        )
                    )
                )
            )

    return finalize_collection(
        CollectionSpec(
            label=COLLECTION_LABEL,
            run_root=str(RUN_ROOT),
            entries=entries,
            summary_fn=_print_summary,
            plot_fn=lambda collection_run: _plot_heatmaps(
                collection_run,
                tof_scales=resolved_tof_scales,
                alpha_pi_multipliers=resolved_alpha_pi_multipliers,
            ),
        ),
        skip_plots=skip_plots,
        print_summary=print_summary,
    )


if __name__ == "__main__":
    args = _parse_args()
    main(
        target_orbit=args.target_orbit,
        tof_scales=args.tof_scales,
        alpha_pi_multipliers=args.alpha_pi_multipliers,
        n_adam=args.n_adam,
        tof_center=args.tof_center,
        tof_span=args.tof_span,
        alpha_center=args.alpha_center,
        alpha_span=args.alpha_span,
        grid_size=args.grid_size,
        skip_plots=args.skip_plots,
        print_summary=not args.skip_summary,
    )

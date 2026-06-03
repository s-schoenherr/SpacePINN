from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
from functools import partial
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import spacepinn
import torch

from spacepinn.config.config_2d import ordinary_2d_config
from spacepinn.config.config_3d import ordinary_3d_config
from spacepinn.paper.runtime import smoke_mode_enabled
from spacepinn.plotting.paper_style import PAPER_STYLE
from spacepinn.runner.context import RunCollectionContext
from spacepinn.runner.loading import load_run
from spacepinn.runner import run_experiment

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.unicode_minus": True,
        "font.size": 11,
    }
)

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
N_ADAM = 2_000
N_LBFGS = 100_000
LBFGS_MAX_ITER = 10
CONVERGENCE_THRESHOLD = 1e-12
WEIGHT_MIN = 1e-3
WEIGHT_MAX = 1e3
NUM_WEIGHTS = 1_000
SMOKE_WEIGHTS = np.array([1e-3, 1e-2, 1e-1, 1.0], dtype=float)
NAN_STREAK_LIMIT = 25
PAPER_SWEEP_AXES_RECT = (0.16, 0.12, 0.78, 0.82)


@dataclass(frozen=True)
class SweepSpec:
    label: str
    dimension: int
    dynamic_tof: bool
    selection_metric: str = "min_bc_loss"
    plot_title: str | None = None


def _weights(*, smoke: bool) -> np.ndarray:
    if smoke:
        return SMOKE_WEIGHTS
    return np.exp(np.linspace(np.log(WEIGHT_MIN), np.log(WEIGHT_MAX), NUM_WEIGHTS))


def _base_config(*, dimension: int, dynamic_tof: bool) -> dict[str, Any]:
    if dimension == 2:
        config = deepcopy(ordinary_2d_config)
    elif dimension == 3:
        config = deepcopy(ordinary_3d_config)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")

    if not dynamic_tof:
        config.pop("extra_parameters", None)

    config["optimizer"]["n_adam"] = N_ADAM
    config["optimizer"]["n_lbfgs"] = N_LBFGS
    config["optimizer"]["opt_lbfgs"] = partial(torch.optim.LBFGS, max_iter=LBFGS_MAX_ITER, lr=0.1)
    config["optimizer"]["convergence_threshold"] = CONVERGENCE_THRESHOLD
    config["optimizer"]["show_progress"] = False
    config["optimizer"]["progress_print_interval"] = 0
    config["plotting"] = {}
    return config


def _smoke_runtime_mutation(config: dict[str, Any]) -> None:
    config["optimizer"]["n_adam"] = 1
    config["optimizer"]["n_lbfgs"] = 0
    config["optimizer"]["progress_print_interval"] = 0


def _run_single_weight(
    *,
    dimension: int,
    dynamic_tof: bool,
    lambda_bc: float,
    smoke: bool,
    run_root: str | None = None,
) -> dict[str, Any]:
    config = _base_config(dimension=dimension, dynamic_tof=dynamic_tof)
    config["label"] = f"oPINN | lambda_bc={lambda_bc:.6g}"
    config["optimizer"]["w_bc"] = float(lambda_bc)
    if smoke:
        _smoke_runtime_mutation(config)

    try:
        run = run_experiment(config, run_root=run_root or str(RUN_ROOT))
        result = run["result"]
        loss = np.asarray(result.loss, dtype=float) if getattr(result, "loss", None) else np.asarray([], dtype=float)
        loss_bc = (
            np.asarray(result.loss_bc, dtype=float) if getattr(result, "loss_bc", None) else np.asarray([], dtype=float)
        )

        nonfinite_loss = loss.size > 0 and (not np.all(np.isfinite(loss)))
        nonfinite_loss_bc = loss_bc.size > 0 and (not np.all(np.isfinite(loss_bc)))
        had_nan = bool(nonfinite_loss or nonfinite_loss_bc)

        min_total_loss = float(np.nanmin(loss)) if loss.size > 0 and np.isfinite(loss).any() else None
        min_bc_loss = float(np.nanmin(loss_bc)) if loss_bc.size > 0 and np.isfinite(loss_bc).any() else None
        final_total_loss = float(loss[-1]) if loss.size > 0 and np.isfinite(loss[-1]) else None
        final_bc_loss = float(loss_bc[-1]) if loss_bc.size > 0 and np.isfinite(loss_bc[-1]) else None

        return {
            "lambda_bc": float(lambda_bc),
            "status": "ok",
            "had_nan": had_nan,
            "min_total_loss": min_total_loss,
            "min_bc_loss": min_bc_loss,
            "final_total_loss": final_total_loss,
            "final_bc_loss": final_bc_loss,
            "delta_v": float(result.delta_v),
            "t_total": float(result.t_total),
            "runtime_seconds": getattr(result, "runtime_seconds", None),
            "epochs_total": len(result.loss) if getattr(result, "loss", None) else 0,
            "run_id": run.get("run_id"),
            "run_dir": run.get("run_dir"),
        }
    except Exception as exc:
        return {
            "lambda_bc": float(lambda_bc),
            "status": "error",
            "had_nan": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "min_total_loss": None,
            "min_bc_loss": None,
            "final_total_loss": None,
            "final_bc_loss": None,
            "delta_v": None,
            "t_total": None,
            "runtime_seconds": None,
            "epochs_total": 0,
            "run_id": None,
            "run_dir": None,
        }


def _run_single_weight_star(args):
    return _run_single_weight(**args)


def _is_nan_result(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return True
    if row.get("had_nan"):
        return True
    return row.get("min_total_loss") is None and row.get("min_bc_loss") is None


def _best_row(rows: list[dict[str, Any]], *, selection_metric: str) -> dict[str, Any] | None:
    finite_rows = [row for row in rows if not _is_nan_result(row)]
    if not finite_rows:
        return None

    if selection_metric == "min_bc_loss":
        keyed = [
            (
                math.inf if row["min_bc_loss"] is None else float(row["min_bc_loss"]),
                math.inf if row["min_total_loss"] is None else float(row["min_total_loss"]),
                float(row["lambda_bc"]),
                row,
            )
            for row in finite_rows
        ]
        return min(keyed, key=lambda item: item[:3])[-1]

    keyed = [
        (
            math.inf if row["min_total_loss"] is None else float(row["min_total_loss"]),
            math.inf if row["min_bc_loss"] is None else float(row["min_bc_loss"]),
            float(row["lambda_bc"]),
            row,
        )
        for row in finite_rows
    ]
    return min(keyed, key=lambda item: item[:3])[-1]


def _plot_results(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    selection_metric: str,
    title: str,
    paper_style: bool = False,
    best_label_precision: int = 4,
) -> None:
    finite_rows = [row for row in rows if not _is_nan_result(row)]
    if not finite_rows:
        return

    lambda_values = np.array([float(row["lambda_bc"]) for row in finite_rows], dtype=float)
    min_total_loss = np.array([float(row["min_total_loss"]) for row in finite_rows], dtype=float)
    min_bc_loss = np.array(
        [np.nan if row["min_bc_loss"] is None else float(row["min_bc_loss"]) for row in finite_rows],
        dtype=float,
    )
    best = _best_row(finite_rows, selection_metric=selection_metric)

    if paper_style:
        fig = plt.figure(figsize=PAPER_STYLE.figure_size)
        ax = fig.add_axes(PAPER_SWEEP_AXES_RECT)
        marker_size = 22
        best_marker_size = 72
    else:
        fig, ax = plt.subplots(figsize=(6.1, 4.6))
        marker_size = 18
        best_marker_size = 60

    ax.scatter(lambda_values, min_total_loss, color="#1f77b4", s=marker_size, label="Smallest total loss")
    if np.isfinite(min_bc_loss).any():
        ax.scatter(lambda_values, min_bc_loss, color="#ff7f0e", s=marker_size, label="Smallest BC loss")

    if best is not None:
        best_y = best["min_bc_loss"] if selection_metric == "min_bc_loss" else best["min_total_loss"]
        if best_y is not None:
            ax.scatter(
                float(best["lambda_bc"]),
                float(best_y),
                marker="x",
                color="red",
                s=best_marker_size,
                label=rf"Best $\lambda_{{BC}} = {float(best['lambda_bc']):.{best_label_precision}g}$",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Boundary loss weight $\lambda_{BC}$")
    ax.set_ylabel("Smallest loss in training process")
    if title:
        ax.set_title(title, fontsize=PAPER_STYLE.title_fontsize if paper_style else None)
    if paper_style:
        ax.xaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
        ax.yaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
        ax.tick_params(axis="both", which="major", labelsize=PAPER_STYLE.tick_label_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=PAPER_STYLE.tick_label_fontsize)
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=PAPER_STYLE.legend_fontsize + 2.0,
            framealpha=PAPER_STYLE.legend_framealpha,
            facecolor="white",
            edgecolor="black",
            handlelength=PAPER_STYLE.legend_handlelength,
        )
        legend.get_frame().set_linewidth(1.0)
        fig.savefig(output_path, bbox_inches=PAPER_STYLE.save_bbox_inches, pad_inches=PAPER_STYLE.save_pad_inches)
    else:
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_aggregate_summary(
    *,
    spec: SweepSpec,
    rows: list[dict[str, Any]],
    weights: np.ndarray,
    workers: int,
    smoke: bool,
    nan_streak_limit: int,
    aborted_early: bool,
    best: dict[str, Any] | None,
    collection_context: RunCollectionContext,
) -> dict[str, Any]:
    return {
        "label": spec.label,
        "run_id": collection_context.run_id,
        "run_dir": str(collection_context.run_dir),
        "dimension": spec.dimension,
        "dynamic_tof": spec.dynamic_tof,
        "selection_metric": spec.selection_metric,
        "plot_title": spec.plot_title,
        "n_adam": 1 if smoke else N_ADAM,
        "n_lbfgs": 0 if smoke else N_LBFGS,
        "lbfgs_max_iter": LBFGS_MAX_ITER,
        "convergence_threshold": CONVERGENCE_THRESHOLD,
        "weight_min": float(weights[0]),
        "weight_max": float(weights[min(len(weights) - 1, len(rows) - 1)]),
        "num_requested_weights": int(len(weights)),
        "num_completed_weights": int(len(rows)),
        "workers": workers,
        "aborted_early": aborted_early,
        "nan_streak_limit": nan_streak_limit,
        "best_row": best,
        "rows": rows,
    }


def run_boundary_weight_sweep(
    *,
    spec: SweepSpec,
    workers: int = 1,
    nan_streak_limit: int = NAN_STREAK_LIMIT,
) -> dict[str, Any]:
    smoke = smoke_mode_enabled()
    weights = _weights(smoke=smoke)
    collection_context = RunCollectionContext(label=spec.label, run_root=str(RUN_ROOT))
    collection_context.start()
    worker_run_root = collection_context.run_dir / "_worker_runs"
    worker_run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    consecutive_nan = 0
    aborted_early = False
    workers = max(1, int(workers))
    batch_size = 1 if workers == 1 else workers

    def process_batch(batch_weights: np.ndarray, pool: Any | None = None) -> list[dict[str, Any]]:
        args = [
            {
                "dimension": spec.dimension,
                "dynamic_tof": spec.dynamic_tof,
                "lambda_bc": float(weight),
                "smoke": smoke,
                "run_root": str(worker_run_root),
            }
            for weight in batch_weights
        ]
        if pool is None:
            return [_run_single_weight_star(arg) for arg in args]
        return pool.map(_run_single_weight_star, args)

    def consume_batches(pool: Any | None = None) -> None:
        nonlocal consecutive_nan, aborted_early
        for start in range(0, len(weights), batch_size):
            batch_rows = process_batch(weights[start : start + batch_size], pool=pool)
            for row in batch_rows:
                rows.append(row)
                if _is_nan_result(row):
                    consecutive_nan += 1
                else:
                    consecutive_nan = 0
            if consecutive_nan >= nan_streak_limit:
                aborted_early = True
                break

    try:
        if workers == 1:
            consume_batches()
        else:
            with mp.get_context("spawn").Pool(processes=workers) as pool:
                consume_batches(pool=pool)

        for row in rows:
            if row.get("status") != "ok" or row.get("run_dir") is None:
                continue
            run_dir = Path(str(row["run_dir"]))
            if not (run_dir / "manifest.json").exists():
                continue
            loaded = load_run(str(run_dir))
            if not loaded.get("entries"):
                continue
            loaded_entry = loaded["entries"][0]
            collection_context.add_entry(
                label=loaded_entry["label"],
                result=loaded_entry["result"],
                config=loaded_entry.get("config"),
                model=None,
                source=loaded_entry.get("source", "pinn"),
            )

        shutil.rmtree(worker_run_root, ignore_errors=True)

        best = _best_row(rows, selection_metric=spec.selection_metric)
        csv_path = collection_context.run_dir / "sweep_results.csv"
        plot_path = collection_context.run_dir / "sweep_plot.pdf"

        _write_csv(rows, csv_path)
        _plot_results(
            rows,
            output_path=plot_path,
            selection_metric=spec.selection_metric,
            title=spec.plot_title or spec.label.replace("_", " "),
        )

        aggregate_summary = _build_aggregate_summary(
            spec=spec,
            rows=rows,
            weights=weights,
            workers=workers,
            smoke=smoke,
            nan_streak_limit=nan_streak_limit,
            aborted_early=aborted_early,
            best=best,
            collection_context=collection_context,
        )

        collection_context.finalize_success()
        collection_context.aggregate_summary_path.write_text(
            json.dumps(aggregate_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        collection_context.register_artifact(csv_path, kind="sweep_results_csv")
        if plot_path.exists():
            collection_context.register_artifact(plot_path, kind="sweep_plot_pdf")
        collection_context.register_artifact(collection_context.aggregate_summary_path, kind="aggregate_summary")

        manifest = json.loads(collection_context.manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("paths", {})["aggregate_summary"] = str(collection_context.aggregate_summary_path)
        manifest["paths"]["sweep_results_csv"] = str(csv_path)
        manifest["paths"]["sweep_plot_pdf"] = str(plot_path)
        collection_context.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        print()
        print(f"[SWEEP] {spec.label}")
        print(f"output_dir: {collection_context.run_dir}")
        print(f"completed_weights: {len(rows)}/{len(weights)}")
        if aborted_early:
            print(f"aborted_early: true (nan_streak_limit={nan_streak_limit})")
        if best is not None:
            metric_label = "min BC loss" if spec.selection_metric == "min_bc_loss" else "min total loss"
            metric_value = best["min_bc_loss"] if spec.selection_metric == "min_bc_loss" else best["min_total_loss"]
            print(
                f"best lambda_BC={float(best['lambda_bc']):.6g} "
                f"| {metric_label}={metric_value:.6g}"
            )

        return {
            "label": spec.label,
            "run_id": collection_context.run_id,
            "run_dir": str(collection_context.run_dir),
            "output_dir": str(collection_context.run_dir),
            "csv_path": str(csv_path),
            "aggregate_summary_path": str(collection_context.aggregate_summary_path),
            "plot_path": str(plot_path),
            "rows": rows,
            "best_row": best,
            "aborted_early": aborted_early,
        }
    except Exception as error:
        shutil.rmtree(worker_run_root, ignore_errors=True)
        collection_context.finalize_failure(error)
        raise

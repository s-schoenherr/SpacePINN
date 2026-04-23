from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import spacepinn
import torch

from spacepinn.config.config_3d import exact_bc_3d_config
from spacepinn.opengoddard.fixed_tof_3d_goddard import fixed_tof_3d_goddard
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_time_of_flight_variants_3d"
FIG_PREFIX = COLLECTION_LABEL
TIME_OF_FLIGHTS = [1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
OPEN_GODDARD_TOFS = (1.0, 2.0, 3.0)
OPEN_GODDARD_COLORS = {1.0: "#475569", 2.0: "#8b5cf6", 3.0: "#c4b5fd"}


def _build_config(t_total: float, *, smoke: bool = False) -> dict:
    config = deepcopy(exact_bc_3d_config)
    config["label"] = rf"PINN with exact BC | $T={t_total:.2f}\,\mathrm{{s}}$"
    config["optimizer"]["n_adam"] = 1_000
    config["optimizer"]["n_lbfgs"] = 10_000
    config["extra_parameters"] = {}
    config["optimizer"]["t_total"] = torch.tensor(t_total, requires_grad=False)
    config["plotting"].pop("color", None)
    if smoke:
        config["optimizer"]["n_adam"] = 1
        config["optimizer"]["n_lbfgs"] = 0
    return config


def _build_additional_entry(result_entry: dict) -> dict:
    return {
        "label": result_entry["label"],
        "result": result_entry["result"],
        "model": result_entry.get("model"),
        "config": result_entry.get("config"),
        "plotting": {key: result_entry[key] for key in ("linestyle", "color", "quiver_scale") if key in result_entry},
        "source": "opengoddard",
        "log_text": result_entry.get("log_text"),
        "log_filename": result_entry.get("log_filename"),
    }


def build_configs(*, smoke: bool | None = None) -> list[dict]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    tofs = TIME_OF_FLIGHTS[:2] if smoke_enabled else TIME_OF_FLIGHTS
    return [_build_config(float(t_total), smoke=smoke_enabled) for t_total in tofs]


def build_additional_entries(*, smoke: bool | None = None) -> list[dict]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    tofs = TIME_OF_FLIGHTS[:2] if smoke_enabled else TIME_OF_FLIGHTS
    entries = []
    for t_total in tofs:
        if t_total not in OPEN_GODDARD_TOFS:
            continue
        result_entry = capture_baseline_entry(
            lambda t=float(t_total): fixed_tof_3d_goddard(
                TOF=t,
                color=OPEN_GODDARD_COLORS[t],
                linestyle="--",
            ),
            log_filename=f"fixed_tof_{float(t_total):.2f}_opengoddard.log",
        )
        entries.append(_build_additional_entry(result_entry))
    return entries


def main(*, skip_plots: bool = False, print_summary: bool = True, smoke: bool | None = None):
    collection_run = run_experiment_collection(
        configs=build_configs(smoke=smoke),
        additional_entries=build_additional_entries(smoke=smoke),
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        plotter = TrajectoryPlotter(
            collection_run["entries"],
            fig_prefix=FIG_PREFIX,
            dim=3,
            figsize=(7, 7),
            output_dir=collection_run["plot_output_dir"],
        )
        plotter.plot_traj_2d(plot_quiver=False)
        plotter.plot_traj_3d(plot_quiver=False)
        plotter.plot_loss()
        plotter.plot_thrust()
        plotter.plot_gravity(legend_mode="compact")

    return collection_run


if __name__ == "__main__":
    main(smoke=(os.getenv("FAST_SMOKE", "0") == "1"))

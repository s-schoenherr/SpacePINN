from __future__ import annotations

import os
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import spacepinn
import torch

from spacepinn.config.config_3d import exact_bc_3d_config
from spacepinn.config.transform_functions import position_fn
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_boundary_condition_variants_3d"
FIG_PREFIX = COLLECTION_LABEL
AMPLITUDES = np.linspace(0.0, 1.0, 11)


def _x0(amplitude: float) -> torch.Tensor:
    return torch.tensor([[-1.0, -1.0, amplitude]], dtype=torch.float32)


def _xN(amplitude: float) -> torch.Tensor:
    return torch.tensor([[1.0, 1.0, -amplitude]], dtype=torch.float32)


def _build_config(amplitude: float, *, smoke: bool = False) -> dict:
    config = deepcopy(exact_bc_3d_config)
    config["label"] = f"Boundary variation A={amplitude:.1f}"
    config["optimizer"]["n_adam"] = 1_000
    config["optimizer"]["n_lbfgs"] = 10_000
    config["plotting"].pop("color", None)

    x0 = _x0(amplitude)
    xN = _xN(amplitude)
    config["pinn"]["output_transform_fn"] = partial(position_fn, x0=x0, xN=xN)
    config["optimizer"]["r0"] = x0
    config["optimizer"]["rN"] = xN
    config.setdefault("extra_parameters", {})["t_total"] = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

    if smoke:
        config["optimizer"]["n_adam"] = 1
        config["optimizer"]["n_lbfgs"] = 0
    return config


def build_configs(*, smoke: bool | None = None) -> list[dict]:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    amplitudes = AMPLITUDES[:2] if smoke_enabled else AMPLITUDES
    return [_build_config(float(amplitude), smoke=smoke_enabled) for amplitude in amplitudes]


def main(*, skip_plots: bool = False, print_summary: bool = True, smoke: bool | None = None):
    collection_run = run_experiment_collection(
        configs=build_configs(smoke=smoke),
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
    )

    if print_summary:
        print_collection_run_summary(collection_run)

    if not skip_plots:
        plotter = TrajectoryPlotter(
            collection_run["entries"],
            dim=3,
            figsize=(7, 7),
            fig_prefix=FIG_PREFIX,
            output_dir=collection_run["plot_output_dir"],
        )
        plotter.plot_traj_2d(plot_quiver=False)
        plotter.plot_traj_3d(plot_quiver=False)
        plotter.plot_loss()
        plotter.plot_thrust(plot_legend=False)
        plotter.plot_gravity()

    return collection_run


if __name__ == "__main__":
    main(smoke=(os.getenv("FAST_SMOKE", "0") == "1"))

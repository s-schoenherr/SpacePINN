from __future__ import annotations

import os
from copy import deepcopy
from functools import partial
from pathlib import Path

import spacepinn
import torch

from spacepinn.config.config_3d import exact_bc_3d_config, pretraining_3d_config
from spacepinn.config.shared_parameters import x0_3d, xN_3d
from spacepinn.config.transform_functions import kinematic_fn
from spacepinn.experiment import (
    CollectionSpec,
    PinnEntrySpec,
    build_pretrained_model,
    finalize_collection,
    run_pinn_entry,
)
from spacepinn.plotter import TrajectoryPlotter

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
PLANE_VELOCITY = torch.tensor([[1.0, 1.0, 0.0]])


def _build_kinematic_pretrain_config() -> dict:
    config = deepcopy(pretraining_3d_config)
    config["label"] = "Kinematic tPINN pretrain"
    config["optimizer"]["n_adam"] = 2_000
    config["optimizer"]["n_lbfgs"] = 0
    config["plotting"]["linestyle"] = "solid"
    config["pinn"]["output_transform_fn"] = partial(
        kinematic_fn,
        x0=x0_3d,
        xN=xN_3d,
        v0=PLANE_VELOCITY,
        vN=PLANE_VELOCITY,
    )
    return config


def _build_geometric_finetune_config(*, initial_t_total: float) -> dict:
    config = deepcopy(exact_bc_3d_config)
    config["label"] = "Geometric tPINN from kinematic pretrain"
    config["optimizer"]["n_adam"] = 2_000
    config["optimizer"]["n_lbfgs"] = 10_000
    config["extra_parameters"]["t_total"] = torch.nn.Parameter(
        torch.tensor(float(initial_t_total), requires_grad=True)
    )
    config["plotting"]["linestyle"] = "solid"
    return config


def _plot_collection(collection_run: dict) -> None:
    plotter = TrajectoryPlotter(
        collection_run["entries"],
        dim=3,
        figsize=(6.1, 6.1),
        fig_prefix="kinematic_to_geometric_pretraining_3d",
        output_dir=collection_run["plot_output_dir"],
    )
    plotter.plot_traj_2d(plot_quiver=False)
    plotter.plot_traj_3d(plot_quiver=False)
    plotter.plot_loss()
    plotter.plot_thrust()
    plotter.plot_gravity(legend_mode="compact")


def _prepend_pretraining_history(*, pretrain_result, finetune_result) -> None:
    finetune_result.history.loss = [*pretrain_result.loss, *finetune_result.loss]
    finetune_result.history.loss_physics = [*pretrain_result.loss_physics, *finetune_result.loss_physics]
    finetune_result.history.loss_bc = [*pretrain_result.loss_bc, *finetune_result.loss_bc]
    finetune_result._sync_legacy_attributes()


def run_kinematic_to_geometric_entries(*, smoke_mode: bool = False):
    def build_kinematic_pretrain_config() -> dict:
        config = _build_kinematic_pretrain_config()
        if smoke_mode:
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0
        return config

    kinematic_entry = run_pinn_entry(PinnEntrySpec(config_builder=build_kinematic_pretrain_config))

    def build_geometric_finetune_config() -> dict:
        config = _build_geometric_finetune_config(initial_t_total=float(kinematic_entry.result.t_total))
        if smoke_mode:
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0
        return config

    geometric_entry = run_pinn_entry(
        PinnEntrySpec(
            config_builder=build_geometric_finetune_config,
            model_factory=lambda config_runtime, source_model=kinematic_entry.model: build_pretrained_model(
                config_runtime,
                source_model,
            ),
        )
    )
    _prepend_pretraining_history(pretrain_result=kinematic_entry.result, finetune_result=geometric_entry.result)

    return kinematic_entry, geometric_entry


def main(*, skip_plots: bool = False, print_summary: bool = True):
    smoke_mode = os.getenv("FAST_SMOKE", "0") == "1"
    kinematic_entry, geometric_entry = run_kinematic_to_geometric_entries(smoke_mode=smoke_mode)

    return finalize_collection(
        CollectionSpec(
            label="kinematic_to_geometric_pretraining_3d",
            run_root=str(RUN_ROOT),
            entries=[kinematic_entry, geometric_entry],
            plot_fn=_plot_collection,
        ),
        skip_plots=skip_plots,
        print_summary=print_summary,
    )


if __name__ == "__main__":
    if os.getenv("FAST_SMOKE", "0") == "1":
        print("FAST_SMOKE=1")
    run = main()

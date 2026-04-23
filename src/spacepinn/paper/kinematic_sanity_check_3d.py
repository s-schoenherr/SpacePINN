import os
from copy import deepcopy
from functools import partial
from pathlib import Path

import spacepinn
import torch

from spacepinn.config.config_3d import pretraining_3d_config, exact_bc_3d_config
from spacepinn.config.shared_parameters import x0_3d, xN_3d
from spacepinn.config.transform_functions import kinematic_fn
from spacepinn.experiment import (
    CollectionSpec,
    ExternalEntrySpec,
    PinnEntrySpec,
    finalize_collection,
    prepare_external_entry,
    run_pinn_entry,
)
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.opengoddard.kinematic_sanity_check_3d_goddard import (
    kinematic_sanity_check_3d_goddard,
)
from spacepinn.plotter import TrajectoryPlotter

RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"


def _build_position_config() -> dict:
    config = deepcopy(exact_bc_3d_config)
    config["optimizer"]["n_adam"] = 2_000
    return config


def _build_kinematic_config(v0: torch.Tensor, vN: torch.Tensor) -> dict:
    config = deepcopy(pretraining_3d_config)
    config["label"] = "Kinematic tPINN"
    config["optimizer"]["n_adam"] = 2_000
    config["pinn"]["output_transform_fn"] = partial(
        kinematic_fn,
        x0=x0_3d,
        xN=xN_3d,
        v0=v0,
        vN=vN,
    )
    return config


def _boundary_velocity_physical(result) -> tuple[torch.Tensor, torch.Tensor]:
    runtime_dtype = torch.tensor(0.0).to(dtype=torch.get_default_dtype()).dtype
    v0 = torch.as_tensor(result.v[0, :], dtype=runtime_dtype).view(1, -1)
    vN = torch.as_tensor(result.v[-1, :], dtype=runtime_dtype).view(1, -1)
    return v0, vN


def _print_boundary_velocity_comparison(position_result, kinematic_result) -> None:
    v0_pos = position_result.v[0, :]
    vN_pos = position_result.v[-1, :]
    v0_kin = kinematic_result.v[0, :]
    vN_kin = kinematic_result.v[-1, :]

    print("Position PINN boundary velocities:")
    print(f"  v0: {v0_pos}")
    print(f"  vN: {vN_pos}\n")
    print("Kinematic PINN boundary velocities:")
    print(f"  v0: {v0_kin}")
    print(f"  vN: {vN_kin}\n")
    print("Difference in boundary velocities:")
    print(f"  delta v0: {v0_kin - v0_pos}")
    print(f"  delta vN: {vN_kin - vN_pos}")


def _print_direct_collocation_boundary_velocity(position_result, direct_collocation_result) -> None:
    v0_pos = position_result.v[0, :]
    vN_pos = position_result.v[-1, :]
    v0_colloc = direct_collocation_result.v[0, :]
    vN_colloc = direct_collocation_result.v[-1, :]

    print("\nDirect collocation boundary velocities:")
    print(f"  v0: {v0_colloc}")
    print(f"  vN: {vN_colloc}\n")
    print("Difference between direct collocation and Geometric tPINN boundary velocities:")
    print(f"  delta v0: {v0_colloc - v0_pos}")
    print(f"  delta vN: {vN_colloc - vN_pos}")


def _plot_collection(collection_run: dict) -> None:
    plotter = TrajectoryPlotter(
        collection_run["entries"],
        dim=3,
        figsize=(6.1, 6.1),
        fig_prefix="kinematic_sanity_check_3d",
        output_dir=collection_run["plot_output_dir"],
    )
    plotter.plot_traj_2d(plot_quiver=False)
    plotter.plot_traj_3d(plot_quiver=False)
    plotter.plot_loss()
    plotter.plot_thrust()
    plotter.plot_gravity(legend_mode="compact")


def main(*, skip_plots: bool = False, print_summary: bool = True):
    smoke_mode = os.getenv("FAST_SMOKE", "0") == "1"

    def build_position_config() -> dict:
        config = _build_position_config()
        if smoke_mode:
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0
        return config

    position_entry = run_pinn_entry(PinnEntrySpec(config_builder=build_position_config))
    v0_tau, vN_tau = _boundary_velocity_physical(position_entry.result)

    def build_kinematic_config() -> dict:
        config = _build_kinematic_config(v0=v0_tau, vN=vN_tau)
        if smoke_mode:
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0
        return config

    kinematic_entry = run_pinn_entry(
        PinnEntrySpec(
            config_builder=build_kinematic_config,
        )
    )
    direct_collocation_payload = capture_baseline_entry(
        lambda: kinematic_sanity_check_3d_goddard(
            "Direct collocation",
            warm_start_result=position_entry.result,
            max_iteration=1 if smoke_mode else 5,
        ),
        log_filename="direct_collocation_opengoddard.log",
    )
    direct_collocation_entry = prepare_external_entry(
        ExternalEntrySpec(
            label=direct_collocation_payload["label"],
            result=direct_collocation_payload["result"],
            model=direct_collocation_payload.get("model"),
            config=direct_collocation_payload.get("config"),
            plotting={
                key: direct_collocation_payload[key]
                for key in ("linestyle", "color", "quiver_scale")
                if key in direct_collocation_payload
            },
            source="opengoddard",
            log_text=direct_collocation_payload.get("log_text"),
            log_filename=direct_collocation_payload.get("log_filename"),
        )
    )

    return finalize_collection(
        CollectionSpec(
            label="kinematic_sanity_check_3d",
            run_root=str(RUN_ROOT),
            entries=[position_entry, kinematic_entry, direct_collocation_entry],
            summary_fn=lambda _run: (
                _print_boundary_velocity_comparison(position_entry.result, kinematic_entry.result),
                _print_direct_collocation_boundary_velocity(position_entry.result, direct_collocation_entry.result),
            ),
            plot_fn=_plot_collection,
        ),
        skip_plots=skip_plots,
        print_summary=print_summary,
    )


if __name__ == "__main__":
    if os.getenv("FAST_SMOKE", "0") == "1":
        print("FAST_SMOKE=1")
    run = main()

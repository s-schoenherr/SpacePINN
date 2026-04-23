from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import spacepinn
from spacepinn.config.config_3d import exact_bc_3d_config, soft_bc_3d_config
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import paper_baseline_solver_kwargs
from spacepinn.paper._baseline_summary import print_baseline_delta_v_summary
from spacepinn.pretraining.kinematic_to_geometric_pretraining_3d import (
    run_kinematic_to_geometric_entries,
)
from spacepinn.opengoddard.geometric_3d_goddard import geometric_3d_opengoddard
from spacepinn.plotting.helpers import register_plot_artifact_if_possible
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

DTYPE = "float32"
RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_3d"
FIG_PREFIX = "swingby_3d"
BASELINE_LABEL = "Baseline (OpenGoddard)"
ORDINARY_LABEL = "PINN without exact BC"
GEOMETRIC_LABEL = "PINN with exact BC"
GEOMETRIC_WARMSTART_LABEL = "PINN with exact BC and pre-conditioning"
DIRECT_COLLOCATION_COLOR = PALETTE["opengoddard"]
MAIN_LINEWIDTH = 2.4
SECONDARY_LINEWIDTH = 2.0
MAIN_FIGSIZE = (7.0, 6.2)
LOSS_FIGSIZE = MAIN_FIGSIZE
ORDINARY_LAMBDA_BC = 0.42133217438472903


def build_configs(*, smoke: bool | None = None) -> list[dict]:
    geometric_runtime_config = deepcopy(exact_bc_3d_config)
    ordinary_runtime_config = deepcopy(soft_bc_3d_config)

    geometric_runtime_config["label"] = GEOMETRIC_LABEL
    ordinary_runtime_config["label"] = ORDINARY_LABEL
    geometric_runtime_config["numeric_dtype"] = DTYPE
    ordinary_runtime_config["numeric_dtype"] = DTYPE
    geometric_runtime_config["optimizer"]["n_adam"] = 2_000
    ordinary_runtime_config["optimizer"]["n_adam"] = 2_000
    ordinary_runtime_config["plotting"]["linestyle"] = "solid"
    ordinary_runtime_config["plotting"]["trajectory_linestyle"] = "solid"
    ordinary_runtime_config["optimizer"]["w_bc"] = ORDINARY_LAMBDA_BC

    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    if smoke_enabled:
        for config in (geometric_runtime_config, ordinary_runtime_config):
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0

    return [geometric_runtime_config, ordinary_runtime_config]


def build_baseline_entry(*, smoke: bool | None = None) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    conventional_result = geometric_3d_opengoddard(
        BASELINE_LABEL,
        **paper_baseline_solver_kwargs(smoke_enabled=smoke_enabled),
    )
    conventional_result["color"] = DIRECT_COLLOCATION_COLOR
    conventional_result["linestyle"] = "solid"
    conventional_result["trajectory_linestyle"] = "solid"
    return {
        "label": conventional_result["label"],
        "result": conventional_result["result"],
        "model": conventional_result.get("model"),
        "config": conventional_result.get("config"),
        "plotting": {
            key: conventional_result[key]
            for key in ("linestyle", "trajectory_linestyle", "color", "quiver_scale")
            if key in conventional_result
        },
        "source": "opengoddard",
    }


def build_warmstart_entry(*, smoke: bool | None = None) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    _, geometric_finetune_entry = run_kinematic_to_geometric_entries(smoke_mode=smoke_enabled)
    geometric_finetune_entry.label = GEOMETRIC_WARMSTART_LABEL
    geometric_finetune_entry.plotting["color"] = PALETTE["kinematic"]
    geometric_finetune_entry.plotting["linestyle"] = "solid"
    geometric_finetune_entry.plotting["trajectory_linestyle"] = "solid"
    return {
        "label": geometric_finetune_entry.label,
        "result": geometric_finetune_entry.result,
        "model": geometric_finetune_entry.model,
        "config": geometric_finetune_entry.config,
        "plotting": dict(geometric_finetune_entry.plotting),
        "source": geometric_finetune_entry.source,
    }


def _resolve_loss_color(label: str, fallback: str) -> str:
    if label == GEOMETRIC_LABEL:
        return PALETTE["position"]
    if label == ORDINARY_LABEL:
        return PALETTE["vanilla"]
    if label == GEOMETRIC_WARMSTART_LABEL:
        return PALETTE["kinematic"]
    return fallback


def _resolve_loss_label(label: str, suffix: str) -> str:
    if label == GEOMETRIC_LABEL:
        return f"Exact BC: {suffix}"
    if label == ORDINARY_LABEL:
        return f"Without exact BC: {suffix}"
    if label == GEOMETRIC_WARMSTART_LABEL:
        return f"Exact BC + pre-conditioning: {suffix}"
    return f"{label} {suffix}"


def plot_loss_figure(entries: list[dict], *, output_dir: str) -> None:
    loss_entries = [entry for entry in entries if entry["label"] != BASELINE_LABEL]
    plotter = TrajectoryPlotter(
        loss_entries,
        dim=3,
        figsize=LOSS_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = MAIN_LINEWIDTH

    fig, ax = plt.subplots(figsize=LOSS_FIGSIZE)
    for label, exp in plotter.experiments.items():
        result = exp["result"]
        color = _resolve_loss_color(label, exp["color"])
        ax.plot(
            result.loss,
            linestyle="solid",
            label=_resolve_loss_label(label, "total"),
            color=color,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        if result.loss_bc:
            ax.plot(
                result.loss_bc,
                linestyle="--",
                label=_resolve_loss_label(label, r"$\lambda_{BC}L_{BC}$"),
                color=color,
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )
        if result.loss_physics:
            ax.plot(
                result.loss_physics,
                linestyle="-.",
                label=_resolve_loss_label(label, r"$\lambda_{P}L_{P}$"),
                color=color,
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )

    visible_lengths = [len(entry["result"].loss) for entry in loss_entries]
    if visible_lengths:
        ax.set_xlim(0, max(visible_lengths))
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(
        loc="upper right",
        ncol=1,
        framealpha=0.58,
        facecolor="white",
        edgecolor="0.3",
        fontsize=9,
    )
    fig.tight_layout()
    figure_path = plotter._build_figure_path("loss")
    fig.savefig(
        figure_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def main(*, skip_plots: bool = False, print_summary: bool = True, smoke: bool | None = None):
    collection_run = run_experiment_collection(
        configs=build_configs(smoke=smoke),
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
        additional_entries=[
            build_warmstart_entry(smoke=smoke),
            capture_baseline_entry(lambda: build_baseline_entry(smoke=smoke), log_filename="baseline_opengoddard.log"),
        ],
    )

    if print_summary:
        print_collection_run_summary(collection_run)
        print_baseline_delta_v_summary(
            collection_run["entries"],
            title=COLLECTION_LABEL,
            baseline_labels=(BASELINE_LABEL,),
        )

    if not skip_plots:
        plotter = TrajectoryPlotter(
            collection_run["entries"],
            dim=3,
            figsize=MAIN_FIGSIZE,
            fig_prefix=FIG_PREFIX,
            output_dir=collection_run["plot_output_dir"],
        )
        plotter.main_linewidth = MAIN_LINEWIDTH
        plotter.secondary_linewidth = SECONDARY_LINEWIDTH
        plotter.plot_traj_2d(plot_quiver=False)
        plotter.plot_traj_3d(plot_quiver=False)
        plotter.plot_thrust()
        plotter.plot_gravity(legend_mode="compact")

        plot_loss_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
    return collection_run


if __name__ == "__main__":
    main()

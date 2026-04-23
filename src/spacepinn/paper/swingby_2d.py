from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import spacepinn
from spacepinn.config.config_2d import exact_bc_2d_config, soft_bc_2d_config
from spacepinn.paper.common import smoke_mode_enabled
from spacepinn.paper._baseline_capture import capture_baseline_entry
from spacepinn.paper._baseline_defaults import paper_baseline_solver_kwargs
from spacepinn.paper._baseline_summary import print_baseline_delta_v_summary
from spacepinn.opengoddard.geometric_2d_goddard import geometric_2d_opengoddard
from spacepinn.plotting.helpers import (
    get_gravity_sources,
    get_quiver_data,
    plot_masses_2d,
    register_plot_artifact_if_possible,
    set_time_axis_labels,
)
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter
from spacepinn.runner import print_collection_run_summary, run_experiment_collection

DTYPE = "float32"
RUN_ROOT = Path(spacepinn.__file__).resolve().parents[2] / "runs"
COLLECTION_LABEL = "swingby_2d"
FIG_PREFIX = "swingby_2d"
BASELINE_LABEL = "Baseline (OpenGoddard)"
ORDINARY_LABEL = "PINN with soft BC"
GEOMETRIC_LABEL = "PINN with exact BC"
DIRECT_COLLOCATION_COLOR = PALETTE["opengoddard"]
QUIVER_COUNT = 10
MAIN_LINEWIDTH = 2.4
SECONDARY_LINEWIDTH = 2.0
MAIN_FIGSIZE = (7.0, 6.2)
LOSS_FIGSIZE = MAIN_FIGSIZE
ORDINARY_LAMBDA_BC = 0.32397426295281967


def build_configs(*, smoke: bool | None = None) -> list[dict]:
    geometric_runtime_config = deepcopy(exact_bc_2d_config)
    ordinary_runtime_config = deepcopy(soft_bc_2d_config)

    geometric_runtime_config["label"] = GEOMETRIC_LABEL
    ordinary_runtime_config["label"] = ORDINARY_LABEL
    geometric_runtime_config["numeric_dtype"] = DTYPE
    ordinary_runtime_config["numeric_dtype"] = DTYPE
    geometric_runtime_config["plotting"]["quiver_count"] = QUIVER_COUNT
    ordinary_runtime_config["plotting"]["linestyle"] = "solid"
    ordinary_runtime_config["plotting"]["trajectory_linestyle"] = "solid"
    ordinary_runtime_config["plotting"]["quiver_count"] = QUIVER_COUNT
    ordinary_runtime_config["optimizer"]["w_bc"] = ORDINARY_LAMBDA_BC

    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    if smoke_enabled:
        for config in (geometric_runtime_config, ordinary_runtime_config):
            config["optimizer"]["n_adam"] = 1
            config["optimizer"]["n_lbfgs"] = 0

    return [geometric_runtime_config, ordinary_runtime_config]


def build_baseline_entry(*, smoke: bool | None = None) -> dict:
    smoke_enabled = smoke_mode_enabled() if smoke is None else smoke
    conventional_result = geometric_2d_opengoddard(
        BASELINE_LABEL,
        **paper_baseline_solver_kwargs(smoke_enabled=smoke_enabled),
    )
    conventional_result["color"] = DIRECT_COLLOCATION_COLOR
    conventional_result["linestyle"] = "solid"
    conventional_result["trajectory_linestyle"] = "solid"
    conventional_result["quiver_count"] = QUIVER_COUNT
    return {
        "label": conventional_result["label"],
        "result": conventional_result["result"],
        "model": conventional_result.get("model"),
        "config": conventional_result.get("config"),
        "plotting": {
            key: conventional_result[key]
            for key in ("linestyle", "color", "quiver_scale", "quiver_count")
            if key in conventional_result
        },
        "source": "opengoddard",
    }


def _build_plotter(entries: list[dict], *, output_dir: str | Path) -> TrajectoryPlotter:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = SECONDARY_LINEWIDTH
    return plotter


def plot_traj_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    if not hasattr(plotter, "experiments"):
        plotter.plot_traj_2d()
        return
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.r[:, 0],
            result.r[:, 1],
            linestyle=exp.get("trajectory_linestyle", exp["linestyle"]),
            color=exp["color"],
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        r_q, G_q, _ = get_quiver_data(
            result,
            step=exp.get("quiver_step", 10),
            count=exp.get("quiver_count"),
        )
        ax.quiver(
            r_q[:, 0],
            r_q[:, 1],
            G_q[:, 0],
            G_q[:, 1],
            color=exp["color"],
            scale=exp["quiver_scale"],
            label="_nolegend_",
        )

    reference_result = entries[0]["result"]
    ax.plot(reference_result.r0[0], reference_result.r0[1], "o", color="red", label=r"$r(t=0)$")
    ax.plot(
        reference_result.rN[0],
        reference_result.rN[1],
        "x",
        color="red",
        markersize=8,
        markeredgewidth=1.8,
        label=r"$r(t=1)$",
    )
    plot_masses_2d(ax, get_gravity_sources(reference_result))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_traj2d.pdf"
    fig.savefig(figure_path, pad_inches=0.05)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_thrust_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    if not hasattr(plotter, "experiments"):
        plotter.plot_thrust()
        return
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.t,
            result.F_mag,
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=label,
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )

    set_time_axis_labels(ax, "Thrust magnitude", plot_legend=True)
    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_thrust.pdf"
    fig.savefig(figure_path, pad_inches=0.05)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_gravity_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    if not hasattr(plotter, "experiments"):
        plotter.plot_gravity(legend_mode="compact")
        return
    fig, ax = plt.subplots(figsize=MAIN_FIGSIZE)

    ymax = 0.0
    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.t,
            result.a_mag,
            linestyle=exp["linestyle"],
            color=exp["color"],
            label=f"{label} RFM",
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        ax.plot(
            result.t,
            result.G_mag,
            linestyle="dashed" if exp["linestyle"] == "solid" else exp["linestyle"],
            color=exp["color"],
            label=f"{label} Gravity",
            linewidth=plotter.secondary_linewidth,
            zorder=exp["zorder"],
        )
        ymax = max(ymax, max(result.a_mag), max(result.G_mag))

    set_time_axis_labels(ax, "Gravity / Required Force magnitude", plot_legend=False)
    ymin = min(
        min(min(exp["result"].a_mag), min(exp["result"].G_mag))
        for exp in plotter.experiments.values()
    )
    lower_margin = max(0.06, 0.08 * (ymax - ymin))
    upper_margin = max(0.10, 0.16 * ymax)
    ax.set_ylim(max(0.0, ymin - lower_margin), ymax + upper_margin)
    ax.legend(
        loc="upper left",
        ncol=2,
        fontsize=9,
        columnspacing=1.0,
        handlelength=1.6,
        labelspacing=0.35,
        framealpha=0.9,
    )
    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_gravity.pdf"
    fig.savefig(figure_path, pad_inches=0.05)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_loss_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    if not hasattr(plotter, "experiments"):
        plotter_loss = TrajectoryPlotter(
            [entry for entry in entries if entry["label"] != BASELINE_LABEL],
            dim=2,
            figsize=LOSS_FIGSIZE,
            fig_prefix=FIG_PREFIX,
            output_dir=output_dir,
        )
        plotter_loss.main_linewidth = MAIN_LINEWIDTH
        plotter_loss.secondary_linewidth = SECONDARY_LINEWIDTH
        plotter_loss.plot_loss()
        return
    fig, ax = plt.subplots(figsize=LOSS_FIGSIZE)

    for label, exp in plotter.experiments.items():
        result = exp["result"]
        ax.plot(
            result.loss,
            linestyle="solid",
            label=f"{label} Total Loss",
            color=exp["color"],
            linewidth=plotter.main_linewidth,
            zorder=exp["zorder"],
        )
        if result.loss_bc:
            ax.plot(
                result.loss_bc,
                linestyle="--",
                label=label + r" $\lambda_{BC}$$L_{BC}$",
                color=exp["color"],
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )
        if result.loss_physics:
            ax.plot(
                result.loss_physics,
                linestyle="-.",
                label=label + r" $\lambda_{P}$$L_{P}$",
                color=exp["color"],
                linewidth=plotter.secondary_linewidth,
                zorder=exp["zorder"],
            )

    visible_lengths = [len(entry["result"].loss) for entry in entries]
    if visible_lengths:
        ax.set_xlim(0, max(visible_lengths))
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_loss.pdf"
    fig.savefig(figure_path, pad_inches=0.05)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def main(*, skip_plots: bool = False, print_summary: bool = True, smoke: bool | None = None):
    collection_run = run_experiment_collection(
        configs=build_configs(smoke=smoke),
        label=COLLECTION_LABEL,
        run_root=str(RUN_ROOT),
        additional_entries=[
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
        plot_traj_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
        loss_entries = [entry for entry in collection_run["entries"] if entry["label"] != BASELINE_LABEL]
        plot_loss_figure(loss_entries, output_dir=collection_run["plot_output_dir"])
        plot_thrust_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
        plot_gravity_figure(collection_run["entries"], output_dir=collection_run["plot_output_dir"])
    return collection_run


if __name__ == "__main__":
    main()

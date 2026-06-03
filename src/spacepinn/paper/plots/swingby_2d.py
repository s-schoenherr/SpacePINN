from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from spacepinn.paper.style import (
    LOSS_AXES_RECT,
    LOSS_FIGSIZE,
    MAIN_AXES_RECT,
    MAIN_FIGSIZE,
    MAIN_LINEWIDTH,
    SECONDARY_LINEWIDTH,
    TRAJECTORY_AXES_RECT,
    TRAJECTORY_FIGSIZE,
    configure_paper_plotter,
)
from spacepinn.plotting.helpers import (
    get_gravity_sources,
    get_quiver_data,
    register_plot_artifact_if_possible,
    set_time_axis_labels,
)
from spacepinn.plotting.paper_style import PAPER_STYLE
from spacepinn.plotting.style import PALETTE
from spacepinn.plotter import TrajectoryPlotter


FIG_PREFIX = "swingby_2d"
GEOMETRIC_LABEL = "PINN with exact BC"
ORDINARY_LABEL = "PINN with soft BC"
BASELINE_LABEL = "Baseline (OpenGoddard)"
QUIVER_COUNT = 10
BOXPLOT_FIGSIZE = (17.2, 4.8)
BOXPLOT_AXIS_LABEL_FONTSIZE = 18
BOXPLOT_DELTA_V_LABEL_FONTSIZE = 20
BOXPLOT_TICK_LABEL_FONTSIZE = 16
BOXPLOT_BASELINE_LEGEND_FONTSIZE = 13
BOXPLOT_SUBPLOT_ADJUST = {
    "left": 0.08,
    "right": 0.985,
    "bottom": 0.20,
    "top": 0.94,
    "wspace": 0.34,
}
COLORS = {
    GEOMETRIC_LABEL: PALETTE["position"],
    ORDINARY_LABEL: PALETTE["vanilla"],
}

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.unicode_minus": True,
        "font.size": 11,
    }
)


def _draw_boxplot_panel(ax, values: list[list[float]], *, display_labels: list[str], colors: dict[str, str], scatter_size: float) -> None:
    boxplot = ax.boxplot(values, patch_artist=True, tick_labels=display_labels)

    for patch, group_name in zip(boxplot["boxes"], colors):
        patch.set_facecolor(colors[group_name])
        patch.set_alpha(0.45)

    for index, group_name in enumerate(colors, start=1):
        group_values = np.array(values[index - 1], dtype=float)
        if group_values.size == 0:
            continue
        x_positions = np.full(group_values.shape, float(index), dtype=float)
        jitter = np.linspace(-0.08, 0.08, group_values.size) if group_values.size > 1 else np.array([0.0])
        ax.scatter(x_positions + jitter, group_values, color=colors[group_name], s=scatter_size, alpha=0.65)


def _apply_boxplot_ylim(ax, values: list[list[float]], *, extra_top_fraction: float = 0.16) -> None:
    flattened = [float(value) for group_values in values for value in group_values]
    if not flattened:
        return
    ymin = min(flattened)
    ymax = max(flattened)
    span = ymax - ymin
    lower_pad = max(0.0, 0.04 * span)
    upper_pad = max(1.0e-3, extra_top_fraction * max(span, ymax))
    ax.set_ylim(max(0.0, ymin - lower_pad), ymax + upper_pad)


def _trim_soft_bc_display_outliers(values: list[list[float]]) -> list[list[float]]:
    trimmed = [list(group_values) for group_values in values]
    soft_bc_index = 1
    if soft_bc_index >= len(trimmed):
        return trimmed

    soft_bc_values = [float(value) for value in trimmed[soft_bc_index]]
    if len(soft_bc_values) < 2:
        return trimmed

    sorted_values = sorted(soft_bc_values)
    largest = sorted_values[-1]
    second_largest = sorted_values[-2]
    if second_largest <= 0:
        return trimmed

    if largest <= 10.0 * second_largest:
        return trimmed

    trimmed[soft_bc_index] = [value for value in soft_bc_values if value != largest]
    return trimmed


def _build_plotter(entries: list[dict], *, output_dir: str | Path) -> TrajectoryPlotter:
    plotter = TrajectoryPlotter(
        entries,
        dim=2,
        figsize=MAIN_FIGSIZE,
        fig_prefix=FIG_PREFIX,
        output_dir=output_dir,
    )
    return configure_paper_plotter(plotter)


def _loss_label(group_name: str, suffix: str) -> str:
    replacements = {
        GEOMETRIC_LABEL: "Exact BC",
        ORDINARY_LABEL: "Soft BC",
    }
    return f"{replacements.get(group_name, group_name)} {suffix}"


def plot_traj_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    fig = plt.figure(figsize=TRAJECTORY_FIGSIZE)
    ax = fig.add_axes(TRAJECTORY_AXES_RECT)

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
    ax.plot(reference_result.r0[0], reference_result.r0[1], "o", color="red", label=r"$\mathbf{r}(t_0)$")
    ax.plot(
        reference_result.rN[0],
        reference_result.rN[1],
        "x",
        color="red",
        markersize=7,
        markeredgewidth=1.5,
        label=r"$\mathbf{r}(T)$",
    )
    gravity_sources = get_gravity_sources(reference_result)
    mass_colors = ["#006400", "#228B22", "#6B8E23"]
    mass_labels = [r"$GM_1 = 0.5$", r"$GM_2 = 1.0$", r"$GM_3 = 0.5$"]
    mass_marker_sizes = [170, 300, 170]
    annotation_offsets = [(12, 10), (12, 2), None]
    for index, (x, y, _gm) in enumerate(gravity_sources):
        ax.scatter(x, y, s=mass_marker_sizes[index], color=mass_colors[index], marker="o", zorder=4)
        if index < 2:
            ax.annotate(
                mass_labels[index],
                (x, y),
                xytext=annotation_offsets[index],
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=plotter.legend_fontsize + 3.0,
                color="black",
            )
        else:
            ax.text(
                0.7,
                0.2,
                mass_labels[index],
                ha="left",
                va="center",
                fontsize=plotter.legend_fontsize + 3.0,
                color="black",
            )
    ax.set_xlabel("x / normalized units", labelpad=10)
    ax.set_ylabel("y / normalized units")
    ax.set_box_aspect(1)
    ax.set_aspect("equal")
    plotter.style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="0.3",
        columnspacing=plotter.legend_columnspacing,
        handlelength=plotter.legend_handlelength,
        labelspacing=0.30,
        borderaxespad=0.35,
    )
    plotter.style_legend(legend)
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_traj2d.pdf"
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_loss_figure(entries: list[dict], *, output_dir: str | Path, fig_name: str = f"{FIG_PREFIX}_loss") -> None:
    loss_entries = [entry for entry in entries if entry["label"] != BASELINE_LABEL and getattr(entry["result"], "loss", None)]
    if not loss_entries:
        return

    plotter = _build_plotter(loss_entries, output_dir=output_dir)
    fig = plt.figure(figsize=LOSS_FIGSIZE)
    ax = fig.add_axes(LOSS_AXES_RECT)

    for index, entry in enumerate(loss_entries):
        label = entry["label"]
        result = entry["result"]
        color = entry.get("color", COLORS.get(label, PALETTE["position"]))
        zorder = len(loss_entries) - index
        ax.plot(
            result.loss,
            linestyle="solid",
            color=color,
            label=_loss_label(label, "Total Loss"),
            linewidth=MAIN_LINEWIDTH,
            zorder=zorder,
        )
        if result.loss_bc:
            ax.plot(
                result.loss_bc,
                linestyle="--",
                color=color,
                label=_loss_label(label, r"$\lambda_{BC}L_{BC}$"),
                linewidth=SECONDARY_LINEWIDTH,
                zorder=zorder,
            )
        if result.loss_physics:
            ax.plot(
                result.loss_physics,
                linestyle="-.",
                color=color,
                label=_loss_label(label, r"$\lambda_{P}L_{P}$"),
                linewidth=SECONDARY_LINEWIDTH,
                zorder=zorder,
            )

    visible_lengths = [len(entry["result"].loss) for entry in loss_entries]
    if visible_lengths:
        ax.set_xlim(0, max(visible_lengths) * 1.18)
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    legend = ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        columnspacing=PAPER_STYLE.legend_columnspacing,
        handlelength=PAPER_STYLE.legend_handlelength,
        labelspacing=0.28,
        borderaxespad=0.35,
    )
    plotter.style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_thrust_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

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
    ax.set_box_aspect(1)
    plotter.style_axes(ax)
    plotter.style_legend(ax.get_legend())
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_thrust.pdf"
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_gravity_figure(entries: list[dict], *, output_dir: str | Path) -> None:
    plotter = _build_plotter(entries, output_dir=output_dir)
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

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
    ax.set_box_aspect(1)
    ymin = min(min(min(exp["result"].a_mag), min(exp["result"].G_mag)) for exp in plotter.experiments.values())
    lower_margin = max(0.06, 0.08 * (ymax - ymin))
    upper_margin = max(0.30, 0.36 * ymax)
    ax.set_ylim(max(0.0, ymin - lower_margin), ymax + upper_margin)
    plotter.style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="0.3",
        columnspacing=plotter.legend_columnspacing,
        handlelength=plotter.legend_handlelength,
        labelspacing=0.28,
        borderaxespad=0.35,
    )
    plotter.style_legend(legend)
    figure_path = Path(output_dir) / f"{FIG_PREFIX}_gravity.pdf"
    plotter.save_figure(fig, figure_path)
    register_plot_artifact_if_possible(str(figure_path))
    plt.show()


def plot_monte_carlo_boxplots_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = BOXPLOT_FIGSIZE,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    metric_specs = [
        ("delta_v", r"$\Delta V$"),
        ("t_total", "Time of Flight"),
        ("iterations_to_convergence", "Iterations to Convergence"),
    ]
    display_labels = [
        "Exact BC",
        "Soft BC",
    ]

    for ax, (metric_name, ylabel) in zip(axes, metric_specs):
        values = []
        for group_name in colors:
            group_values = []
            for entry in grouped_entries[group_name]:
                result = entry["result"]
                if metric_name == "iterations_to_convergence":
                    group_values.append(len(result.loss))
                else:
                    group_values.append(getattr(result, metric_name))
            values.append(group_values)
        display_values = _trim_soft_bc_display_outliers(values)
        _draw_boxplot_panel(ax, display_values, display_labels=display_labels, colors=colors, scatter_size=18)

        if metric_name == "delta_v":
            _apply_boxplot_ylim(ax, display_values, extra_top_fraction=0.22)

        ylabel_size = BOXPLOT_DELTA_V_LABEL_FONTSIZE if metric_name == "delta_v" else BOXPLOT_AXIS_LABEL_FONTSIZE
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight="normal")
        ax.tick_params(axis="x", labelrotation=0, labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")
        if metric_name == "delta_v" and baseline_entry is not None:
            baseline_delta_v = float(baseline_entry["result"].delta_v)
            baseline_legend = Line2D(
                [],
                [],
                linestyle="None",
                marker=None,
                linewidth=0.0,
                label=f"{BASELINE_LABEL}: {baseline_delta_v:.3g}",
            )
            ax.legend(
                handles=[baseline_legend],
                loc="upper left",
                frameon=True,
                facecolor="white",
                edgecolor="black",
                framealpha=0.95,
                handlelength=0.0,
                handletextpad=0.0,
                borderpad=0.25,
                labelcolor="black",
                prop={"weight": "bold", "size": BOXPLOT_BASELINE_LEGEND_FONTSIZE},
            )

    fig.subplots_adjust(**BOXPLOT_SUBPLOT_ADJUST)
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()

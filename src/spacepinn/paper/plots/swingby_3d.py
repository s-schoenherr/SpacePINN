from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from spacepinn.paper.style import (
    LOSS_AXES_RECT,
    LOSS_FIGSIZE,
    MAIN_AXES_RECT,
    MAIN_FIGSIZE,
    MAIN_LINEWIDTH,
    SECONDARY_LINEWIDTH,
    TRAJECTORY_AXES_RECT,
    TRAJECTORY_FIGSIZE,
)
from spacepinn.plotting.helpers import get_gravity_sources, register_plot_artifact_if_possible
from spacepinn.plotting.paper_style import PAPER_STYLE
from spacepinn.plotting.style import PALETTE


FIG_PREFIX = "swingby_3d"
BASELINE_LABEL = "Baseline (OpenGoddard)"
GEOMETRIC_LABEL = "PINN with exact BC"
ORDINARY_LABEL = "PINN with soft BC"
PRETRAINED_LABEL = "PINN with exact BC and pre-conditioning"
BOXPLOT_FIGSIZE = (17.2, 4.8)
BOXPLOT_AXIS_LABEL_FONTSIZE = 18
BOXPLOT_DELTA_V_LABEL_FONTSIZE = 20
BOXPLOT_TICK_LABEL_FONTSIZE = 16
BOXPLOT_BASELINE_LEGEND_FONTSIZE = 13
BOXPLOT_INSET_TICK_LABEL_FONTSIZE = 10
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
    PRETRAINED_LABEL: PALETTE["kinematic"],
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


def _select_best_entry(entries: list[dict]) -> dict:
    return min(entries, key=lambda entry: float(entry["result"].delta_v))


def _style_axes(ax, *, is_3d: bool = False) -> None:
    ax.xaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    ax.yaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    if is_3d and hasattr(ax, "zaxis"):
        ax.zaxis.label.set_size(PAPER_STYLE.axis_label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=PAPER_STYLE.tick_label_fontsize)
    if is_3d and hasattr(ax, "zaxis"):
        ax.tick_params(axis="z", which="both", labelsize=PAPER_STYLE.tick_label_fontsize)


def _style_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(PAPER_STYLE.legend_fontsize)
    frame = legend.get_frame()
    if frame is not None:
        frame.set_alpha(PAPER_STYLE.legend_framealpha)
        frame.set_edgecolor("black")
        frame.set_linewidth(1.0)
        frame.set_facecolor("white")


def _style_boxplot_axis(ax, *, axis_label_fontsize: float, tick_label_fontsize: float) -> None:
    _style_axes(ax)
    ax.xaxis.label.set_size(axis_label_fontsize)
    ax.yaxis.label.set_size(axis_label_fontsize)
    for label in ax.get_xticklabels():
        label.set_fontsize(tick_label_fontsize)
    for label in ax.get_yticklabels():
        label.set_fontsize(tick_label_fontsize)


def _save_figure(fig, figure_path: Path) -> None:
    save_kwargs = {"pad_inches": PAPER_STYLE.save_pad_inches}
    if PAPER_STYLE.save_bbox_inches is not None:
        save_kwargs["bbox_inches"] = PAPER_STYLE.save_bbox_inches
    fig.savefig(figure_path, **save_kwargs)


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


def _loss_label(group_name: str, suffix: str) -> str:
    replacements = {
        GEOMETRIC_LABEL: "Exact BC",
        ORDINARY_LABEL: "Soft BC",
        PRETRAINED_LABEL: "Exact BC + pre-cond.",
    }
    short_group = replacements.get(group_name, group_name.replace("pre-conditioning", "pre-cond."))
    return f"{short_group} {suffix}"


def _projection_label(group_name: str) -> str:
    replacements = {
        GEOMETRIC_LABEL: "Exact BC",
        ORDINARY_LABEL: "Soft BC",
        PRETRAINED_LABEL: "Exact BC + pre-cond.",
        BASELINE_LABEL: "Baseline",
    }
    return replacements.get(group_name, group_name)


def _projected_mass_handles(result) -> list[Line2D]:
    gravity_sources = get_gravity_sources(result)
    if gravity_sources is None:
        return []
    colors = ["#006400", "#228B22", "#6B8E23"]
    sizes = [10, 14, 10]
    handles = []
    for index, source in enumerate(gravity_sources):
        mass = float(source[-1])
        handles.append(
            Line2D(
                [],
                [],
                linestyle="None",
                marker="o",
                color=colors[index % len(colors)],
                markersize=sizes[index % len(sizes)],
                label=rf"$GM_{index + 1} = {mass:.1f}$",
            )
        )
    return handles


def _plot_projected_masses(ax, result, *, dims: tuple[int, int]) -> None:
    gravity_sources = get_gravity_sources(result)
    if gravity_sources is None:
        return
    colors = ["#006400", "#228B22", "#6B8E23"]
    sizes = [120, 220, 120]
    for index, source in enumerate(gravity_sources):
        ax.scatter(
            source[dims[0]],
            source[dims[1]],
            s=sizes[index % len(sizes)],
            color=colors[index % len(colors)],
            marker="o",
            zorder=4,
        )


def _set_projection_limits(ax, result, *, dims: tuple[int, int]) -> None:
    values = [result.r[:, dims[0]], result.r[:, dims[1]], result.r0[list(dims)], result.rN[list(dims)]]
    gravity_sources = get_gravity_sources(result)
    if gravity_sources is not None:
        for source in gravity_sources:
            values.append(np.asarray([source[dims[0]], source[dims[1]]], dtype=float))
    flattened = np.concatenate([np.ravel(np.asarray(value, dtype=float)) for value in values])
    finite = flattened[np.isfinite(flattened)]
    if finite.size == 0:
        lower, upper = -1.0, 1.0
    else:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
        padding = max(0.08, 0.05 * (upper - lower))
        lower -= padding
        upper += padding
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)


def plot_monte_carlo_traj_2d_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.9))
    any_entry = next(entry for entries in grouped_entries.values() for entry in entries if entries)
    reference_result = any_entry["result"]
    projection_specs = [
        (axes[0, 0], (0, 1), "x / normalized units", "y / normalized units"),
        (axes[0, 1], (0, 2), "x / normalized units", "z / normalized units"),
        (axes[1, 0], (1, 2), "y / normalized units", "z / normalized units"),
    ]
    line_handles: list[Line2D] = []

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = _select_best_entry(entries)
        best_result = best_entry["result"]
        handle = None
        for ax, dims, _xlabel, _ylabel in projection_specs:
            (line,) = ax.plot(
                best_result.r[:, dims[0]],
                best_result.r[:, dims[1]],
                color=color,
                linewidth=MAIN_LINEWIDTH,
                label=_projection_label(group_name),
            )
            handle = line
        if handle is not None:
            line_handles.append(handle)

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        baseline_handle = None
        for ax, dims, _xlabel, _ylabel in projection_specs:
            (line,) = ax.plot(
                baseline_result.r[:, dims[0]],
                baseline_result.r[:, dims[1]],
                color=PALETTE["opengoddard"],
                linewidth=MAIN_LINEWIDTH,
                linestyle="dashed",
                label=_projection_label(BASELINE_LABEL),
            )
            baseline_handle = line
        if baseline_handle is not None:
            line_handles.append(baseline_handle)

    start_handle = Line2D([], [], linestyle="None", marker="o", color="red", label=r"$\mathbf{r}(t_0)$")
    end_handle = Line2D(
        [],
        [],
        linestyle="None",
        marker="x",
        color="red",
        markeredgewidth=1.5,
        markersize=7,
        label=r"$\mathbf{r}(T)$",
    )
    for ax, dims, xlabel, ylabel in projection_specs:
        ax.plot(reference_result.r0[dims[0]], reference_result.r0[dims[1]], "o", color="red")
        ax.plot(
            reference_result.rN[dims[0]],
            reference_result.rN[dims[1]],
            "x",
            color="red",
            markersize=7,
            markeredgewidth=1.5,
        )
        _plot_projected_masses(ax, reference_result, dims=dims)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _set_projection_limits(ax, reference_result, dims=dims)
        ax.set_box_aspect(1)
        _style_axes(ax)

    axes[1, 1].set_frame_on(False)
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    legend = axes[1, 1].legend(
        handles=[*line_handles, start_handle, end_handle, *_projected_mass_handles(reference_result)],
        loc="center",
        frameon=True,
        borderpad=0.35,
        handlelength=PAPER_STYLE.legend_handlelength,
        labelspacing=0.22,
    )
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.985, wspace=0.26, hspace=0.34)
    fig.savefig(figure_path, pad_inches=PAPER_STYLE.save_pad_inches)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_traj_3d_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig = plt.figure(figsize=TRAJECTORY_FIGSIZE)
    ax = fig.add_axes(TRAJECTORY_AXES_RECT, projection="3d")
    any_entry = next(entry for entries in grouped_entries.values() for entry in entries if entries)

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = _select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(
            best_result.r[:, 0],
            best_result.r[:, 1],
            best_result.r[:, 2],
            color=color,
            linewidth=MAIN_LINEWIDTH,
            label=group_name,
        )

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        ax.plot(
            baseline_result.r[:, 0],
            baseline_result.r[:, 1],
            baseline_result.r[:, 2],
            color=PALETTE["opengoddard"],
            linewidth=MAIN_LINEWIDTH,
            linestyle="dashdot",
            label=BASELINE_LABEL,
        )

    ax.scatter(*any_entry["result"].r0, marker="o", color="red", label=r"$r(t=0)$")
    ax.scatter(*any_entry["result"].rN, marker="x", color="red", label=r"$r(t=1)$")
    ax.set_xlabel("x / normalized units")
    ax.set_ylabel("y / normalized units")
    ax.set_zlabel("z / normalized units")
    _style_axes(ax, is_3d=True)
    legend = ax.legend(loc="upper left", ncol=1, frameon=True)
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_loss_figure(entries: list[dict], *, output_dir: str | Path, fig_name: str = f"{FIG_PREFIX}_loss") -> None:
    loss_entries = [entry for entry in entries if entry["label"] != BASELINE_LABEL]
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
    _style_axes(ax)
    legend = ax.legend(
        loc="upper right",
        ncol=1,
        frameon=True,
        columnspacing=PAPER_STYLE.legend_columnspacing,
        handlelength=PAPER_STYLE.legend_handlelength,
        labelspacing=0.28,
        borderaxespad=0.35,
    )
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
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
        "Exact +\npre-cond.",
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

        _draw_boxplot_panel(ax, values, display_labels=display_labels, colors=colors, scatter_size=18)

        if metric_name == "delta_v":
            _apply_boxplot_ylim(ax, values, extra_top_fraction=0.18)

        ylabel_size = BOXPLOT_DELTA_V_LABEL_FONTSIZE if metric_name == "delta_v" else BOXPLOT_AXIS_LABEL_FONTSIZE
        ax.set_ylabel(ylabel, fontsize=ylabel_size, fontweight="normal")
        ax.tick_params(axis="x", labelrotation=0, labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        ax.tick_params(axis="y", labelsize=BOXPLOT_TICK_LABEL_FONTSIZE)
        _style_boxplot_axis(
            ax,
            axis_label_fontsize=ylabel_size,
            tick_label_fontsize=BOXPLOT_TICK_LABEL_FONTSIZE,
        )
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

            inset = inset_axes(
                ax,
                width="57%",
                height="58%",
                loc="upper left",
                bbox_to_anchor=(0.19, 0.00, 0.78, 0.80),
                bbox_transform=ax.transAxes,
                borderpad=0.0,
            )
            _draw_boxplot_panel(inset, values, display_labels=display_labels, colors=colors, scatter_size=10)
            inset.set_ylim(0.0, 0.009)
            inset.tick_params(axis="x", bottom=False, labelbottom=False)
            inset.tick_params(axis="y", labelsize=BOXPLOT_INSET_TICK_LABEL_FONTSIZE, pad=1)
            inset.set_xticks([])

            cluster_box_transform = blended_transform_factory(ax.transData, ax.transAxes)
            source_box_top = 0.06
            source_rect = Rectangle(
                (0.78, 0.0),
                2.46,
                source_box_top,
                fill=False,
                edgecolor="0.35",
                linewidth=1.1,
                alpha=0.9,
                transform=cluster_box_transform,
            )
            ax.add_patch(source_rect)
            connector = ConnectionPatch(
                xyA=(0.78, source_box_top),
                coordsA=cluster_box_transform,
                xyB=(0.0, 0.0),
                coordsB=inset.transAxes,
                color="0.35",
                linewidth=1.1,
                alpha=0.9,
            )
            ax.add_artist(connector)

    fig.subplots_adjust(**BOXPLOT_SUBPLOT_ADJUST)
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_thrust_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = _select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(best_result.t, best_result.F_mag, color=color, linewidth=MAIN_LINEWIDTH, label=group_name)

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        ax.plot(
            baseline_result.t,
            baseline_result.F_mag,
            color=PALETTE["opengoddard"],
            linewidth=MAIN_LINEWIDTH,
            linestyle="dashdot",
            label=BASELINE_LABEL,
        )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Thrust magnitude")
    ax.set_xlim(0, 1)
    ax.set_box_aspect(1)
    _style_axes(ax)
    legend = ax.legend(loc="upper left", ncol=1, frameon=True)
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_gravity_paper(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    baseline_entry: dict | None = None,
    figsize: tuple[float, float] = MAIN_FIGSIZE,
) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(MAIN_AXES_RECT)

    def _gravity_label(group_name: str, quantity: str) -> str:
        short_group = group_name.replace("pre-conditioning", "pre-cond.")
        short_quantity = "Grav." if quantity == "Gravity" else quantity
        return f"{short_group} {short_quantity}"

    for group_name, entries in grouped_entries.items():
        if not entries:
            continue
        color = colors[group_name]
        best_entry = _select_best_entry(entries)
        best_result = best_entry["result"]
        ax.plot(
            best_result.t,
            best_result.a_mag,
            color=color,
            linewidth=MAIN_LINEWIDTH,
            label=_gravity_label(group_name, "RFM"),
        )
        ax.plot(
            best_result.t,
            best_result.G_mag,
            color=color,
            linewidth=SECONDARY_LINEWIDTH,
            linestyle="--",
            label=_gravity_label(group_name, "Gravity"),
        )

    if baseline_entry is not None:
        baseline_result = baseline_entry["result"]
        ax.plot(
            baseline_result.t,
            baseline_result.a_mag,
            color=PALETTE["opengoddard"],
            linewidth=MAIN_LINEWIDTH,
            linestyle="dashdot",
            label=_gravity_label(BASELINE_LABEL, "RFM"),
        )
        ax.plot(
            baseline_result.t,
            baseline_result.G_mag,
            color=PALETTE["opengoddard"],
            linewidth=SECONDARY_LINEWIDTH,
            linestyle=(0, (5, 2, 1, 2)),
            label=_gravity_label(BASELINE_LABEL, "Gravity"),
        )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Gravity / Required Force magnitude")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 5.0)
    ax.set_box_aspect(1)
    _style_axes(ax)
    legend = ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        columnspacing=0.8,
        handlelength=1.4,
        labelspacing=0.28,
        borderaxespad=0.35,
    )
    _style_legend(legend)

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    _save_figure(fig, figure_path)
    register_plot_artifact_if_possible(figure_path)
    plt.show()

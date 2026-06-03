from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import spacepinn

from spacepinn.paper.style import MAIN_AXES_RECT, MAIN_FIGSIZE
from spacepinn.plotting.helpers import get_gravity_sources
from spacepinn.plotting.paper_style import PAPER_STYLE
from spacepinn.plotting.style import PALETTE
from spacepinn.runner import load_run

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.unicode_minus": True,
        "font.size": 11,
    }
)

DATA_DIR = Path(spacepinn.__file__).resolve().parents[2] / "data" / "runs" / "appendix" / "static_total_time_sweep"
PINN_COLORS = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377"]
BASELINE_COLORS = ["#4D4D4D", "#808080", "#B0B0B0"]
MASS_COLORS = ["#006400", "#228B22", "#6B8E23"]
MASS_LABELS = [r"$GM_1 = 0.5$", r"$GM_2 = 1.0$", r"$GM_3 = 0.5$"]
MASS_MARKER_SIZES = [70, 110, 70]
SINGLE_AXES_RECT = (0.24, 0.16, 0.70, 0.76)
TRAJ_LABEL_FONTSIZE = 16.0
AXIS_LABEL_FONTSIZE = 16.0
TICK_LABEL_FONTSIZE = 14.0
LEGEND_FONTSIZE = 11.0
TRAJ_LEGEND_FONTSIZE = 13.0
INSET_TICK_LABEL_FONTSIZE = 10.0
LINEWIDTH = 2.3
BASELINE_DASHES = [(0, (5, 2)), (0, (3, 2, 1, 2)), (0, (1, 1))]


def _style_axis(ax) -> None:
    ax.xaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)


def _style_legend(legend) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(LEGEND_FONTSIZE)
    frame = legend.get_frame()
    if frame is not None:
        frame.set_alpha(PAPER_STYLE.legend_framealpha)
        frame.set_edgecolor("black")
        frame.set_facecolor("white")


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches=PAPER_STYLE.save_bbox_inches, pad_inches=PAPER_STYLE.save_pad_inches)
    plt.close(fig)


def _format_entry(*, source: str, result: Any, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    t_total = float(getattr(result, "t_total", summary.get("t_total") if summary else 0.0))
    if source == "opengoddard":
        label = rf"Baseline $T={t_total:.2f}$"
        color = BASELINE_COLORS[_format_entry.baseline_index % len(BASELINE_COLORS)]
        linestyle = BASELINE_DASHES[_format_entry.baseline_index % len(BASELINE_DASHES)]
        _format_entry.baseline_index += 1
    else:
        label = rf"PINN $T={t_total:.2f}$"
        color = PINN_COLORS[_format_entry.pinn_index % len(PINN_COLORS)]
        linestyle = "-"
        _format_entry.pinn_index += 1
    return {
        "label": label,
        "result": result,
        "source": source,
        "summary": summary or {},
        "color": color,
        "linestyle": linestyle,
    }


_format_entry.pinn_index = 0
_format_entry.baseline_index = 0


def _reset_label_counters() -> None:
    _format_entry.pinn_index = 0
    _format_entry.baseline_index = 0


def _load_entries(record_dir: str | Path) -> list[dict[str, Any]]:
    record_dir = Path(record_dir)
    summary = json.loads((record_dir / "expected_summary.json").read_text(encoding="utf-8"))
    entries: list[dict[str, Any]] = []
    _reset_label_counters()
    for item in summary["entries"]:
        entry_id = item["entry_id"]
        data = np.load(record_dir / "expected_timeseries" / f"{entry_id}.npz")
        source = item.get("source", "pinn")
        t_total = float(item["summary"]["t_total"])
        result = SimpleNamespace(
            t=data["t"],
            t_total=t_total,
            r=data["r"],
            v=data["v"],
            a=data["a"],
            F=data["F"],
            G=data["G"],
            F_mag=np.linalg.norm(data["F"], axis=1),
            G_mag=np.linalg.norm(data["G"], axis=1),
            r0=data["r"][0],
            rN=data["r"][-1],
            ao=np.array([[-0.5, -1.0, 0.0, 0.5], [-0.2, 0.4, 0.0, 1.0], [0.8, 0.3, 0.0, 0.5]]),
            gravity_sources=np.array([[-0.5, -1.0, 0.0, 0.5], [-0.2, 0.4, 0.0, 1.0], [0.8, 0.3, 0.0, 0.5]]),
            loss=[],
            loss_physics=[],
            loss_bc=[],
        )
        entries.append(_format_entry(source=source, result=result, summary=item["summary"]))
    return entries


def _load_run_entries(run_dir: str | Path) -> list[dict[str, Any]]:
    _reset_label_counters()
    run = load_run(run_dir)
    entries: list[dict[str, Any]] = []
    for entry in run["entries"]:
        result = entry["result"]
        summary = {
            "t_total": float(result.t_total),
            "final_loss": float(result.loss[-1]) if getattr(result, "loss", None) else 0.0,
        }
        entries.append(_format_entry(source=entry.get("source", "pinn"), result=result, summary=summary))
    return entries


def _plot_masses(ax, result, *, projection: tuple[int, int]) -> None:
    sources = get_gravity_sources(result)
    for index, source in enumerate(sources):
        x = source[projection[0]]
        y = source[projection[1]]
        ax.scatter(x, y, s=MASS_MARKER_SIZES[index], color=MASS_COLORS[index], marker="o", zorder=4, label=MASS_LABELS[index])

# keep the plotting functions from the current file below this marker
def plot_traj2d(entries: list[dict[str, Any]], output_path: str | Path) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.16,
        right=0.96,
        bottom=0.12,
        top=0.95,
        wspace=0.42,
        hspace=0.34,
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]
    legend_ax = fig.add_subplot(grid[1, 1])
    legend_ax.axis("off")
    projections = [
        ((0, 1), "x / normalized units", "y / normalized units"),
        ((0, 2), "x / normalized units", "z / normalized units"),
        ((1, 2), "y / normalized units", "z / normalized units"),
    ]
    handles: list[Any] = []
    labels: list[str] = []

    for ax, (projection, xlabel, ylabel) in zip(axes, projections):
        for entry in entries:
            result = entry["result"]
            handle = ax.plot(
                result.r[:, projection[0]],
                result.r[:, projection[1]],
                color=entry["color"],
                linestyle=entry["linestyle"],
                linewidth=LINEWIDTH,
                label=entry["label"],
                zorder=2,
            )[0]
            if ax is axes[0]:
                handles.append(handle)
                labels.append(entry["label"])

        reference = entries[0]["result"]
        start = ax.plot(
            reference.r0[projection[0]],
            reference.r0[projection[1]],
            "o",
            color="red",
            label=r"$\mathbf{r}(t_0)$",
            zorder=5,
        )[0]
        end = ax.plot(
            reference.rN[projection[0]],
            reference.rN[projection[1]],
            "x",
            color="red",
            markersize=7,
            markeredgewidth=1.5,
            label=r"$\mathbf{r}(T)$",
            zorder=5,
        )[0]
        _plot_masses(ax, reference, projection=projection)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        x_values = np.concatenate([entry["result"].r[:, projection[0]] for entry in entries])
        y_values = np.concatenate([entry["result"].r[:, projection[1]] for entry in entries])
        x_min = min(-1.08, float(np.nanmin(x_values)) - 0.04)
        x_max = max(1.08, float(np.nanmax(x_values)) + 0.04)
        y_min = min(-1.08, float(np.nanmin(y_values)) - 0.04)
        y_max = max(1.08, float(np.nanmax(y_values)) + 0.04)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.label.set_size(TRAJ_LABEL_FONTSIZE)
        ax.yaxis.label.set_size(TRAJ_LABEL_FONTSIZE)
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABEL_FONTSIZE)

    handles.extend([start, end])
    labels.extend([r"$\mathbf{r}(t_0)$", r"$\mathbf{r}(T)$"])
    mass_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=MASS_COLORS[i], markersize=size / 18)
        for i, size in enumerate(MASS_MARKER_SIZES)
    ]
    handles.extend(mass_handles)
    labels.extend(MASS_LABELS)

    line_legend = legend_ax.legend(
        handles[: len(entries)],
        labels[: len(entries)],
        loc="center left",
        bbox_to_anchor=(-0.42, 0.50),
        frameon=True,
        ncol=1,
        handlelength=1.15,
        labelspacing=0.05,
        borderpad=0.25,
    )
    _style_legend(line_legend)
    for text in line_legend.get_texts():
        text.set_fontsize(TRAJ_LEGEND_FONTSIZE)
    legend_ax.add_artist(line_legend)
    marker_legend = legend_ax.legend(
        handles[len(entries) :],
        labels[len(entries) :],
        loc="center left",
        bbox_to_anchor=(0.45, 0.50),
        frameon=True,
        ncol=1,
        handlelength=1.0,
        labelspacing=0.05,
        borderpad=0.25,
    )
    _style_legend(marker_legend)
    for text in marker_legend.get_texts():
        text.set_fontsize(TRAJ_LEGEND_FONTSIZE)
    _save(fig, Path(output_path))


def plot_traj3d(entries: list[dict[str, Any]], output_path: str | Path) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(SINGLE_AXES_RECT, projection="3d")
    for entry in entries:
        result = entry["result"]
        ax.plot(
            result.r[:, 0],
            result.r[:, 1],
            result.r[:, 2],
            color=entry["color"],
            linestyle=entry["linestyle"],
            linewidth=LINEWIDTH,
            label=entry["label"],
        )
    reference = entries[0]["result"]
    ax.scatter(reference.r0[0], reference.r0[1], reference.r0[2], color="red", marker="o", label=r"$\mathbf{r}(t_0)$")
    ax.scatter(reference.rN[0], reference.rN[1], reference.rN[2], color="red", marker="x", label=r"$\mathbf{r}(T)$")
    for index, source in enumerate(get_gravity_sources(reference)):
        ax.scatter(
            source[0],
            source[1],
            source[2],
            s=MASS_MARKER_SIZES[index],
            color=MASS_COLORS[index],
            marker="o",
            label=MASS_LABELS[index],
        )
    ax.set_xlabel("x / normalized units")
    ax.set_ylabel("y / normalized units")
    ax.set_zlabel("z / normalized units")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_zlim(-1.08, 1.08)
    ax.view_init(elev=24, azim=-55)
    _style_axis(ax)
    ax.zaxis.label.set_size(AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="z", which="both", labelsize=TICK_LABEL_FONTSIZE)
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=TRAJ_LEGEND_FONTSIZE,
        ncol=1,
        handlelength=1.5,
        labelspacing=0.25,
        borderpad=0.35,
    )
    _style_legend(legend)
    for text in legend.get_texts():
        text.set_fontsize(TRAJ_LEGEND_FONTSIZE)
    _save(fig, Path(output_path))


def _plot_timeseries(
    entries: list[dict[str, Any]],
    output_path: str | Path,
    *,
    values_attr: str,
    ylabel: str,
    legend_loc: str = "upper right",
    ylim_top_factor: float = 1.18,
    ylim_top: float | None = None,
    add_outlier_inset: bool = False,
) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(SINGLE_AXES_RECT)
    series_maxima: list[float] = []
    for entry in entries:
        result = entry["result"]
        values = getattr(result, values_attr)
        series_maxima.append(float(np.nanmax(values)))
        ax.plot(
            result.t.squeeze(),
            values,
            color=entry["color"],
            linestyle=entry["linestyle"],
            linewidth=LINEWIDTH,
            label=entry["label"],
        )
    ax.set_xlabel("Normalized time")
    ax.set_ylabel(ylabel)
    ymin, ymax = ax.get_ylim()
    value_max = max(float(np.nanmax(getattr(entry["result"], values_attr))) for entry in entries)
    ax.set_ylim(min(ymin, -0.02 * value_max), ylim_top if ylim_top is not None else max(ymax, value_max * ylim_top_factor))
    _style_axis(ax)
    legend = ax.legend(
        loc=legend_loc,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        ncol=1,
        handlelength=1.5,
        labelspacing=0.25,
        borderpad=0.35,
    )
    _style_legend(legend)
    if add_outlier_inset and values_attr == "F_mag" and len(series_maxima) > 1:
        ordered_maxima = sorted(series_maxima)
        largest = ordered_maxima[-1]
        second_largest = ordered_maxima[-2]
        if second_largest > 0.0 and largest > 8.0 * second_largest:
            inset = fig.add_axes((0.33, 0.56, 0.32, 0.36))
            zoom_top = 1.55 * second_largest
            source_box_bottom = -1.7 * zoom_top
            source_box_top = 2.3 * zoom_top
            current_bottom, current_top = ax.get_ylim()
            ax.set_ylim(min(current_bottom, 1.25 * source_box_bottom), current_top)
            for entry in entries:
                result = entry["result"]
                inset.plot(
                    result.t.squeeze(),
                    getattr(result, values_attr),
                    color=entry["color"],
                    linestyle=entry["linestyle"],
                    linewidth=1.5,
                )
            inset.set_xlim(0.0, 1.0)
            inset.set_ylim(-0.02 * zoom_top, zoom_top)
            inset.tick_params(axis="both", which="both", labelsize=INSET_TICK_LABEL_FONTSIZE, pad=1)
            source_rect = Rectangle(
                (0.0, source_box_bottom),
                1.0,
                source_box_top - source_box_bottom,
                fill=False,
                edgecolor="0.30",
                linewidth=1.6,
                alpha=0.95,
                transform=ax.transData,
            )
            ax.add_patch(source_rect)
            connector = ConnectionPatch(
                xyA=(0.0, source_box_top),
                coordsA=ax.transData,
                xyB=(0.0, 0.0),
                coordsB=inset.transAxes,
                color="0.30",
                linewidth=1.6,
                alpha=0.95,
            )
            ax.add_artist(connector)
    _save(fig, Path(output_path))


def plot_final_loss(entries: list[dict[str, Any]], output_path: str | Path) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(SINGLE_AXES_RECT)
    for source, marker, label in [
        ("pinn", "o", "PINN with exact BC"),
        ("opengoddard", "s", "Baseline (OpenGoddard)"),
    ]:
        subset = [entry for entry in entries if entry["source"] == source]
        if not subset:
            continue
        t_total = np.array([float(entry["summary"]["t_total"]) for entry in subset], dtype=float)
        final_loss = np.array([float(entry["summary"]["final_loss"]) for entry in subset], dtype=float)
        order = np.argsort(t_total)
        ax.plot(
            t_total[order],
            final_loss[order],
            marker=marker,
            linewidth=LINEWIDTH,
            color=PALETTE["position"] if source == "pinn" else PALETTE["opengoddard"],
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"Fixed total time $T$")
    ax.set_ylabel("Final loss")
    _style_axis(ax)
    legend = ax.legend(loc="upper right", frameon=True, fontsize=LEGEND_FONTSIZE)
    _style_legend(legend)
    _save(fig, Path(output_path))


def plot_training_loss(entries: list[dict[str, Any]], output_path: str | Path) -> None:
    fig = plt.figure(figsize=MAIN_FIGSIZE)
    ax = fig.add_axes(SINGLE_AXES_RECT)
    for entry in entries:
        result = entry["result"]
        if entry["source"] == "opengoddard" or not getattr(result, "loss", None):
            continue
        epochs = np.arange(len(result.loss), dtype=float)
        ax.plot(
            epochs,
            np.asarray(result.loss, dtype=float),
            color=entry["color"],
            linestyle=entry["linestyle"],
            linewidth=LINEWIDTH,
            label=entry["label"],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Loss")
    _style_axis(ax)
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.5,
        labelspacing=0.25,
        borderpad=0.35,
    )
    _style_legend(legend)
    _save(fig, Path(output_path))


def plot_record(record_dir: str | Path, *, output_dir: str | Path | None = None, prefix: str | None = None) -> None:
    record_dir = Path(record_dir)
    output_dir = Path(output_dir) if output_dir is not None else record_dir / "source_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or record_dir.name.replace("paper_", "")
    entries = _load_entries(record_dir)
    plot_traj2d(entries, output_dir / f"{stem}_traj2d.pdf")
    plot_traj3d(entries, output_dir / f"{stem}_traj3d.pdf")
    plot_final_loss(entries, output_dir / f"{stem}_loss.pdf")
    _plot_timeseries(
        entries,
        output_dir / f"{stem}_thrust.pdf",
        values_attr="F_mag",
        ylabel="Thrust magnitude / normalized units",
        ylim_top_factor=1.12,
        ylim_top=0.8 if stem.endswith("_large") else None,
        add_outlier_inset=True,
    )
    _plot_timeseries(
        entries,
        output_dir / f"{stem}_gravity.pdf",
        values_attr="G_mag",
        ylabel="Gravity magnitude / normalized units",
        legend_loc="upper left",
        ylim_top_factor=1.10,
        ylim_top=7.0 if stem.endswith("_large") else 6.0,
    )


def plot_run(run_dir: str | Path, *, output_dir: str | Path, prefix: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _load_run_entries(run_dir)
    plot_traj2d(entries, output_dir / f"{prefix}_traj2d.pdf")
    plot_traj3d(entries, output_dir / f"{prefix}_traj3d.pdf")
    plot_training_loss(entries, output_dir / f"{prefix}_loss.pdf")
    _plot_timeseries(
        entries,
        output_dir / f"{prefix}_thrust.pdf",
        values_attr="F_mag",
        ylabel="Thrust magnitude / normalized units",
        ylim_top_factor=1.12,
        ylim_top=0.8 if prefix.endswith("_large") else None,
        add_outlier_inset=True,
    )
    _plot_timeseries(
        entries,
        output_dir / f"{prefix}_gravity.pdf",
        values_attr="G_mag",
        ylabel="Gravity magnitude / normalized units",
        legend_loc="upper left",
        ylim_top_factor=1.10,
        ylim_top=7.0 if prefix.endswith("_large") else 6.0,
    )



def main(
    *,
    record_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    prefix: str | None = None,
) -> None:
    if record_dir is not None:
        plot_record(record_dir, output_dir=output_dir, prefix=prefix)
        return

    target_dir = Path(output_dir) if output_dir is not None else DATA_DIR / "source_plots"
    plot_record(DATA_DIR / "medium", output_dir=target_dir, prefix="static_total_time_sweep_medium")
    plot_record(DATA_DIR / "large", output_dir=target_dir, prefix="static_total_time_sweep_large")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render the static total-time sweep appendix figures.")
    parser.add_argument("--record-dir", type=Path, default=None, help="Single exported record directory to replot.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plot output.")
    parser.add_argument("--prefix", type=str, default=None, help="Filename prefix for --record-dir mode.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(record_dir=args.record_dir, output_dir=args.output_dir, prefix=args.prefix)

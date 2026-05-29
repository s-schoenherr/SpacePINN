from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperPlotStyle:
    figure_size: tuple[float, float]
    axes_rect: tuple[float, float, float, float]
    axis_label_fontsize: float
    tick_label_fontsize: float
    legend_fontsize: float
    title_fontsize: float
    main_linewidth: float
    secondary_linewidth: float
    legend_framealpha: float
    legend_handlelength: float
    legend_columnspacing: float
    save_pad_inches: float
    save_bbox_inches: str | None


PAPER_STYLE = PaperPlotStyle(
    figure_size=(6.4, 5.6),
    axes_rect=(0.12, 0.12, 0.82, 0.82),
    axis_label_fontsize=13.0,
    tick_label_fontsize=12.0,
    legend_fontsize=10.0,
    title_fontsize=11.0,
    main_linewidth=2.0,
    secondary_linewidth=1.6,
    legend_framealpha=0.95,
    legend_handlelength=1.5,
    legend_columnspacing=0.9,
    save_pad_inches=0.05,
    save_bbox_inches=None,
)


def paper_figure_size(*, kind: str = "orbit") -> tuple[float, float]:
    if kind in {"orbit", "trajectory", "timeseries"}:
        return PAPER_STYLE.figure_size
    raise ValueError(f"Unknown paper figure kind: {kind}")


def paper_axes_rect(*, kind: str = "orbit") -> tuple[float, float, float, float]:
    if kind in {"orbit", "trajectory", "timeseries"}:
        return PAPER_STYLE.axes_rect
    raise ValueError(f"Unknown paper axes kind: {kind}")

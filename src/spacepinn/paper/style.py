from __future__ import annotations

from spacepinn.plotting.paper_style import PAPER_STYLE, paper_axes_rect, paper_figure_size

MAIN_FIGSIZE = paper_figure_size(kind="orbit")
TRAJECTORY_FIGSIZE = MAIN_FIGSIZE
LOSS_FIGSIZE = MAIN_FIGSIZE
MAIN_AXES_RECT = paper_axes_rect(kind="orbit")
TRAJECTORY_AXES_RECT = paper_axes_rect(kind="trajectory")
LOSS_AXES_RECT = paper_axes_rect(kind="timeseries")
MAIN_LINEWIDTH = PAPER_STYLE.main_linewidth
SECONDARY_LINEWIDTH = PAPER_STYLE.secondary_linewidth


def configure_paper_plotter(plotter):
    plotter.figsize = MAIN_FIGSIZE
    plotter.main_linewidth = MAIN_LINEWIDTH
    plotter.secondary_linewidth = SECONDARY_LINEWIDTH
    plotter.axis_label_fontsize = PAPER_STYLE.axis_label_fontsize
    plotter.tick_label_fontsize = PAPER_STYLE.tick_label_fontsize
    plotter.legend_fontsize = PAPER_STYLE.legend_fontsize
    plotter.title_fontsize = PAPER_STYLE.title_fontsize
    plotter.legend_framealpha = PAPER_STYLE.legend_framealpha
    plotter.save_pad_inches = PAPER_STYLE.save_pad_inches
    plotter.save_bbox_inches = PAPER_STYLE.save_bbox_inches
    return plotter

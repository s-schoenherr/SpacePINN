import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .plotting.paper_style import PAPER_STYLE
from .plotting.forces import plot_gravity as plot_gravity_impl
from .plotting.forces import plot_thrust as plot_thrust_impl
from .plotting.helpers import (
    get_quiver_data,
    plot_masses_2d,
    plot_masses_3d,
    set_time_axis_labels,
)
from .plotting.loss import plot_loss as plot_loss_impl
from .plotting.orbit import plot_orbit_traj as plot_orbit_traj_impl
from .plotting.trajectory import plot_traj_2d as plot_traj_2d_impl
from .plotting.trajectory import plot_traj_3d as plot_traj_3d_impl
from .plotting.trajectory import plot_traj_3d_projection as plot_traj_3d_projection_impl

plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.unicode_minus": True,
        "font.size": 11,
    }
)


class TrajectoryPlotter:
    def __init__(
        self,
        experiments,
        fig_prefix=None,
        dim=None,
        figsize=(6.0, 6.0),
        color_palette_fn=None,
        output_dir=None,
    ):
        """
        Class to plot the results of the trajectory optimization.
        **experiments: list of dicts where each dict contains:
            {
                'label': experiment name (required),
                'result': TrajectoryResult object (required),
                'linestyle': matplotlib linestyle (optional),
                'trajectory_linestyle': matplotlib linestyle for trajectory-only plots (optional),
                'color': matplotlib color (optional),
                'zorder': matplotlib draw order (optional)
            }
        """
        self.prefix = fig_prefix
        self.dim = dim
        self.figsize = figsize
        self.main_linewidth = PAPER_STYLE.main_linewidth
        self.secondary_linewidth = PAPER_STYLE.secondary_linewidth
        self.axis_label_fontsize = PAPER_STYLE.axis_label_fontsize
        self.tick_label_fontsize = PAPER_STYLE.tick_label_fontsize
        self.legend_fontsize = PAPER_STYLE.legend_fontsize
        self.title_fontsize = PAPER_STYLE.title_fontsize
        self.legend_framealpha = PAPER_STYLE.legend_framealpha
        self.legend_handlelength = PAPER_STYLE.legend_handlelength
        self.legend_columnspacing = PAPER_STYLE.legend_columnspacing
        self.save_pad_inches = PAPER_STYLE.save_pad_inches
        self.save_bbox_inches = PAPER_STYLE.save_bbox_inches
        self.experiments = {}
        self.output_dir = Path(output_dir) if output_dir is not None else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if experiments:
            if color_palette_fn is None:
                self.colors = self.get_color_palette(len(experiments))
            else:
                self.colors = color_palette_fn(len(experiments))

            for i, exp in enumerate(experiments):
                self.add_experiment(
                    label=exp["label"],
                    result=exp["result"],
                    linestyle=exp.get("linestyle", "-"),
                    trajectory_linestyle=exp.get("trajectory_linestyle"),
                    color=exp.get("color", self.colors[i]),
                    quiver_scale=exp.get("quiver_scale", 20),
                    quiver_step=exp.get("quiver_step", 10),
                    quiver_count=exp.get("quiver_count"),
                    zorder=exp.get("zorder", 2),
                )

    def add_experiment(
        self,
        label,
        result,
        linestyle="-",
        trajectory_linestyle=None,
        color=None,
        quiver_scale=10,
        quiver_step=10,
        quiver_count=None,
        zorder=2,
    ):
        self.experiments[label] = {
            "result": result,
            "linestyle": linestyle,
            "trajectory_linestyle": trajectory_linestyle if trajectory_linestyle is not None else linestyle,
            "color": color if color is not None else self.get_random_hex_color(),
            "quiver_scale": quiver_scale,
            "quiver_step": quiver_step,
            "quiver_count": quiver_count,
            "zorder": zorder,
        }

    def get_color_palette(self, num_colors):
        return plt.cm.viridis(np.linspace(0, 0.9, num_colors))

    def get_random_hex_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def _generate_fig_name(self):
        if self.prefix:
            return self.prefix
        return "_".join([f"{label.strip()}" for label, _ in self.experiments.items()]) + f"{self.dim}d"

    def _build_figure_path(self, suffix):
        return str(self.output_dir / f"{self._generate_fig_name()}_{suffix}.pdf")

    def style_axes(self, ax):
        ax.xaxis.label.set_size(self.axis_label_fontsize)
        ax.yaxis.label.set_size(self.axis_label_fontsize)
        if hasattr(ax, "zaxis"):
            ax.zaxis.label.set_size(self.axis_label_fontsize)
        ax.tick_params(axis="both", which="both", labelsize=self.tick_label_fontsize)
        if hasattr(ax, "zaxis"):
            ax.tick_params(axis="z", which="both", labelsize=self.tick_label_fontsize)

    def style_legend(self, legend):
        if legend is None:
            return
        for text in legend.get_texts():
            text.set_fontsize(self.legend_fontsize)
        frame = legend.get_frame()
        if frame is not None:
            frame.set_alpha(self.legend_framealpha)
            frame.set_edgecolor("black")
            frame.set_linewidth(1.0)
            frame.set_facecolor("white")

    def save_figure(self, fig, figure_path):
        save_kwargs = {"pad_inches": self.save_pad_inches}
        if self.save_bbox_inches is not None:
            save_kwargs["bbox_inches"] = self.save_bbox_inches
        fig.savefig(figure_path, **save_kwargs)

    # Backward-compatible helper methods.
    def _get_quiver_data(self, result, step=10):
        return get_quiver_data(result, step=step)

    def _set_time_axis_labels(self, ax, ylabel):
        return set_time_axis_labels(ax, ylabel)

    def _plot_masses_2d(self, ax, ao, planet_size=200):
        return plot_masses_2d(ax, ao, planet_size=planet_size)

    def _plot_masses_3d(self, ax, ao, projection=None, planet_size=200):
        return plot_masses_3d(ax, ao, projection=projection, planet_size=planet_size)

    def plot_thrust(self, plot_legend=True):
        return plot_thrust_impl(self, plot_legend=plot_legend)

    def plot_gravity(self, legend_mode=None):
        return plot_gravity_impl(self, legend_mode=legend_mode)

    def _plot_traj_3d_projection(self, plot_quiver=True, plot_legend=True):
        return plot_traj_3d_projection_impl(self, plot_quiver=plot_quiver, plot_legend=plot_legend)

    def plot_traj_2d(self, plot_quiver=True, plot_legend=True):
        return plot_traj_2d_impl(self, plot_quiver=plot_quiver, plot_legend=plot_legend)

    def plot_traj_3d(self, plot_quiver=True, plot_legend=True):
        return plot_traj_3d_impl(self, plot_quiver=plot_quiver, plot_legend=plot_legend)

    def plot_loss(self, x_lim=None):
        return plot_loss_impl(self, x_lim=x_lim)

    def plot_orbit_traj(self, radii_names, plot_gravity=False, plot_thrust=True):
        return plot_orbit_traj_impl(
            self,
            radii_names=radii_names,
            plot_gravity=plot_gravity,
            plot_thrust=plot_thrust,
        )

    def plot_all(self, plot_quiver=True):
        self.plot_traj_2d(plot_quiver)
        if self.dim == 3:
            self.plot_traj_3d(plot_quiver=False)
        self.plot_loss()
        self.plot_thrust()
        self.plot_gravity()

import matplotlib.pyplot as plt

from .helpers import register_plot_artifact_if_possible


def plot_loss(plotter, x_lim=None):
    plotter.fig_loss, plotter.ax_loss = plt.subplots(figsize=plotter.figsize)
    fig, ax = plotter.fig_loss, plotter.ax_loss

    for label, exp in plotter.experiments.items():
        if label.startswith("Direct collocation"):
            continue

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
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")

    if x_lim:
        ax.set_xlim(0, x_lim)
    else:
        visible_lengths = [
            len(exp["result"].loss)
            for label, exp in plotter.experiments.items()
            if not label.startswith("Direct collocation")
        ]
        if visible_lengths:
            ax.set_xlim(0, max(visible_lengths))
    fig.tight_layout()
    ax.legend()
    figure_path = plotter._build_figure_path("loss")
    fig.savefig(
        figure_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    register_plot_artifact_if_possible(figure_path)
    plt.show()

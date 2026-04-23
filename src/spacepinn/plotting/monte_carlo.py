from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .helpers import get_gravity_sources, plot_masses_2d, register_plot_artifact_if_possible


def select_best_entry(entries: list[dict]) -> dict:
    return min(entries, key=lambda entry: entry["result"].delta_v)


def print_monte_carlo_summary(grouped_entries: dict[str, list[dict]], *, title: str) -> None:
    print()
    print("*" * 92)
    print(f"[SWINGBY] Monte Carlo Summary: {title}")
    for group_name, entries in grouped_entries.items():
        if not entries:
            print(f"{group_name}: n=0 | no entries")
            continue
        delta_v = np.array([entry["result"].delta_v for entry in entries], dtype=float)
        t_total = np.array([entry["result"].t_total for entry in entries], dtype=float)
        runtime_seconds = np.array(
            [
                float(entry["result"].runtime_seconds)
                for entry in entries
                if getattr(entry["result"], "runtime_seconds", None) is not None
            ],
            dtype=float,
        )
        runtime_summary = ""
        if runtime_seconds.size:
            runtime_summary = (
                f" | runtime mean={runtime_seconds.mean():.6g} std={runtime_seconds.std(ddof=0):.6g}"
            )
        print(
            f"{group_name}: n={len(entries)} | "
            f"delta_v mean={delta_v.mean():.6g} std={delta_v.std(ddof=0):.6g} best={delta_v.min():.6g} | "
            f"t_total mean={t_total.mean():.6g} std={t_total.std(ddof=0):.6g}"
            f"{runtime_summary}"
        )
    print("*" * 92)


def plot_monte_carlo_traj_2d(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize=(7, 7),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    any_entry = next(entry for entries in grouped_entries.values() for entry in entries)

    for group_name, entries in grouped_entries.items():
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        for entry in entries:
            result = entry["result"]
            ax.plot(result.r[:, 0], result.r[:, 1], color=color, linewidth=0.8, alpha=0.18)

        best_result = best_entry["result"]
        ax.plot(best_result.r[:, 0], best_result.r[:, 1], color=color, linewidth=2.6, label=f"{group_name} best")

    ax.plot(any_entry["result"].r0[0], any_entry["result"].r0[1], "o", color="red", label=r"$r(t=0)$")
    ax.plot(any_entry["result"].rN[0], any_entry["result"].rN[1], "x", color="red", label=r"$r(t=1)$")
    plot_masses_2d(ax, get_gravity_sources(any_entry["result"]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_traj_3d(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize=(7, 7),
) -> None:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    any_entry = next(entry for entries in grouped_entries.values() for entry in entries)

    for group_name, entries in grouped_entries.items():
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        for entry in entries:
            result = entry["result"]
            ax.plot(result.r[:, 0], result.r[:, 1], result.r[:, 2], color=color, linewidth=0.8, alpha=0.18)

        best_result = best_entry["result"]
        ax.plot(
            best_result.r[:, 0],
            best_result.r[:, 1],
            best_result.r[:, 2],
            color=color,
            linewidth=2.6,
            label=f"{group_name} best",
        )

    ax.scatter(*any_entry["result"].r0, marker="o", color="red", label=r"$r(t=0)$")
    ax.scatter(*any_entry["result"].rN, marker="x", color="red", label=r"$r(t=1)$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_boxplots(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize=(11.5, 4.8),
    tick_labels: dict[str, str] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    metric_specs = [("delta_v", r"$\Delta v$"), ("t_total", "Total time")]
    display_labels = [tick_labels.get(group_name, group_name) for group_name in colors] if tick_labels else list(colors.keys())

    for ax, (metric_name, title) in zip(axes, metric_specs):
        values = [[getattr(entry["result"], metric_name) for entry in grouped_entries[group_name]] for group_name in colors]
        boxplot = ax.boxplot(values, patch_artist=True, tick_labels=display_labels)
        for patch, group_name in zip(boxplot["boxes"], colors):
            patch.set_facecolor(colors[group_name])
            patch.set_alpha(0.45)

        for index, group_name in enumerate(colors, start=1):
            group_values = np.array(values[index - 1], dtype=float)
            x_positions = np.full_like(group_values, index, dtype=float)
            jitter = np.linspace(-0.08, 0.08, len(group_values))
            ax.scatter(x_positions + jitter, group_values, color=colors[group_name], s=20, alpha=0.7)

        ax.set_title(title)
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", labelrotation=15)

    fig.tight_layout()
    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_thrust(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize=(7, 4.5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, entries in grouped_entries.items():
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        for entry in entries:
            result = entry["result"]
            ax.plot(result.t, result.F_mag, color=color, linewidth=0.8, alpha=0.18)

        best_result = best_entry["result"]
        ax.plot(best_result.t, best_result.F_mag, color=color, linewidth=2.6, label=f"{group_name} best")

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Thrust magnitude")
    ax.set_xlim(0, 1)
    ax.legend()
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()


def plot_monte_carlo_gravity(
    grouped_entries: dict[str, list[dict]],
    *,
    colors: dict[str, str],
    output_dir: str | Path,
    fig_name: str,
    figsize=(7, 4.5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for group_name, entries in grouped_entries.items():
        color = colors[group_name]
        best_entry = select_best_entry(entries)
        for entry in entries:
            result = entry["result"]
            ax.plot(result.t, result.a_mag, color=color, linewidth=0.8, alpha=0.10)
            ax.plot(result.t, result.G_mag, color=color, linewidth=0.8, alpha=0.10, linestyle="--")

        best_result = best_entry["result"]
        ax.plot(best_result.t, best_result.a_mag, color=color, linewidth=2.6, label=f"{group_name} best RFM")
        ax.plot(
            best_result.t,
            best_result.G_mag,
            color=color,
            linewidth=2.0,
            linestyle="--",
            label=f"{group_name} best Gravity",
        )

    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Gravity / Required Force magnitude")
    ax.set_xlim(0, 1)
    ax.legend(ncol=2)
    fig.tight_layout()

    figure_path = Path(output_dir) / f"{fig_name}.pdf"
    fig.savefig(figure_path, bbox_inches="tight", pad_inches=0.05)
    register_plot_artifact_if_possible(figure_path)
    plt.show()

import json
from pathlib import Path

import numpy as np


def get_quiver_data(result, step=10, count=None):
    if count is not None:
        sample_count = max(1, min(int(count), int(len(result.r))))
        indices = np.linspace(0, len(result.r) - 1, num=sample_count, dtype=int)
        r_q = result.r[indices, :]
        G_q = result.G[indices, :]
        F_q = result.F[indices, :]
        return r_q, G_q, F_q

    r_q = result.r[::step, :]
    G_q = result.G[::step, :]
    F_q = result.F[::step, :]
    return r_q, G_q, F_q


def set_time_axis_labels(ax, ylabel, *, plot_legend=True):
    ax.set_xlabel("Normalized time")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    if plot_legend:
        return ax.legend()
    return None


def get_gravity_sources(result):
    return getattr(result, "gravity_sources", getattr(result, "ao", None))


def _compute_mass_marker_sizes(
    gravity_sources,
    planet_size=200,
    *,
    linear_span=1.0,
    compressed_span=0.75,
    compression_threshold=8.0,
):
    """Return marker areas that stay readable while still reflecting mass ordering."""
    if gravity_sources is None:
        return np.asarray([], dtype=float)

    try:
        if len(gravity_sources) == 0:
            return np.asarray([], dtype=float)
    except TypeError:
        return np.asarray([], dtype=float)

    masses = np.asarray([mass for *_, mass in gravity_sources], dtype=float)
    if masses.size == 0:
        return masses

    masses = np.where(np.isfinite(masses) & (masses > 0), masses, 1.0)
    mass_min = float(np.min(masses))
    mass_max = float(np.max(masses))

    if mass_max == mass_min:
        return np.full_like(masses, float(planet_size), dtype=float)

    spread = mass_max / mass_min
    if spread <= compression_threshold:
        scaled = (masses - mass_min) / (mass_max - mass_min)
        span = linear_span
    else:
        scaled = np.log1p(masses / mass_min) / np.log1p(mass_max / mass_min)
        span = compressed_span

    # Keep the size differences readable without allowing a dominant body to fill the axes.
    return float(planet_size) * (1.0 + span * (scaled - 0.5))


def plot_masses_2d(ax, gravity_sources, planet_size=200):
    if gravity_sources is None:
        return

    sizes = _compute_mass_marker_sizes(gravity_sources, planet_size=planet_size)
    colors = ["#006400", "#228B22", "#6B8E23"]
    for i, (x, y, mass) in enumerate(gravity_sources):
        ax.scatter(
            x,
            y,
            s=sizes[i],
            color=colors[i % len(colors)],
            marker="o",
            label=f"$GM_{i+1}={mass}$",
        )


def plot_masses_3d(ax, gravity_sources, projection=None, planet_size=200):
    if gravity_sources is None:
        return

    sizes = _compute_mass_marker_sizes(gravity_sources, planet_size=planet_size)
    colors = ["#006400", "#228B22", "#6B8E23"]
    for i, (x, y, z, mass) in enumerate(gravity_sources):
        color = colors[i % len(colors)]
        if projection == "3d":
            ax.scatter(
                x,
                y,
                z,
                s=sizes[i],
                color=color,
                marker="o",
                label=f"$GM_{i+1}={mass}$",
            )
        elif projection == "2d":
            ax[0, 0].scatter(
                x,
                y,
                s=sizes[i],
                color=color,
                marker="o",
                label=f"$GM_{i+1}={mass}$",
            )
            ax[0, 1].scatter(
                x,
                z,
                s=sizes[i],
                color=color,
                marker="o",
                label=rf"$GM_{i+1}={mass}$",
            )
            ax[1, 0].scatter(
                y,
                z,
                s=sizes[i],
                color=color,
                marker="o",
                label=f"$GM_{i+1}={mass}$",
            )


def all_z_components_smaller_than_one(values):
    return np.all(np.abs(values) < 1)


def register_plot_artifact_if_possible(figure_path):
    figure_path = Path(figure_path)
    artifact_index = figure_path.parent.parent / "index.json"
    if not artifact_index.exists():
        return

    try:
        payload = json.loads(artifact_index.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {"artifacts": []}

    artifacts = payload.get("artifacts", [])
    run_dir = figure_path.parents[2]
    rel_path = str(figure_path.relative_to(run_dir))
    entry = {"kind": "plot", "path": rel_path}
    if entry not in artifacts:
        artifacts.append(entry)
        artifact_index.write_text(json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True), encoding="utf-8")

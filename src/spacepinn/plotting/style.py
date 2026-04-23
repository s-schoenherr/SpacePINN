from __future__ import annotations

from typing import Any

POSITION_BLUE = "#1f77b4"
VANILLA_ORANGE = "#ff7f0e"
KINEMATIC_GREEN = "#2ca02c"
OPENGODDARD_CHARCOAL = "#4d4d4d"
OPENGODDARD_REFERENCE_GREEN = "#2ca02c"


PALETTE = {
    "position": POSITION_BLUE,
    "vanilla": VANILLA_ORANGE,
    "kinematic": KINEMATIC_GREEN,
    "opengoddard": OPENGODDARD_CHARCOAL,
    "opengoddard_reference": OPENGODDARD_REFERENCE_GREEN,
}


def plotting_style(*, color: str, linestyle: str, quiver_scale: float | None = None) -> dict[str, Any]:
    style: dict[str, Any] = {"color": color, "linestyle": linestyle}
    if quiver_scale is not None:
        style["quiver_scale"] = quiver_scale
    return style


def infer_plotting_style(*, label: str | None, source: str | None) -> dict[str, Any]:
    label = label or ""
    source = source or ""

    if source == "opengoddard" or label.startswith("Direct collocation"):
        return plotting_style(color=PALETTE["opengoddard"], linestyle="dashdot")
    if label.startswith("Geometric") or label.startswith("Position"):
        return plotting_style(color=PALETTE["position"], linestyle="solid", quiver_scale=20)
    if label.startswith("oPINN"):
        return plotting_style(color=PALETTE["vanilla"], linestyle="dashed")
    if label.startswith("Kinematic"):
        return plotting_style(color=PALETTE["kinematic"], linestyle="dashdot")
    return {}


def resolve_plotting_style(
    *,
    label: str | None,
    source: str | None,
    existing_plotting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing_plotting = dict(existing_plotting or {})
    semantic = infer_plotting_style(label=label, source=source)
    if not semantic:
        return existing_plotting

    resolved = dict(existing_plotting)
    # Explicit experiment-level plotting should win over semantic defaults.
    for key, value in semantic.items():
        resolved.setdefault(key, value)
    return resolved

from spacepinn.plotting.style import PALETTE, resolve_plotting_style


def test_explicit_plotting_style_beats_semantic_defaults():
    resolved = resolve_plotting_style(
        label="Baseline (OpenGoddard)",
        source="opengoddard",
        existing_plotting={"color": "#123456", "linestyle": "solid"},
    )

    assert resolved["color"] == "#123456"
    assert resolved["linestyle"] == "solid"


def test_semantic_defaults_still_fill_missing_plotting_fields():
    resolved = resolve_plotting_style(
        label="Baseline (OpenGoddard)",
        source="opengoddard",
        existing_plotting={"trajectory_linestyle": "solid"},
    )

    assert resolved["color"] == PALETTE["opengoddard"]
    assert resolved["linestyle"] == "dashdot"
    assert resolved["trajectory_linestyle"] == "solid"

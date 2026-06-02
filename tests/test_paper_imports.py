from __future__ import annotations

import importlib
from pathlib import Path


CURATED_MODULES = [
    "spacepinn.paper.swingby_2d",
    "spacepinn.paper.swingby_3d",
    "spacepinn.paper.orbit_transfer_fixed_angle",
    "spacepinn.paper.orbit_transfer_free_angle",
    "spacepinn.paper.rendezvous_hold_point_eci",
    "spacepinn.paper.appendix.boundary_weight_search_2d",
    "spacepinn.paper.appendix.boundary_weight_search_3d",
    "spacepinn.paper.appendix.static_total_time_sweep",
]


def test_curated_paper_modules_import():
    for module in CURATED_MODULES:
        importlib.import_module(module)


def test_curated_paper_entry_points_smoke(monkeypatch, tmp_path):
    monkeypatch.setenv("FAST_SMOKE", "1")

    training_modules = [
        "spacepinn.paper.swingby_2d",
        "spacepinn.paper.swingby_3d",
        "spacepinn.paper.orbit_transfer_fixed_angle",
        "spacepinn.paper.orbit_transfer_free_angle",
        "spacepinn.paper.rendezvous_hold_point_eci",
    ]
    for module_name in training_modules:
        module = importlib.import_module(module_name)
        kwargs = {"skip_plots": True, "print_summary": False, "smoke": True}
        if module_name in {"spacepinn.paper.swingby_2d", "spacepinn.paper.swingby_3d"}:
            kwargs["workers"] = 1
        module.main(**kwargs)

    appendix_modules = [
        "spacepinn.paper.appendix.boundary_weight_search_2d",
        "spacepinn.paper.appendix.boundary_weight_search_3d",
        "spacepinn.paper.appendix.static_total_time_sweep",
    ]
    for module_name in appendix_modules:
        module = importlib.import_module(module_name)
        module_output = Path(tmp_path) / module_name.rsplit(".", 1)[-1]
        kwargs = {"output_dir": module_output}
        if "boundary_weight_search" in module_name:
            kwargs["print_summary"] = False
        module.main(**kwargs)

from __future__ import annotations

import importlib


def test_swingby_soft_bc_weight_search_2d_calls_dynamic_sweep(monkeypatch):
    module = importlib.import_module("spacepinn.paper.sweeps.swingby_soft_bc_weight_search_2d")
    captured = {}

    def fake_run_boundary_weight_sweep(*, spec, workers, nan_streak_limit):
        captured["spec"] = spec
        captured["workers"] = workers
        captured["nan_streak_limit"] = nan_streak_limit
        return {"label": spec.label}

    monkeypatch.setattr(module, "run_boundary_weight_sweep", fake_run_boundary_weight_sweep)

    run = module.main(workers=2, nan_streak_limit=7)

    assert run["label"] == "swingby_soft_bc_weight_search_2d"
    assert captured["spec"].dimension == 2
    assert captured["spec"].dynamic_tof is True
    assert captured["workers"] == 2
    assert captured["nan_streak_limit"] == 7


def test_swingby_soft_bc_weight_search_3d_calls_dynamic_sweep(monkeypatch):
    module = importlib.import_module("spacepinn.paper.sweeps.swingby_soft_bc_weight_search_3d")
    captured = {}

    def fake_run_boundary_weight_sweep(*, spec, workers, nan_streak_limit):
        captured["spec"] = spec
        captured["workers"] = workers
        captured["nan_streak_limit"] = nan_streak_limit
        return {"label": spec.label}

    monkeypatch.setattr(module, "run_boundary_weight_sweep", fake_run_boundary_weight_sweep)

    run = module.main(workers=3, nan_streak_limit=9)

    assert run["label"] == "swingby_soft_bc_weight_search_3d"
    assert captured["spec"].dimension == 3
    assert captured["spec"].dynamic_tof is True
    assert captured["workers"] == 3
    assert captured["nan_streak_limit"] == 9

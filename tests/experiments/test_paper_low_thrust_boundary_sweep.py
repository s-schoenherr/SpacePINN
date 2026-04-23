from __future__ import annotations

import importlib
import math
from types import SimpleNamespace


def test_build_sweep_config_sets_explicit_alpha_and_tof_scale():
    module = importlib.import_module(
        "spacepinn.paper.sweeps.low_thrust_boundary"
    )

    config = module._build_sweep_config(target_orbit="geo", tof_scale=2.5, alpha_pi_multiplier=5.0)

    assert config["label"] == "PINN with exact BC | tof=2.50 | alpha=5.00pi"
    assert math.isclose(float(config["scenario"]["tof_scale"]), 2.5, rel_tol=0.0)
    assert math.isclose(float(config["scenario"]["alpha_pi_multiplier"]), 5.0, rel_tol=0.0)
    assert math.isclose(
        float(config["extra_parameters"]["alpha_N"].detach().cpu().item()),
        5.0 * math.pi,
        rel_tol=1e-6,
    )


def test_build_sweep_config_can_override_adam_budget():
    module = importlib.import_module(
        "spacepinn.paper.sweeps.low_thrust_boundary"
    )

    config = module._build_sweep_config(target_orbit="geo", tof_scale=1.2, alpha_pi_multiplier=2.5, n_adam=50_000)

    assert int(config["optimizer"]["n_adam"]) == 50_000
    assert int(config["optimizer"]["n_lbfgs"]) == 0


def test_main_builds_cartesian_grid_of_pinn_runs(monkeypatch):
    module = importlib.import_module(
        "spacepinn.paper.sweeps.low_thrust_boundary"
    )
    captured = {}

    def fake_run_pinn_entry(spec):
        config = spec.config_builder()
        return SimpleNamespace(
            label=config["label"],
            config=config,
            result=SimpleNamespace(delta_v=1.0, loss=[0.1], epochs_total=3),
        )

    def fake_finalize_collection(spec, *, skip_plots=False, print_summary=True):
        captured["label"] = spec.label
        captured["entries"] = spec.entries
        captured["skip_plots"] = skip_plots
        captured["print_summary"] = print_summary
        return {"label": spec.label, "entries": spec.entries}

    monkeypatch.setattr(module, "run_pinn_entry", fake_run_pinn_entry)
    monkeypatch.setattr(module, "finalize_collection", fake_finalize_collection)

    run = module.main(
        target_orbit="geo",
        tof_scales=(1.5, 2.0),
        alpha_pi_multipliers=(4.0, 5.0, 6.0),
        skip_plots=True,
        print_summary=False,
    )

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["label"] == module.COLLECTION_LABEL
    assert len(captured["entries"]) == 6
    labels = [entry.label for entry in captured["entries"]]
    assert labels[0] == "PINN with exact BC | tof=1.50 | alpha=4.00pi"
    assert labels[-1] == "PINN with exact BC | tof=2.00 | alpha=6.00pi"


def test_resolve_sweep_defaults_respect_fast_smoke(monkeypatch):
    module = importlib.import_module(
        "spacepinn.paper.sweeps.low_thrust_boundary"
    )

    monkeypatch.setenv("FAST_SMOKE", "1")

    assert module._resolve_tof_scales(None) == module.SMOKE_TOF_SCALES
    assert module._resolve_alpha_pi_multipliers(None) == module.SMOKE_ALPHA_PI_MULTIPLIERS


def test_main_supports_neighborhood_grid(monkeypatch):
    module = importlib.import_module(
        "spacepinn.paper.sweeps.low_thrust_boundary"
    )
    captured = {}

    def fake_run_pinn_entry(spec):
        config = spec.config_builder()
        return SimpleNamespace(
            label=config["label"],
            config=config,
            result=SimpleNamespace(delta_v=1.0, loss=[0.1], epochs_total=3),
        )

    def fake_finalize_collection(spec, *, skip_plots=False, print_summary=True):
        captured["entries"] = spec.entries
        return {"label": spec.label, "entries": spec.entries}

    monkeypatch.setattr(module, "run_pinn_entry", fake_run_pinn_entry)
    monkeypatch.setattr(module, "finalize_collection", fake_finalize_collection)

    module.main(
        target_orbit="geo",
        tof_center=1.2,
        tof_span=0.4,
        alpha_center=2.5,
        alpha_span=0.8,
        grid_size=3,
        n_adam=50_000,
        skip_plots=True,
        print_summary=False,
    )

    assert len(captured["entries"]) == 9
    labels = [entry.label for entry in captured["entries"]]
    assert labels[0] == "PINN with exact BC | tof=1.00 | alpha=2.10pi"
    assert labels[-1] == "PINN with exact BC | tof=1.40 | alpha=2.90pi"

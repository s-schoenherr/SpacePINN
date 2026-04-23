from __future__ import annotations

import importlib
from copy import deepcopy

from spacepinn.config.config_orbit_transfer import circular_ot_kinematic_polar_config


def test_build_config_sets_paper_defaults_without_mutating_preset():
    module = importlib.import_module("spacepinn.paper.low_thrust_transfer")
    original_optimizer = deepcopy(circular_ot_kinematic_polar_config["optimizer"])

    config = module.build_config(smoke=False)

    assert config["label"] == module.PINN_LABEL
    assert config["plotting"]["linestyle"] == "solid"
    assert config["plotting"]["trajectory_linestyle"] == "solid"
    assert config["optimizer"]["n_adam"] == module.PAPER_N_ADAM
    assert config["optimizer"]["n_lbfgs"] == module.PAPER_N_LBFGS
    assert config["optimizer"]["convergence_threshold"] == module.PAPER_CONVERGENCE_THRESHOLD

    smoke_config = module.build_config(smoke=True)
    assert smoke_config["optimizer"]["n_adam"] == 1
    assert smoke_config["optimizer"]["n_lbfgs"] == 0
    assert circular_ot_kinematic_polar_config["optimizer"]["n_adam"] == original_optimizer["n_adam"]
    assert circular_ot_kinematic_polar_config["optimizer"]["n_lbfgs"] == original_optimizer["n_lbfgs"]


def test_build_baseline_entry_uses_paper_opengoddard_budget(monkeypatch):
    module = importlib.import_module("spacepinn.paper.low_thrust_transfer")
    captured = {}

    def _fake_baseline(label, **kwargs):
        captured["label"] = label
        captured.update(kwargs)
        return {"label": label, "result": None}

    monkeypatch.setattr(module, "kinematic_ot_goddard_free_final_angle_no_alpha", _fake_baseline)

    entry = module.build_baseline_entry(smoke=False)

    assert entry["label"] == module.BASELINE_LABEL
    assert captured["label"] == module.BASELINE_LABEL
    assert captured["max_iteration"] == module.OPENGODDARD_MAX_ITERATION


def test_main_uses_expected_collection_labels(monkeypatch):
    module = importlib.import_module("spacepinn.paper.low_thrust_transfer")
    captured = {}

    monkeypatch.setattr(
        module,
        "run_experiment_collection",
        lambda **kwargs: captured.setdefault("kwargs", kwargs) or {
            "label": kwargs["label"],
            "entries": kwargs["additional_entries"],
            "plot_output_dir": kwargs["run_root"],
            "run_dir": kwargs["run_root"],
        },
    )
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda **kwargs: {"label": module.BASELINE_LABEL, "result": None, "source": "opengoddard"},
    )

    run = module.main(skip_plots=True, print_summary=False, smoke=True)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["kwargs"]["label"] == module.COLLECTION_LABEL
    assert captured["kwargs"]["configs"][0]["label"] == module.PINN_LABEL
    assert captured["kwargs"]["additional_entries"][0]["label"] == module.BASELINE_LABEL

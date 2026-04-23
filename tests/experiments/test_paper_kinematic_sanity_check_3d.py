from __future__ import annotations

import importlib

import numpy as np


def test_main_uses_expected_collection_label_for_kinematic_sanity_check(monkeypatch):
    module = importlib.import_module("spacepinn.paper.kinematic_sanity_check_3d")

    class DummyEntry:
        def __init__(self, label):
            self.label = label
            self.result = type("Result", (), {"v": np.array([[0, 0, 0], [0, 0, 0]], dtype=float), "t_total": 1.0})()
            self.model = None
            self.config = {}
            self.plotting = {}
            self.source = "pinn"

    monkeypatch.setattr(module, "run_pinn_entry", lambda spec: DummyEntry(spec.config_builder()["label"]))
    captured_external_spec = {}

    def fake_prepare_external_entry(spec):
        captured_external_spec["log_text"] = spec.log_text
        captured_external_spec["log_filename"] = spec.log_filename
        return {"label": spec.label, "result": spec.result, "source": spec.source}

    def fake_goddard(*args, **kwargs):
        print("OpenGoddard log from sanity check")
        return {"label": "Direct collocation", "result": type("R", (), {"v": np.array([[0,0,0],[0,0,0]], dtype=float)})(), "config": {}, "linestyle": "solid", "color": "#000"}

    monkeypatch.setattr(module, "prepare_external_entry", fake_prepare_external_entry)
    monkeypatch.setattr(module, "kinematic_sanity_check_3d_goddard", fake_goddard)
    monkeypatch.setattr(module, "finalize_collection", lambda spec, skip_plots, print_summary: {"label": spec.label, "entries": spec.entries})

    run = module.main(skip_plots=True, print_summary=False)

    assert run["label"] == "kinematic_sanity_check_3d"
    assert len(run["entries"]) == 3
    assert captured_external_spec["log_text"] == "OpenGoddard log from sanity check\n"
    assert captured_external_spec["log_filename"] == "direct_collocation_opengoddard.log"


def test_position_and_kinematic_config_builders_set_expected_labels():
    module = importlib.import_module("spacepinn.paper.kinematic_sanity_check_3d")

    position_config = module._build_position_config()
    assert position_config["optimizer"]["n_adam"] == 2000

    v0, vN = module._boundary_velocity_physical(type("Result", (), {"v": np.array([[1,2,3],[4,5,6]], dtype=float)})())
    kinematic_config = module._build_kinematic_config(v0=v0, vN=vN)
    assert kinematic_config["label"] == "Kinematic tPINN"
    assert kinematic_config["optimizer"]["n_adam"] == 2000

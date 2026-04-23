from __future__ import annotations

import importlib


def test_build_configs_smoke_for_boundary_condition_variants():
    module = importlib.import_module("spacepinn.paper.swingby_boundary_condition_variants_3d")

    configs = module.build_configs(smoke=True)

    assert len(configs) == 2
    assert configs[0]["label"].startswith("Boundary variation A=")
    assert configs[0]["optimizer"]["n_adam"] == 1
    assert configs[0]["optimizer"]["n_lbfgs"] == 0


def test_main_uses_expected_collection_label_for_boundary_condition_variants(monkeypatch):
    module = importlib.import_module("spacepinn.paper.swingby_boundary_condition_variants_3d")
    captured = {}

    def fake_run_experiment_collection(*, configs, label, run_root):
        captured["configs"] = configs
        captured["label"] = label
        return {"label": label, "entries": [], "plot_output_dir": run_root, "run_dir": run_root}

    monkeypatch.setattr(module, "run_experiment_collection", fake_run_experiment_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)

    run = module.main(skip_plots=True, print_summary=False, smoke=True)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["label"] == module.COLLECTION_LABEL
    assert len(captured["configs"]) == 2

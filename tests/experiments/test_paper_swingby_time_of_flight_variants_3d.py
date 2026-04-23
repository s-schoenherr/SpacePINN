from __future__ import annotations

import importlib


def test_build_configs_and_baselines_for_time_of_flight_variants(monkeypatch):
    module = importlib.import_module("spacepinn.paper.swingby_time_of_flight_variants_3d")

    configs = module.build_configs(smoke=True)
    assert len(configs) == 2
    assert configs[0]["label"].startswith("PINN with exact BC")
    assert configs[0]["optimizer"]["n_adam"] == 1

    monkeypatch.setattr(
        module,
        "fixed_tof_3d_goddard",
        lambda TOF, color, linestyle: {"label": f"OpenGoddard T={TOF}", "result": None, "config": None, "color": color, "linestyle": linestyle},
    )
    monkeypatch.setattr(
        module,
        "capture_baseline_entry",
        lambda builder, log_filename: {**builder(), "log_text": "solver log", "log_filename": log_filename},
    )
    additional_entries = module.build_additional_entries(smoke=True)
    assert len(additional_entries) == 1
    assert additional_entries[0]["source"] == "opengoddard"
    assert additional_entries[0]["log_text"] == "solver log"
    assert additional_entries[0]["log_filename"] == "fixed_tof_1.00_opengoddard.log"


def test_main_uses_expected_collection_label_for_time_of_flight_variants(monkeypatch):
    module = importlib.import_module("spacepinn.paper.swingby_time_of_flight_variants_3d")
    captured = {}

    def fake_run_experiment_collection(*, configs, additional_entries, label, run_root):
        captured["configs"] = configs
        captured["additional_entries"] = additional_entries
        captured["label"] = label
        return {"label": label, "entries": [], "plot_output_dir": run_root, "run_dir": run_root}

    monkeypatch.setattr(module, "run_experiment_collection", fake_run_experiment_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(module, "build_additional_entries", lambda **kwargs: [{"label": "OpenGoddard T=1.0", "result": None, "source": "opengoddard"}])

    run = module.main(skip_plots=True, print_summary=False, smoke=True)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["label"] == module.COLLECTION_LABEL
    assert len(captured["configs"]) == 2
    assert len(captured["additional_entries"]) == 1

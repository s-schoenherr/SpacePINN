from __future__ import annotations

import importlib


def test_build_configs_sets_seeded_labels_and_smoke():
    module = importlib.import_module("spacepinn.paper.monte_carlo.descent_landing_2d")

    configs = module.build_configs(smoke=True)

    assert len(configs) == module.SMOKE_NUM_SEEDS
    assert configs[0]["label"] == f"{module.PINN_LABEL} | seed={module.SEEDS[0]}"
    assert configs[0]["seed"] == module.SEEDS[0]
    assert configs[0]["optimizer"]["coordinate_system"] == "polar"
    assert configs[0]["optimizer"]["convergence_threshold"] == module.MONTE_CARLO_CONVERGENCE_THRESHOLD
    assert configs[0]["optimizer"]["n_adam"] == 1
    assert configs[0]["optimizer"]["n_lbfgs"] == 0
    assert configs[0]["scenario"]["fixed_final_angle"] is True
    assert configs[0]["scenario"]["atmosphere"] is True


def test_load_reused_baseline_entries_returns_baseline(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.descent_landing_2d")

    monkeypatch.setattr(
        module,
        "load_run",
        lambda run_dir: {
            "entries": [
                {
                    "label": module.BASELINE_LABEL,
                    "result": object(),
                    "source": "opengoddard",
                    "plotting": {},
                    "paths": {},
                }
            ]
        },
    )

    entries = module.load_reused_baseline_entries("runs/example")

    assert [entry["label"] for entry in entries] == [module.BASELINE_LABEL]
    assert all(entry["source"] == "opengoddard" for entry in entries)


def test_select_best_entry_returns_lowest_delta_v():
    module = importlib.import_module("spacepinn.paper.monte_carlo.descent_landing_2d")

    class _Result:
        def __init__(self, delta_v):
            self.delta_v = delta_v

    best = module.select_best_entry(
        [
            {"label": "a", "result": _Result(2.0)},
            {"label": "b", "result": _Result(1.5)},
            {"label": "c", "result": _Result(1.7)},
        ]
    )

    assert best["label"] == "b"


def test_main_uses_expected_collection_label_and_baseline_run(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.descent_landing_2d")
    captured = {}
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    def _fake_run_collection(*, smoke=None, baseline_run=None, **kwargs):
        captured["smoke"] = smoke
        captured["baseline_run"] = baseline_run
        return {
            "label": module.COLLECTION_LABEL,
            "entries": [],
            "plot_output_dir": str(module.RUN_ROOT),
            "run_dir": str(module.RUN_ROOT),
        }

    monkeypatch.setattr(module, "run_collection", _fake_run_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(module, "print_monte_carlo_summary", lambda *args, **kwargs: None)

    run = module.main(skip_plots=True, print_summary=False, smoke=True, baseline_run="runs/2026/04/example")

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["smoke"] is True
    assert captured["baseline_run"] == "runs/2026/04/example"


def test_main_replots_saved_run_without_retraining(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.descent_landing_2d")
    plot_calls = []
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "load_run",
        lambda run_dir: {
            "label": module.COLLECTION_LABEL,
            "run_id": "20260420_000000_descent_landing_2d_monte_carlo",
            "run_dir": str(tmp_path / "saved_run"),
            "plot_output_dir": str(tmp_path / "saved_run" / "artifacts" / "plots"),
            "entries": [{"label": f"{module.PINN_LABEL} | seed=11000", "result": object(), "source": "pinn"}],
        },
    )
    monkeypatch.setattr(module, "run_collection", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not train")))
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "plot_collection_run",
        lambda collection_run, baseline_run, output_dir=None: plot_calls.append(
            {"run_id": collection_run["run_id"], "baseline_run": baseline_run, "output_dir": output_dir}
        ),
    )

    run = module.main(
        from_run="runs/2026/04/20260420_000000_descent_landing_2d_monte_carlo",
        print_summary=False,
        baseline_run="runs/2026/04/example",
    )

    assert run["run_id"] == "20260420_000000_descent_landing_2d_monte_carlo"
    assert plot_calls == [
        {
            "run_id": "20260420_000000_descent_landing_2d_monte_carlo",
            "baseline_run": "runs/2026/04/example",
            "output_dir": None,
        }
    ]

from __future__ import annotations

import importlib
from copy import deepcopy

from spacepinn.config.config_2d import exact_bc_2d_config, soft_bc_2d_config


def test_build_configs_sets_seeded_labels_and_smoke_without_mutating_presets():
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_2d")

    original_geometric_optimizer = deepcopy(exact_bc_2d_config["optimizer"])
    original_ordinary_optimizer = deepcopy(soft_bc_2d_config["optimizer"])

    configs = module.build_configs(smoke=True)

    assert len(configs) == module.SMOKE_NUM_SEEDS * 2
    assert configs[0]["label"] == f"{module.GEOMETRIC_LABEL} | seed={module.SEEDS[0]}"
    assert configs[1]["label"] == f"{module.ORDINARY_LABEL} | seed={module.SEEDS[0]}"
    assert configs[0]["numeric_dtype"] == module.DTYPE
    assert configs[1]["numeric_dtype"] == module.DTYPE
    assert configs[0]["optimizer"]["n_adam"] == 1
    assert configs[0]["optimizer"]["n_lbfgs"] == 0
    assert configs[1]["optimizer"]["n_adam"] == 1
    assert configs[1]["optimizer"]["n_lbfgs"] == 0
    assert configs[1]["optimizer"]["w_bc"] == module.ORDINARY_LAMBDA_BC
    assert configs[1]["plotting"]["linestyle"] == "solid"
    assert configs[1]["plotting"]["trajectory_linestyle"] == "solid"
    assert exact_bc_2d_config["optimizer"]["n_adam"] == original_geometric_optimizer["n_adam"]
    assert exact_bc_2d_config["optimizer"]["n_lbfgs"] == original_geometric_optimizer["n_lbfgs"]
    assert soft_bc_2d_config["optimizer"]["n_adam"] == original_ordinary_optimizer["n_adam"]
    assert soft_bc_2d_config["optimizer"]["n_lbfgs"] == original_ordinary_optimizer["n_lbfgs"]
    assert soft_bc_2d_config["optimizer"]["w_bc"] == original_ordinary_optimizer["w_bc"]


def test_group_entries_uses_paper_labels():
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_2d")

    grouped = module.group_entries(
        [
            {"label": f"{module.GEOMETRIC_LABEL} | seed=1000", "result": object()},
            {"label": f"{module.ORDINARY_LABEL} | seed=1000", "result": object()},
            {"label": f"{module.GEOMETRIC_LABEL} | seed=1001", "result": object()},
        ]
    )

    assert len(grouped[module.GEOMETRIC_LABEL]) == 2
    assert len(grouped[module.ORDINARY_LABEL]) == 1


def test_select_median_entry_uses_middle_delta_v():
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_2d")

    class DummyResult:
        def __init__(self, delta_v):
            self.delta_v = delta_v

    median_entry = module.select_median_entry(
        [
            {"result": DummyResult(4.0)},
            {"result": DummyResult(2.0)},
            {"result": DummyResult(3.0)},
        ]
    )

    assert median_entry["result"].delta_v == 3.0


def test_main_uses_expected_collection_label(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_2d")
    captured = {}
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    def _fake_run_collection(*, smoke=None, workers=1):
        captured["smoke"] = smoke
        captured["workers"] = workers
        return {
            "label": module.COLLECTION_LABEL,
            "entries": [
                {"label": f"{module.GEOMETRIC_LABEL} | seed={module.SEEDS[0]}", "result": None, "source": "pinn"},
                {"label": f"{module.ORDINARY_LABEL} | seed={module.SEEDS[0]}", "result": None, "source": "pinn"},
                {"label": module.BASELINE_LABEL, "result": None, "source": "opengoddard"},
            ],
            "plot_output_dir": str(module.RUN_ROOT),
            "run_dir": str(module.RUN_ROOT),
        }

    monkeypatch.setattr(module, "run_collection", _fake_run_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)

    run = module.main(skip_plots=True, print_summary=False, smoke=True)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["smoke"] is True
    assert captured["workers"] == 1


def test_main_replots_saved_run_without_retraining(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_2d")
    plot_calls = []
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        module,
        "load_run",
        lambda run_dir: {
            "label": module.COLLECTION_LABEL,
            "run_id": "20260410_000000_swingby_2d_monte_carlo",
            "run_dir": str(tmp_path / "saved_run"),
            "plot_output_dir": str(tmp_path / "saved_run" / "artifacts" / "plots"),
            "entries": [
                {"label": f"{module.GEOMETRIC_LABEL} | seed=1000", "result": object(), "source": "pinn"},
                {"label": f"{module.ORDINARY_LABEL} | seed=1000", "result": object(), "source": "pinn"},
            ],
        },
    )
    monkeypatch.setattr(module, "run_experiment_collection", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not train")))
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "plot_collection_run",
        lambda collection_run, output_dir=None: plot_calls.append(
            {"run_id": collection_run["run_id"], "output_dir": output_dir}
        ),
    )

    run = module.main(from_run="runs/2026/04/20260410_000000_swingby_2d_monte_carlo", print_summary=False)

    assert run["run_id"] == "20260410_000000_swingby_2d_monte_carlo"
    assert plot_calls == [{"run_id": "20260410_000000_swingby_2d_monte_carlo", "output_dir": None}]


def test_plot_collection_run_passes_baseline_into_boxplots(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_2d")
    calls = {"boxplots": [], "traj": 0, "thrust": 0, "gravity": 0}

    class DummyResult:
        delta_v = 1.0
        t_total = 2.0
        loss = [1.0, 0.5]

    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda: {"label": module.BASELINE_LABEL, "result": DummyResult(), "source": "opengoddard"},
    )
    monkeypatch.setattr(module, "print_baseline_delta_v_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "print_monte_carlo_summary", lambda grouped_entries, title: None)
    monkeypatch.setattr(module, "plot_monte_carlo_traj_2d_paper", lambda *args, **kwargs: calls.__setitem__("traj", calls["traj"] + 1))
    monkeypatch.setattr(module, "plot_monte_carlo_thrust_paper", lambda *args, **kwargs: calls.__setitem__("thrust", calls["thrust"] + 1))
    monkeypatch.setattr(module, "plot_monte_carlo_gravity_paper", lambda *args, **kwargs: calls.__setitem__("gravity", calls["gravity"] + 1))
    monkeypatch.setattr(
        module,
        "plot_monte_carlo_boxplots_paper",
        lambda grouped_entries, *, colors, output_dir, fig_name, baseline_entry, figsize=module.BOXPLOT_FIGSIZE: calls["boxplots"].append(
            {
                "labels": list(grouped_entries.keys()),
                "baseline_label": baseline_entry["label"],
                "fig_name": fig_name,
            }
        ),
    )

    module.plot_collection_run(
        {
            "entries": [
                {"label": f"{module.GEOMETRIC_LABEL} | seed=1000", "result": DummyResult(), "source": "pinn"},
                {"label": f"{module.ORDINARY_LABEL} | seed=1000", "result": DummyResult(), "source": "pinn"},
            ],
            "plot_output_dir": str(tmp_path),
        }
    )

    assert calls["traj"] == 1
    assert calls["thrust"] == 1
    assert calls["gravity"] == 1
    assert calls["boxplots"] == [
        {
            "labels": [module.GEOMETRIC_LABEL, module.ORDINARY_LABEL],
            "baseline_label": module.BASELINE_LABEL,
            "fig_name": f"{module.FIG_PREFIX}_boxplots",
        }
    ]

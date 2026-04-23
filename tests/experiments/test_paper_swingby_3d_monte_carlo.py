from __future__ import annotations

import importlib
from copy import deepcopy

import numpy as np

from spacepinn.config.config_3d import exact_bc_3d_config, pretraining_3d_config, soft_bc_3d_config


def test_build_configs_sets_seeded_labels_and_smoke_without_mutating_presets():
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")

    original_geometric_optimizer = deepcopy(exact_bc_3d_config["optimizer"])
    original_ordinary_optimizer = deepcopy(soft_bc_3d_config["optimizer"])
    original_kinematic_optimizer = deepcopy(pretraining_3d_config["optimizer"])

    configs = module.build_configs(smoke=True)

    assert len(configs) == module.SMOKE_NUM_SEEDS * 3
    assert configs[0]["label"] == f"{module.GEOMETRIC_LABEL} | seed={module.SEEDS[0]}"
    assert configs[1]["label"] == f"{module.ORDINARY_LABEL} | seed={module.SEEDS[0]}"
    assert configs[2]["label"] == f"Kinematic pretrain | seed={module.SEEDS[0]}"
    assert configs[0]["numeric_dtype"] == module.DTYPE
    assert configs[1]["numeric_dtype"] == module.DTYPE
    assert configs[2]["numeric_dtype"] == module.DTYPE
    assert configs[0]["optimizer"]["n_adam"] == 1
    assert configs[0]["optimizer"]["n_lbfgs"] == 0
    assert configs[1]["optimizer"]["n_adam"] == 1
    assert configs[1]["optimizer"]["n_lbfgs"] == 0
    assert configs[2]["optimizer"]["n_adam"] == 1
    assert configs[2]["optimizer"]["n_lbfgs"] == 0
    assert configs[1]["optimizer"]["w_bc"] == module.ORDINARY_LAMBDA_BC
    assert configs[1]["plotting"]["linestyle"] == "solid"
    assert configs[1]["plotting"]["trajectory_linestyle"] == "solid"
    assert configs[2]["plotting"]["linestyle"] == "solid"
    assert exact_bc_3d_config["optimizer"]["n_adam"] == original_geometric_optimizer["n_adam"]
    assert exact_bc_3d_config["optimizer"]["n_lbfgs"] == original_geometric_optimizer["n_lbfgs"]
    assert soft_bc_3d_config["optimizer"]["n_adam"] == original_ordinary_optimizer["n_adam"]
    assert soft_bc_3d_config["optimizer"]["n_lbfgs"] == original_ordinary_optimizer["n_lbfgs"]
    assert soft_bc_3d_config["optimizer"]["w_bc"] == original_ordinary_optimizer["w_bc"]
    assert pretraining_3d_config["optimizer"]["n_adam"] == original_kinematic_optimizer["n_adam"]
    assert pretraining_3d_config["optimizer"]["n_lbfgs"] == original_kinematic_optimizer["n_lbfgs"]


def test_group_entries_uses_paper_labels():
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")

    grouped = module.group_entries(
        [
            {"label": f"{module.GEOMETRIC_LABEL} | seed=2000", "result": object()},
            {"label": f"{module.ORDINARY_LABEL} | seed=2000", "result": object()},
            {"label": f"{module.GEOMETRIC_LABEL} | seed=2001", "result": object()},
            {"label": f"{module.PRETRAINED_LABEL} | seed=2001", "result": object()},
        ]
    )

    assert len(grouped[module.GEOMETRIC_LABEL]) == 2
    assert len(grouped[module.ORDINARY_LABEL]) == 1
    assert len(grouped[module.PRETRAINED_LABEL]) == 1


def test_select_median_entry_uses_middle_delta_v():
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")

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


def test_select_best_entry_uses_smallest_delta_v():
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")

    class DummyResult:
        def __init__(self, delta_v):
            self.delta_v = delta_v

    best_entry = module.select_best_entry(
        [
            {"result": DummyResult(4.0)},
            {"result": DummyResult(2.0)},
            {"result": DummyResult(3.0)},
        ]
    )

    assert best_entry["result"].delta_v == 2.0


def test_main_uses_expected_collection_label(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")
    captured = {}
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    def _fake_run_collection(*, smoke=None, workers=1):
        captured["smoke"] = smoke
        captured["workers"] = workers
        return {
            "label": module.COLLECTION_LABEL,
            "entries": [],
            "plot_output_dir": str(module.RUN_ROOT),
            "run_dir": str(module.RUN_ROOT),
        }

    monkeypatch.setattr(module, "run_collection", _fake_run_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)

    run = module.main(skip_plots=True, print_summary=False, smoke=True, workers=3)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["smoke"] is True
    assert captured["workers"] == 3


def test_main_replots_saved_run_without_retraining(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")
    plot_calls = []
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        module,
        "load_run",
        lambda run_dir: {
            "label": module.COLLECTION_LABEL,
            "run_id": "20260410_000000_swingby_3d_monte_carlo",
            "run_dir": str(tmp_path / "saved_run"),
            "plot_output_dir": str(tmp_path / "saved_run" / "artifacts" / "plots"),
            "entries": [
                {"label": f"{module.GEOMETRIC_LABEL} | seed=2000", "result": object(), "source": "pinn"},
                {"label": f"{module.ORDINARY_LABEL} | seed=2000", "result": object(), "source": "pinn"},
            ],
        },
    )
    monkeypatch.setattr(module, "run_collection", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not train")))
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "plot_collection_run",
        lambda collection_run, output_dir=None: plot_calls.append(
            {"run_id": collection_run["run_id"], "output_dir": output_dir}
        ),
    )

    run = module.main(from_run="runs/2026/04/20260410_000000_swingby_3d_monte_carlo", print_summary=False)

    assert run["run_id"] == "20260410_000000_swingby_3d_monte_carlo"
    assert plot_calls == [{"run_id": "20260410_000000_swingby_3d_monte_carlo", "output_dir": None}]


def test_run_collection_parallel_uses_context_and_returns_collection_shape(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")
    added_entries = []
    finalized = {"success": False}

    class DummyContext:
        def __init__(self, *, label, run_root):
            self.label = label
            self.run_root = run_root
            self.run_id = "parallel_run_id"
            self.run_dir = tmp_path / "run"
            self.plot_dir = self.run_dir / "artifacts" / "plots"
            self.summary_path = self.run_dir / "summary.json"
            self.manifest_path = self.run_dir / "manifest.json"
            self.config_path = self.run_dir / "config.json"

        def start(self):
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        def add_entry(self, *, label, result, config, model, source):
            added_entries.append({"label": label, "config": config, "model": model, "source": source, "result": result})

        def finalize_success(self):
            finalized["success"] = True

        def finalize_failure(self, error):
            raise AssertionError(f"unexpected failure: {error}")

    class DummyPool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, items):
            results = []
            for seed, smoke in items:
                results.append(
                    [
                        {
                            "label": f"{module.GEOMETRIC_LABEL} | seed={seed}",
                            "result": object(),
                            "config": {"label": f"{module.GEOMETRIC_LABEL} | seed={seed}", "plotting": {"color": "blue"}},
                            "plotting": {"color": "blue"},
                            "source": "pinn",
                        },
                        {
                            "label": f"{module.ORDINARY_LABEL} | seed={seed}",
                            "result": object(),
                            "config": {"label": f"{module.ORDINARY_LABEL} | seed={seed}", "plotting": {"color": "orange"}},
                            "plotting": {"color": "orange"},
                            "source": "pinn",
                        },
                        {
                            "label": f"{module.PRETRAINED_LABEL} | seed={seed}",
                            "result": object(),
                            "config": {"label": f"{module.PRETRAINED_LABEL} | seed={seed}", "plotting": {"color": "green"}},
                            "plotting": {"color": "green"},
                            "source": "pinn",
                        },
                    ]
                )
            return results

    class DummyContextFactory:
        def Pool(self, processes):
            return DummyPool(processes)

    monkeypatch.setattr(module, "RunCollectionContext", DummyContext)
    monkeypatch.setattr(module.mp, "get_context", lambda method: DummyContextFactory())
    monkeypatch.setattr(module, "get_seeds", lambda smoke=None: [2000, 2001])
    monkeypatch.setattr(module, "_run_seed", lambda seed, smoke=False: (_ for _ in ()).throw(AssertionError("pool should supply results directly")) if False else [])
    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda **kwargs: {
            "label": module.BASELINE_LABEL,
            "result": object(),
            "config": {"backend": "OpenGoddard"},
            "source": "opengoddard",
        },
    )

    collection_run = module.run_collection(smoke=False, workers=2)

    assert collection_run["label"] == module.COLLECTION_LABEL
    assert collection_run["run_id"] == "parallel_run_id"
    assert len(collection_run["entries"]) == 7
    assert [entry["model"] for entry in collection_run["entries"]] == [None] * 7
    assert len(added_entries) == 7
    assert added_entries[0]["label"] == f"{module.GEOMETRIC_LABEL} | seed=2000"
    assert added_entries[2]["label"] == f"{module.PRETRAINED_LABEL} | seed=2000"
    assert added_entries[-1]["label"] == module.BASELINE_LABEL
    assert finalized["success"] is True


def test_run_seed_returns_three_entries(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")

    class DummyHistory:
        def __init__(self):
            self.loss = [1.0]
            self.loss_physics = [0.5]
            self.loss_bc = [0.25]

    class DummyResult:
        def __init__(self, label):
            self.label = label
            self.t_total = 1.5
            self.loss = [1.0]
            self.loss_physics = [0.5]
            self.loss_bc = [0.25]
            self.history = DummyHistory()

        def _sync_legacy_attributes(self):
            self.loss = list(self.history.loss)
            self.loss_physics = list(self.history.loss_physics)
            self.loss_bc = list(self.history.loss_bc)

    def fake_execute(config, model=None):
        return config, object() if model is None else model, DummyResult(config["label"])

    monkeypatch.setattr(module, "_execute_config", fake_execute)
    monkeypatch.setattr(module, "build_pretrained_model", lambda config_runtime, source_model: object())
    monkeypatch.setattr(module, "execute_single_experiment", lambda config_runtime, model=None: (model, DummyResult(config_runtime["label"])))
    monkeypatch.setattr(module, "_prepare_runtime_config", lambda config: config)

    entries = module._run_seed(2000, smoke=True)

    assert [entry["label"] for entry in entries] == [
        f"{module.GEOMETRIC_LABEL} | seed=2000",
        f"{module.ORDINARY_LABEL} | seed=2000",
        f"{module.PRETRAINED_LABEL} | seed=2000",
    ]


def test_plot_collection_run_passes_baseline_into_boxplots(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.swingby_3d")
    calls = {"boxplots": [], "traj": 0, "thrust": 0, "gravity": 0}

    class DummyResult:
        delta_v = 1.0
        t_total = 2.0
        loss = [1.0, 0.5]
        r = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        r0 = np.array([0.0, 0.0, 0.0])
        rN = np.array([1.0, 1.0, 1.0])
        t = np.array([0.0, 1.0])
        F_mag = np.array([0.1, 0.2])
        a_mag = np.array([0.3, 0.4])
        G_mag = np.array([0.5, 0.6])

    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda: {"label": module.BASELINE_LABEL, "result": DummyResult(), "source": "opengoddard"},
    )
    monkeypatch.setattr(module, "print_monte_carlo_summary", lambda grouped_entries, title: None)
    monkeypatch.setattr(module, "plot_monte_carlo_traj_3d_paper", lambda *args, **kwargs: calls.__setitem__("traj", calls["traj"] + 1))
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
                {"label": f"{module.GEOMETRIC_LABEL} | seed=2000", "result": DummyResult(), "source": "pinn"},
                {"label": f"{module.ORDINARY_LABEL} | seed=2000", "result": DummyResult(), "source": "pinn"},
                {"label": f"{module.PRETRAINED_LABEL} | seed=2000", "result": DummyResult(), "source": "pinn"},
            ],
            "plot_output_dir": str(tmp_path),
        }
    )

    assert calls["traj"] == 1
    assert calls["thrust"] == 1
    assert calls["gravity"] == 1
    assert calls["boxplots"] == [
        {
            "labels": [module.GEOMETRIC_LABEL, module.ORDINARY_LABEL, module.PRETRAINED_LABEL],
            "baseline_label": module.BASELINE_LABEL,
            "fig_name": f"{module.FIG_PREFIX}_boxplots",
        }
    ]

from __future__ import annotations

import importlib
from copy import deepcopy

import numpy as np

from spacepinn.config.config_orbit_transfer import circular_ot_kinematic_polar_config


def test_build_configs_sets_seeded_labels_and_smoke_without_mutating_preset():
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")

    original_optimizer = deepcopy(circular_ot_kinematic_polar_config["optimizer"])

    configs = module.build_configs(smoke=True)

    assert len(configs) == module.SMOKE_NUM_SEEDS
    assert configs[0]["label"] == f"{module.KINEMATIC_LABEL} | seed={module.SEEDS[0]}"
    assert configs[0]["seed"] == module.SEEDS[0]
    assert configs[0]["optimizer"]["n_adam"] == 1
    assert configs[0]["optimizer"]["n_lbfgs"] == 0
    assert configs[0]["optimizer"]["convergence_threshold"] == module.MONTE_CARLO_CONVERGENCE_THRESHOLD
    assert circular_ot_kinematic_polar_config["optimizer"]["n_adam"] == original_optimizer["n_adam"]
    assert circular_ot_kinematic_polar_config["optimizer"]["n_lbfgs"] == original_optimizer["n_lbfgs"]


def test_main_uses_expected_collection_label(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")
    captured = {}
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    def _fake_run_collection(*, smoke=None, workers=1, reuse_baseline_from=None):
        captured["smoke"] = smoke
        captured["workers"] = workers
        captured["reuse_baseline_from"] = reuse_baseline_from
        return {
            "label": module.COLLECTION_LABEL,
            "entries": [],
            "plot_output_dir": str(module.RUN_ROOT),
            "run_dir": str(module.RUN_ROOT),
        }

    monkeypatch.setattr(module, "run_collection", _fake_run_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)

    run = module.main(
        skip_plots=True,
        print_summary=False,
        smoke=True,
        workers=3,
        reuse_baseline_from="runs/2026/04/existing_baseline_run",
    )

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["smoke"] is True
    assert captured["workers"] == 3
    assert captured["reuse_baseline_from"] == "runs/2026/04/existing_baseline_run"


def test_main_replots_saved_run_without_retraining(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")
    plot_calls = []
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        module,
        "load_run",
        lambda run_dir: {
            "label": module.COLLECTION_LABEL,
            "run_id": "20260410_000000_hohnmann_transfer_monte_carlo",
            "run_dir": str(tmp_path / "saved_run"),
            "plot_output_dir": str(tmp_path / "saved_run" / "artifacts" / "plots"),
            "entries": [
                {"label": f"{module.KINEMATIC_LABEL} | seed=4000", "result": object(), "source": "pinn"},
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

    run = module.main(from_run="runs/2026/04/20260410_000000_hohnmann_transfer_monte_carlo", print_summary=False)

    assert run["run_id"] == "20260410_000000_hohnmann_transfer_monte_carlo"
    assert plot_calls == [{"run_id": "20260410_000000_hohnmann_transfer_monte_carlo", "output_dir": None}]


def test_run_collection_parallel_uses_context_and_returns_collection_shape(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")
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
            return [
                {
                    "label": f"{module.KINEMATIC_LABEL} | seed={seed}",
                    "result": object(),
                    "config": {"label": f"{module.KINEMATIC_LABEL} | seed={seed}", "plotting": {"color": "blue"}},
                    "plotting": {"color": "blue"},
                    "source": "pinn",
                }
                for seed, smoke in items
            ]

    class DummyContextFactory:
        def Pool(self, processes):
            return DummyPool(processes)

    monkeypatch.setattr(module, "RunCollectionContext", DummyContext)
    monkeypatch.setattr(module.mp, "get_context", lambda method: DummyContextFactory())
    monkeypatch.setattr(module, "get_seeds", lambda smoke=None: [4000, 4001])
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
    assert len(collection_run["entries"]) == 3
    assert [entry["model"] for entry in collection_run["entries"]] == [None] * 3
    assert len(added_entries) == 3
    assert added_entries[0]["label"] == f"{module.KINEMATIC_LABEL} | seed=4000"
    assert added_entries[-1]["label"] == module.BASELINE_LABEL
    assert finalized["success"] is True


def test_run_collection_parallel_can_reuse_saved_baseline(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")
    added_entries = []

    class DummyContext:
        def __init__(self, *, label, run_root):
            self.run_id = "parallel_run_id"
            self.run_dir = tmp_path / "run"
            self.plot_dir = self.run_dir / "artifacts" / "plots"
            self.summary_path = self.run_dir / "summary.json"
            self.manifest_path = self.run_dir / "manifest.json"
            self.config_path = self.run_dir / "config.json"

        def start(self):
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        def add_entry(self, *, label, result, config, model, source, **kwargs):
            added_entries.append({"label": label, "source": source, "config": config})

        def finalize_success(self):
            return None

        def finalize_failure(self, error):
            raise AssertionError(f"unexpected failure: {error}")

    class DummyPool:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starmap(self, func, items):
            return [
                {
                    "label": f"{module.KINEMATIC_LABEL} | seed={seed}",
                    "result": object(),
                    "config": {"label": f"{module.KINEMATIC_LABEL} | seed={seed}", "plotting": {"color": "blue"}},
                    "plotting": {"color": "blue"},
                    "source": "pinn",
                }
                for seed, smoke in items
            ]

    class DummyContextFactory:
        def Pool(self, processes):
            return DummyPool()

    monkeypatch.setattr(module, "RunCollectionContext", DummyContext)
    monkeypatch.setattr(module.mp, "get_context", lambda method: DummyContextFactory())
    monkeypatch.setattr(module, "get_seeds", lambda smoke=None: [4000])
    monkeypatch.setattr(
        module,
        "load_reused_baseline_entry",
        lambda run_dir: {
            "label": module.BASELINE_LABEL,
            "result": object(),
            "config": {"backend": "OpenGoddard"},
            "source": "opengoddard",
            "plotting": {"color": "black"},
        },
    )
    monkeypatch.setattr(module, "build_baseline_entry", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should reuse baseline")))

    collection_run = module.run_collection(workers=2, reuse_baseline_from="runs/existing")

    assert collection_run["label"] == module.COLLECTION_LABEL
    assert [entry["label"] for entry in added_entries] == [f"{module.KINEMATIC_LABEL} | seed=4000", module.BASELINE_LABEL]


def test_get_baseline_entry_normalizes_style(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")
    reused_entry = {
        "label": module.BASELINE_LABEL,
        "result": object(),
        "source": "opengoddard",
        "plotting": {"linestyle": "dashdot", "trajectory_linestyle": "dashdot", "color": "yellow"},
    }

    monkeypatch.setattr(module, "get_baseline_entries", lambda entries, baseline_labels: [reused_entry])

    entry = module.get_baseline_entry({"entries": [reused_entry]})

    assert entry["plotting"]["linestyle"] == "solid"
    assert entry["plotting"]["trajectory_linestyle"] == "solid"
    assert entry["plotting"]["color"] == module.DIRECT_COLLOCATION_COLOR


def test_select_best_entry_uses_lowest_delta_v():
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")

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


def test_plot_collection_run_passes_baseline_into_boxplots(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.hohnmann_transfer")
    calls = {"boxplots": [], "thrust": 0, "gravity": 0, "orbit": 0, "loss": 0, "traj": 0}

    class DummyResult:
        delta_v = 1.0
        t_total = 2.0
        loss = [1.0, 0.5]
        r = np.array([[0.0, 0.0], [1.0, 1.0]])
        r0 = np.array([0.0, 0.0])
        rN = np.array([1.0, 1.0])
        t = np.array([0.0, 1.0])
        F_mag = np.array([0.1, 0.2])
        a_mag = np.array([0.3, 0.4])
        G_mag = np.array([0.5, 0.6])

    class DummyPlotter:
        def __init__(self, *args, **kwargs):
            pass

        def plot_traj_2d(self):
            calls["traj"] += 1

    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda: {"label": module.BASELINE_LABEL, "result": DummyResult(), "source": "opengoddard"},
    )
    monkeypatch.setattr(module, "print_monte_carlo_summary", lambda grouped_entries, title: None)
    monkeypatch.setattr(module, "TrajectoryPlotter", DummyPlotter)
    monkeypatch.setattr(module, "plot_thrust_figure", lambda *args, **kwargs: calls.__setitem__("thrust", calls["thrust"] + 1))
    monkeypatch.setattr(module, "plot_gravity_figure", lambda *args, **kwargs: calls.__setitem__("gravity", calls["gravity"] + 1))
    monkeypatch.setattr(module, "plot_orbit_figure", lambda *args, **kwargs: calls.__setitem__("orbit", calls["orbit"] + 1))
    monkeypatch.setattr(module, "plot_loss_figure", lambda *args, **kwargs: calls.__setitem__("loss", calls["loss"] + 1))
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
                {"label": f"{module.KINEMATIC_LABEL} | seed=4000", "result": DummyResult(), "source": "pinn"},
                {"label": f"{module.KINEMATIC_LABEL} | seed=4001", "result": DummyResult(), "source": "pinn"},
            ],
            "plot_output_dir": str(tmp_path),
        }
    )

    assert calls["traj"] == 1
    assert calls["thrust"] == 1
    assert calls["gravity"] == 1
    assert calls["orbit"] == 1
    assert calls["loss"] == 1
    assert calls["boxplots"] == [
        {
            "labels": [module.KINEMATIC_LABEL],
            "baseline_label": module.BASELINE_LABEL,
            "fig_name": f"{module.FIG_PREFIX}_boxplots",
        }
    ]

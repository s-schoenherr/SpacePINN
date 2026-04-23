from __future__ import annotations

import importlib
from copy import deepcopy

import numpy as np

from spacepinn.config.config_orbit_transfer import circular_ot_kinematic_polar_config


def test_build_configs_sets_seeded_labels_and_smoke_without_mutating_preset():
    module = importlib.import_module("spacepinn.paper.monte_carlo.low_thrust_transfer")

    original_optimizer = deepcopy(circular_ot_kinematic_polar_config["optimizer"])

    configs = module.build_configs(smoke=True, target_orbit="geo")

    assert len(configs) == module.SMOKE_NUM_SEEDS
    assert configs[0]["label"] == f"{module.PINN_LABEL} | seed={module.SEEDS[0]}"
    assert configs[0]["seed"] == module.SEEDS[0]
    assert configs[0]["optimizer"]["n_adam"] == 1
    assert configs[0]["optimizer"]["n_lbfgs"] == 0
    assert configs[0]["optimizer"]["convergence_threshold"] == module.MONTE_CARLO_CONVERGENCE_THRESHOLD
    assert circular_ot_kinematic_polar_config["optimizer"]["n_adam"] == original_optimizer["n_adam"]
    assert circular_ot_kinematic_polar_config["optimizer"]["n_lbfgs"] == original_optimizer["n_lbfgs"]


def test_main_uses_expected_collection_label(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.low_thrust_transfer")
    captured = {}
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    def _fake_run_collection(
        *,
        target_orbit="geo",
        terminal_angle_pi=None,
        time_guess_scale=None,
        extra_turns=None,
        tof_scale=None,
        smoke=None,
        workers=1,
        baseline_run=None,
    ):
        captured["target_orbit"] = target_orbit
        captured["terminal_angle_pi"] = terminal_angle_pi
        captured["time_guess_scale"] = time_guess_scale
        captured["extra_turns"] = extra_turns
        captured["tof_scale"] = tof_scale
        captured["smoke"] = smoke
        captured["workers"] = workers
        captured["baseline_run"] = baseline_run
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
        target_orbit="heo",
        terminal_angle_pi=2.1,
        time_guess_scale=1.09,
        baseline_run="runs/2026/04/example_low_thrust_transfer",
    )

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["target_orbit"] == "heo"
    assert captured["terminal_angle_pi"] == 2.1
    assert captured["time_guess_scale"] == 1.09
    assert captured["baseline_run"] == "runs/2026/04/example_low_thrust_transfer"
    assert captured["smoke"] is True
    assert captured["workers"] == 3


def test_main_replots_saved_run_without_retraining(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.low_thrust_transfer")
    plot_calls = []
    monkeypatch.setattr(module, "persist_paper_monte_carlo_aggregate_summary", lambda *args, **kwargs: None)

    monkeypatch.setattr(
        module,
        "load_run",
        lambda run_dir: {
            "label": module.COLLECTION_LABEL,
            "run_id": "20260410_000000_low_thrust_transfer_monte_carlo",
            "run_dir": str(tmp_path / "saved_run"),
            "plot_output_dir": str(tmp_path / "saved_run" / "artifacts" / "plots"),
            "entries": [
                {"label": f"{module.PINN_LABEL} | seed=5000", "result": object(), "source": "pinn"},
            ],
        },
    )
    monkeypatch.setattr(module, "run_collection", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not train")))
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "plot_collection_run",
        lambda collection_run, **kwargs: plot_calls.append(
            {"run_id": collection_run["run_id"], "target_orbit": kwargs["target_orbit"], "output_dir": kwargs["output_dir"]}
        ),
    )

    run = module.main(
        from_run="runs/2026/04/20260410_000000_low_thrust_transfer_monte_carlo",
        target_orbit="heo",
        print_summary=False,
    )

    assert run["run_id"] == "20260410_000000_low_thrust_transfer_monte_carlo"
    assert plot_calls == [{"run_id": "20260410_000000_low_thrust_transfer_monte_carlo", "target_orbit": "heo", "output_dir": None}]


def test_run_collection_parallel_uses_context_and_returns_collection_shape(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.low_thrust_transfer")
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
                    "label": f"{module.PINN_LABEL} | seed={seed}",
                    "result": object(),
                    "config": {"label": f"{module.PINN_LABEL} | seed={seed}", "plotting": {"color": "green"}},
                    "plotting": {"color": "green"},
                    "source": "pinn",
                }
                for seed, target_orbit, terminal_angle_pi, time_guess_scale, extra_turns, tof_scale, smoke in items
            ]

    class DummyContextFactory:
        def Pool(self, processes):
            return DummyPool(processes)

    monkeypatch.setattr(module, "RunCollectionContext", DummyContext)
    monkeypatch.setattr(module.mp, "get_context", lambda method: DummyContextFactory())
    monkeypatch.setattr(module, "get_seeds", lambda smoke=None: [5000, 5001])
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

    collection_run = module.run_collection(smoke=False, workers=2, target_orbit="heo", extra_turns=2, tof_scale=2.0)

    assert collection_run["label"] == module.COLLECTION_LABEL
    assert collection_run["run_id"] == "parallel_run_id"
    assert len(collection_run["entries"]) == 3
    assert [entry["model"] for entry in collection_run["entries"]] == [None] * 3
    assert len(added_entries) == 3
    assert added_entries[0]["label"] == f"{module.PINN_LABEL} | seed=5000"
    assert added_entries[-1]["label"] == module.BASELINE_LABEL
    assert finalized["success"] is True


def test_plot_collection_run_passes_baseline_into_boxplots(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.paper.monte_carlo.low_thrust_transfer")
    calls = {"boxplots": [], "thrust": 0, "gravity": 0, "orbit": 0, "loss": 0}

    class DummyResult:
        delta_v = 1.0
        t_total = 2.0
        loss = [1.0, 0.5]
        r = np.array([[0.0, 0.0], [1.0, 1.0]])
        t = np.array([0.0, 1.0])
        F_mag = np.array([0.1, 0.2])
        a_mag = np.array([0.3, 0.4])
        G_mag = np.array([0.5, 0.6])

    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda **kwargs: {"label": module.BASELINE_LABEL, "result": DummyResult(), "source": "opengoddard"},
    )
    monkeypatch.setattr(module, "print_monte_carlo_summary", lambda grouped_entries, title: None)
    monkeypatch.setattr(module, "plot_loss_figure", lambda *args, **kwargs: calls.__setitem__("loss", calls["loss"] + 1))
    monkeypatch.setattr(module, "plot_thrust_figure", lambda *args, **kwargs: calls.__setitem__("thrust", calls["thrust"] + 1))
    monkeypatch.setattr(module, "plot_gravity_figure", lambda *args, **kwargs: calls.__setitem__("gravity", calls["gravity"] + 1))
    monkeypatch.setattr(module, "plot_orbit_figure", lambda *args, **kwargs: calls.__setitem__("orbit", calls["orbit"] + 1))
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
                {"label": f"{module.PINN_LABEL} | seed=5000", "result": DummyResult(), "source": "pinn"},
                {"label": f"{module.PINN_LABEL} | seed=5001", "result": DummyResult(), "source": "pinn"},
            ],
            "plot_output_dir": str(tmp_path),
        },
        target_orbit="geo",
    )

    assert calls["loss"] == 1
    assert calls["thrust"] == 1
    assert calls["gravity"] == 1
    assert calls["orbit"] == 1
    assert calls["boxplots"] == [
        {
            "labels": [module.PINN_LABEL],
            "baseline_label": module.BASELINE_LABEL,
            "fig_name": f"{module.FIG_PREFIX}_boxplots",
        }
    ]


def test_get_baseline_entry_can_reuse_saved_run(monkeypatch):
    module = importlib.import_module("spacepinn.paper.monte_carlo.low_thrust_transfer")
    reused_entry = {
        "label": module.BASELINE_LABEL,
        "result": object(),
        "source": "opengoddard",
        "plotting": {"linestyle": "dashdot", "trajectory_linestyle": "dashdot", "color": "pink"},
    }

    monkeypatch.setattr(
        module,
        "load_run",
        lambda run_dir: {
            "entries": [
                reused_entry,
            ]
        },
    )
    monkeypatch.setattr(
        module,
        "get_baseline_entries",
        lambda entries, baseline_labels: [reused_entry],
    )

    entry = module.get_baseline_entry({"entries": []}, baseline_run="runs/2026/04/example_low_thrust_transfer")

    assert entry["label"] == module.BASELINE_LABEL
    assert entry["plotting"]["linestyle"] == "solid"
    assert entry["plotting"]["trajectory_linestyle"] == "solid"
    assert entry["plotting"]["color"] == module.BASELINE_COLOR

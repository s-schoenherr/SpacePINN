from __future__ import annotations

import importlib
from copy import deepcopy

from spacepinn.config.config_orbit_transfer import circular_ot_kinematic_polar_config


def test_build_config_sets_paper_label_and_smoke_without_mutating_preset():
    module = importlib.import_module("spacepinn.paper.hohnmann_transfer")

    original_optimizer = deepcopy(circular_ot_kinematic_polar_config["optimizer"])

    config = module.build_config(smoke=False)

    assert config["label"] == module.KINEMATIC_LABEL
    assert config["plotting"]["linestyle"] == "solid"
    assert config["plotting"]["trajectory_linestyle"] == "solid"
    assert config["optimizer"]["n_adam"] == module.PAPER_N_ADAM
    assert module.PAPER_N_ADAM == 10_000
    assert config["optimizer"]["n_lbfgs"] == module.PAPER_N_LBFGS
    assert config["optimizer"]["convergence_threshold"] == module.PAPER_CONVERGENCE_THRESHOLD
    smoke_config = module.build_config(smoke=True)
    assert smoke_config["optimizer"]["n_adam"] == 1
    assert smoke_config["optimizer"]["n_lbfgs"] == 0
    assert circular_ot_kinematic_polar_config["optimizer"]["n_adam"] == original_optimizer["n_adam"]
    assert circular_ot_kinematic_polar_config["optimizer"]["n_lbfgs"] == original_optimizer["n_lbfgs"]


def test_main_uses_expected_collection_labels(monkeypatch):
    module = importlib.import_module("spacepinn.paper.hohnmann_transfer")
    captured = {}

    def _fake_run_experiment_collection(*, configs, label, run_root, additional_entries):
        captured["configs"] = configs
        captured["label"] = label
        captured["run_root"] = run_root
        captured["additional_entries"] = additional_entries
        return {
            "label": label,
            "entries": [{"label": configs[0]["label"], "result": None, "source": "pinn"}] + additional_entries,
            "plot_output_dir": run_root,
            "run_dir": run_root,
        }

    monkeypatch.setattr(module, "run_experiment_collection", _fake_run_experiment_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda **kwargs: {"label": module.BASELINE_LABEL, "result": None, "source": "opengoddard"},
    )

    run = module.main(skip_plots=True, print_summary=False, smoke=True)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["label"] == module.COLLECTION_LABEL
    assert captured["configs"][0]["label"] == module.KINEMATIC_LABEL
    assert captured["additional_entries"][0]["label"] == module.BASELINE_LABEL


def test_build_baseline_entry_uses_paper_opengoddard_budget(monkeypatch):
    module = importlib.import_module("spacepinn.paper.hohnmann_transfer")
    captured = {}

    def _fake_baseline(label, **kwargs):
        captured["label"] = label
        captured.update(kwargs)
        return {"label": label, "result": None}

    monkeypatch.setattr(module, "kinematic_ot_goddard", _fake_baseline)

    entry = module.build_baseline_entry(smoke=False)

    assert entry["label"] == module.BASELINE_LABEL
    assert captured["label"] == module.BASELINE_LABEL
    assert captured["max_iteration"] == module.OPENGODDARD_MAX_ITERATION


def test_main_configures_paper_plotter_styling(monkeypatch):
    module = importlib.import_module("spacepinn.paper.hohnmann_transfer")
    events = []
    init_calls = []
    loss_calls = []
    thrust_calls = []
    gravity_calls = []
    orbit_calls = []

    class DummyPlotter:
        def __init__(self, entries, dim, figsize, fig_prefix, output_dir):
            init_calls.append(
                {
                    "labels": [entry["label"] for entry in entries],
                    "dim": dim,
                    "figsize": figsize,
                    "fig_prefix": fig_prefix,
                    "output_dir": output_dir,
                }
            )
            self.main_linewidth = None
            self.secondary_linewidth = None

        def plot_traj_2d(self):
            events.append(("traj2d", self.main_linewidth, self.secondary_linewidth))

    monkeypatch.setattr(
        module,
        "run_experiment_collection",
        lambda **kwargs: {
            "label": kwargs["label"],
            "entries": [
                {"label": module.KINEMATIC_LABEL, "result": None, "source": "pinn"},
                {"label": module.BASELINE_LABEL, "result": None, "source": "opengoddard"},
            ],
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
    monkeypatch.setattr(module, "TrajectoryPlotter", DummyPlotter)
    monkeypatch.setattr(
        module,
        "plot_loss_figure",
        lambda entries, output_dir: loss_calls.append(
            {"labels": [entry["label"] for entry in entries], "output_dir": output_dir}
        ),
    )
    monkeypatch.setattr(
        module,
        "plot_thrust_figure",
        lambda entries, output_dir: thrust_calls.append(
            {"labels": [entry["label"] for entry in entries], "output_dir": output_dir}
        ),
    )
    monkeypatch.setattr(
        module,
        "plot_gravity_figure",
        lambda entries, output_dir: gravity_calls.append(
            {"labels": [entry["label"] for entry in entries], "output_dir": output_dir}
        ),
    )
    monkeypatch.setattr(
        module,
        "plot_orbit_figure",
        lambda entries, output_dir: orbit_calls.append(
            {"labels": [entry["label"] for entry in entries], "output_dir": output_dir}
        ),
    )

    module.main(skip_plots=False, print_summary=False, smoke=True)

    assert init_calls[0]["labels"] == [module.KINEMATIC_LABEL, module.BASELINE_LABEL]
    assert init_calls[0]["figsize"] == module.MAIN_FIGSIZE
    assert ("traj2d", module.MAIN_LINEWIDTH, module.SECONDARY_LINEWIDTH) in events
    assert thrust_calls[0]["labels"] == [module.KINEMATIC_LABEL, module.BASELINE_LABEL]
    assert gravity_calls[0]["labels"] == [module.KINEMATIC_LABEL, module.BASELINE_LABEL]
    assert orbit_calls[0]["labels"] == [module.KINEMATIC_LABEL, module.BASELINE_LABEL]
    assert loss_calls[0]["labels"] == [module.KINEMATIC_LABEL, module.BASELINE_LABEL]

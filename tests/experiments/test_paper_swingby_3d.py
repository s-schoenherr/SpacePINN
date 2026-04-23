from __future__ import annotations

import importlib
from copy import deepcopy

from spacepinn.config.config_3d import exact_bc_3d_config, soft_bc_3d_config


def test_build_configs_sets_paper_labels_and_smoke_without_mutating_presets():
    module = importlib.import_module("spacepinn.paper.swingby_3d")

    original_geometric_optimizer = deepcopy(exact_bc_3d_config["optimizer"])
    original_ordinary_optimizer = deepcopy(soft_bc_3d_config["optimizer"])

    configs = module.build_configs(smoke=True)

    assert [config["label"] for config in configs] == [
        module.GEOMETRIC_LABEL,
        module.ORDINARY_LABEL,
    ]
    assert configs[0]["numeric_dtype"] == module.DTYPE
    assert configs[1]["numeric_dtype"] == module.DTYPE
    assert configs[0]["optimizer"]["n_adam"] == 1
    assert configs[0]["optimizer"]["n_lbfgs"] == 0
    assert configs[1]["optimizer"]["n_adam"] == 1
    assert configs[1]["optimizer"]["n_lbfgs"] == 0
    assert configs[1]["plotting"]["linestyle"] == "solid"
    assert configs[1]["plotting"]["trajectory_linestyle"] == "solid"
    assert configs[1]["optimizer"]["w_bc"] == module.ORDINARY_LAMBDA_BC
    assert exact_bc_3d_config["optimizer"]["n_adam"] == original_geometric_optimizer["n_adam"]
    assert exact_bc_3d_config["optimizer"]["n_lbfgs"] == original_geometric_optimizer["n_lbfgs"]
    assert soft_bc_3d_config["optimizer"]["n_adam"] == original_ordinary_optimizer["n_adam"]
    assert soft_bc_3d_config["optimizer"]["n_lbfgs"] == original_ordinary_optimizer["n_lbfgs"]
    assert soft_bc_3d_config["optimizer"]["w_bc"] == original_ordinary_optimizer["w_bc"]


def test_main_uses_expected_collection_labels(monkeypatch):
    module = importlib.import_module("spacepinn.paper.swingby_3d")
    captured = {}

    def _fake_run_experiment_collection(*, configs, label, run_root, additional_entries):
        captured["configs"] = configs
        captured["label"] = label
        captured["run_root"] = run_root
        captured["additional_entries"] = additional_entries
        return {
            "label": label,
            "entries": [
                {"label": config["label"], "result": None, "source": "pinn"}
                for config in configs
            ]
            + additional_entries,
            "plot_output_dir": run_root,
            "run_dir": run_root,
        }

    monkeypatch.setattr(module, "run_experiment_collection", _fake_run_experiment_collection)
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "build_warmstart_entry",
        lambda smoke=None: {"label": module.GEOMETRIC_WARMSTART_LABEL, "result": None, "source": "pinn"},
    )
    monkeypatch.setattr(
        module,
        "build_baseline_entry",
        lambda **kwargs: {"label": module.BASELINE_LABEL, "result": None, "source": "opengoddard"},
    )

    run = module.main(skip_plots=True, print_summary=False, smoke=True)

    assert run["label"] == module.COLLECTION_LABEL
    assert captured["label"] == module.COLLECTION_LABEL
    assert [config["label"] for config in captured["configs"]] == [
        module.GEOMETRIC_LABEL,
        module.ORDINARY_LABEL,
    ]
    assert captured["additional_entries"][0]["label"] == module.GEOMETRIC_WARMSTART_LABEL
    assert captured["additional_entries"][1]["label"] == module.BASELINE_LABEL


def test_main_configures_paper_plotter_styling(monkeypatch):
    module = importlib.import_module("spacepinn.paper.swingby_3d")
    events = []
    init_calls = []
    loss_calls = []

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
            self.entries = entries
            self.dim = dim
            self.figsize = figsize
            self.fig_prefix = fig_prefix
            self.output_dir = output_dir
            self.main_linewidth = None
            self.secondary_linewidth = None

        def plot_traj_2d(self, plot_quiver=True):
            events.append(("traj2d", plot_quiver, self.main_linewidth, self.secondary_linewidth))

        def plot_traj_3d(self, plot_quiver=True):
            events.append(("traj3d", plot_quiver, self.main_linewidth, self.secondary_linewidth))

        def plot_loss(self):
            events.append(("loss", self.main_linewidth, self.secondary_linewidth))

        def plot_thrust(self):
            events.append(("thrust", self.main_linewidth, self.secondary_linewidth))

        def plot_gravity(self, legend_mode=None):
            events.append(("gravity", legend_mode, self.main_linewidth, self.secondary_linewidth))

    monkeypatch.setattr(
        module,
        "run_experiment_collection",
        lambda **kwargs: {
            "label": kwargs["label"],
            "entries": [
                {"label": module.GEOMETRIC_LABEL, "result": None, "source": "pinn"},
                {"label": module.ORDINARY_LABEL, "result": None, "source": "pinn"},
                {"label": module.GEOMETRIC_WARMSTART_LABEL, "result": None, "source": "pinn"},
                {"label": module.BASELINE_LABEL, "result": None, "source": "opengoddard"},
            ],
            "plot_output_dir": kwargs["run_root"],
            "run_dir": kwargs["run_root"],
        },
    )
    monkeypatch.setattr(module, "print_collection_run_summary", lambda _run: None)
    monkeypatch.setattr(
        module,
        "build_warmstart_entry",
        lambda smoke=None: {"label": module.GEOMETRIC_WARMSTART_LABEL, "result": None, "source": "pinn"},
    )
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
            {
                "labels": [entry["label"] for entry in entries],
                "output_dir": output_dir,
            }
        ),
    )

    module.main(skip_plots=False, print_summary=False, smoke=True)

    assert init_calls[0]["labels"] == [
        module.GEOMETRIC_LABEL,
        module.ORDINARY_LABEL,
        module.GEOMETRIC_WARMSTART_LABEL,
        module.BASELINE_LABEL,
    ]
    assert init_calls[0]["figsize"] == module.MAIN_FIGSIZE
    assert loss_calls[0]["labels"] == [
        module.GEOMETRIC_LABEL,
        module.ORDINARY_LABEL,
        module.GEOMETRIC_WARMSTART_LABEL,
        module.BASELINE_LABEL,
    ]
    assert ("traj2d", False, module.MAIN_LINEWIDTH, module.SECONDARY_LINEWIDTH) in events
    assert ("traj3d", False, module.MAIN_LINEWIDTH, module.SECONDARY_LINEWIDTH) in events
    assert ("thrust", module.MAIN_LINEWIDTH, module.SECONDARY_LINEWIDTH) in events
    assert ("gravity", "compact", module.MAIN_LINEWIDTH, module.SECONDARY_LINEWIDTH) in events

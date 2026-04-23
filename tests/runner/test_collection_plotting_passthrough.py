from __future__ import annotations

import importlib


def test_additional_entry_top_level_plotting_is_preserved(monkeypatch, tmp_path):
    module = importlib.import_module("spacepinn.runner")

    class _FakeCollectionContext:
        def __init__(self, *, label, run_root="runs"):
            self.label = label
            self.run_root = run_root
            self.run_id = "test_run"
            self.run_dir = tmp_path / "run"
            self.plot_dir = self.run_dir / "plots"
            self.summary_path = self.run_dir / "summary.json"
            self.manifest_path = self.run_dir / "manifest.json"
            self.config_path = self.run_dir / "config.json"

        def start(self):
            return None

        def add_entry(self, **kwargs):
            return kwargs

        def finalize_success(self):
            return None

        def finalize_failure(self, error):
            raise error

    monkeypatch.setattr(module, "RunCollectionContext", _FakeCollectionContext)

    baseline_entry = {
        "label": "Baseline (OpenGoddard)",
        "source": "opengoddard",
        "result": object(),
        "color": "#4d4d4d",
        "linestyle": "solid",
        "trajectory_linestyle": "solid",
        "zorder": 2,
        "config": {"backend": "OpenGoddard"},
    }

    collection_run = module.run_experiment_collection(
        configs=[],
        additional_entries=[baseline_entry],
        label="demo",
        run_root=str(tmp_path),
    )

    loaded_entry = collection_run["entries"][0]
    assert loaded_entry["color"] == "#4d4d4d"
    assert loaded_entry["linestyle"] == "solid"
    assert loaded_entry["trajectory_linestyle"] == "solid"
    assert loaded_entry["zorder"] == 2

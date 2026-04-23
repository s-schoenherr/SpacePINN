from __future__ import annotations

import math
from pathlib import Path

from spacepinn.paper.sweeps import _boundary_weight_sweep as sweep


def test_weights_cover_requested_range():
    weights = sweep._weights(smoke=False)

    assert len(weights) == sweep.NUM_WEIGHTS
    assert math.isclose(float(weights[0]), sweep.WEIGHT_MIN, rel_tol=1e-12)
    assert math.isclose(float(weights[-1]), sweep.WEIGHT_MAX, rel_tol=1e-12)


def test_best_row_prefers_smallest_bc_loss_for_dynamic_selection():
    rows = [
        {"lambda_bc": 1e-3, "status": "ok", "had_nan": False, "min_total_loss": 1e-3, "min_bc_loss": 1e-2},
        {"lambda_bc": 1e-2, "status": "ok", "had_nan": False, "min_total_loss": 2e-3, "min_bc_loss": 5e-3},
        {"lambda_bc": 1e-1, "status": "ok", "had_nan": False, "min_total_loss": 5e-4, "min_bc_loss": 7e-3},
    ]

    best = sweep._best_row(rows, selection_metric="min_bc_loss")

    assert best is not None
    assert math.isclose(float(best["lambda_bc"]), 1e-2, rel_tol=0.0, abs_tol=1e-12)


def test_best_row_ignores_nan_rows():
    rows = [
        {"lambda_bc": 1e-3, "status": "error", "had_nan": True, "min_total_loss": None, "min_bc_loss": None},
        {"lambda_bc": 1e-2, "status": "ok", "had_nan": False, "min_total_loss": 2e-3, "min_bc_loss": 5e-3},
    ]

    best = sweep._best_row(rows, selection_metric="min_bc_loss")

    assert best is not None
    assert math.isclose(float(best["lambda_bc"]), 1e-2, rel_tol=0.0, abs_tol=1e-12)


def test_best_row_can_prefer_smallest_total_loss_for_static_selection():
    rows = [
        {"lambda_bc": 1e-3, "status": "ok", "had_nan": False, "min_total_loss": 1e-2, "min_bc_loss": 1e-3},
        {"lambda_bc": 1e-2, "status": "ok", "had_nan": False, "min_total_loss": 5e-3, "min_bc_loss": 2e-3},
        {"lambda_bc": 1e-1, "status": "ok", "had_nan": False, "min_total_loss": 7e-3, "min_bc_loss": 1e-4},
    ]

    best = sweep._best_row(rows, selection_metric="min_total_loss")

    assert best is not None
    assert math.isclose(float(best["lambda_bc"]), 1e-2, rel_tol=0.0, abs_tol=1e-12)


def test_run_boundary_weight_sweep_aborts_after_nan_streak(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sweep, "RUN_ROOT", tmp_path)
    monkeypatch.setattr(sweep, "_weights", lambda *, smoke: [1e-3, 1e-2, 1e-1, 1.0])

    def fake_run_single_weight_star(args):
        lambda_bc = float(args["lambda_bc"])
        if lambda_bc >= 1e-2:
            return {
                "lambda_bc": lambda_bc,
                "status": "error",
                "had_nan": True,
                "min_total_loss": None,
                "min_bc_loss": None,
                "final_total_loss": None,
                "final_bc_loss": None,
                "delta_v": None,
                "t_total": None,
                "runtime_seconds": None,
                "epochs_total": 0,
                "run_id": None,
                "run_dir": None,
            }
        return {
            "lambda_bc": lambda_bc,
            "status": "ok",
            "had_nan": False,
            "min_total_loss": 1.0,
            "min_bc_loss": 0.1,
            "final_total_loss": 1.0,
            "final_bc_loss": 0.1,
            "delta_v": 0.0,
            "t_total": 1.0,
            "runtime_seconds": 0.0,
            "epochs_total": 10,
            "run_id": "dummy",
            "run_dir": str(tmp_path / "dummy"),
        }

    monkeypatch.setattr(sweep, "_run_single_weight_star", fake_run_single_weight_star)

    result = sweep.run_boundary_weight_sweep(
        spec=sweep.SweepSpec(
            label="test_dynamic_nan_break",
            dimension=2,
            dynamic_tof=True,
            selection_metric="min_bc_loss",
        ),
        workers=1,
        nan_streak_limit=2,
    )

    assert result["aborted_early"] is True
    assert len(result["rows"]) == 3

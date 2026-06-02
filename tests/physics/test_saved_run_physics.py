from __future__ import annotations

import numpy as np
import pytest

from spacepinn.runner.loading import load_run


def _manual_gravity_acceleration(positions: np.ndarray, gravity_sources: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    gravity = np.zeros_like(positions, dtype=np.float64)
    for source in gravity_sources:
        source_position = source[:-1]
        gm = source[-1]
        r_diff = positions - source_position
        denominator = (np.linalg.norm(r_diff, axis=1) + eps) ** 3
        gravity -= gm * r_diff / denominator[:, None]
    return gravity


def _iter_compatible_entries(saved_run_dirs):
    for run_dir in saved_run_dirs:
        run = load_run(run_dir)
        for entry in run["entries"]:
            result = entry["result"]
            time_grid = np.asarray(getattr(result, "t", None)).reshape(-1)
            positions = np.asarray(getattr(result, "r", None))
            gravity_sources = np.asarray(getattr(result, "gravity_sources", getattr(result, "ao", None)))
            gravity = np.asarray(getattr(result, "G", None))
            acceleration = np.asarray(getattr(result, "a", None))
            thrust = np.asarray(getattr(result, "F", None))

            if time_grid.ndim != 1 or positions.ndim != 2:
                continue
            if time_grid.shape[0] != positions.shape[0]:
                continue
            if gravity_sources.ndim != 2 or gravity_sources.shape[1] != positions.shape[1] + 1:
                continue
            if gravity.shape != positions.shape:
                continue
            if acceleration.shape != positions.shape or thrust.shape != positions.shape:
                continue

            yield run_dir, entry, positions, gravity_sources, gravity, acceleration, thrust


def _native_boundary_data(entry):
    result = entry["result"]
    coordinate_system = getattr(result, "coordinate_system", None)
    config = entry.get("config") or {}

    if coordinate_system == "polar" and hasattr(result, "r_polar"):
        positions = np.asarray(result.r_polar)
        optimizer_config = config.get("optimizer", {}) if isinstance(config, dict) else {}
        boundary_start = _array_or_none(optimizer_config.get("r0"))
        boundary_end = _array_or_none(optimizer_config.get("rN"))
        return positions, boundary_start, boundary_end

    return np.asarray(getattr(result, "r", None)), _array_or_none(getattr(result, "r0", None)), _array_or_none(
        getattr(result, "rN", None)
    )


def _array_or_none(value):
    if value is None:
        return None
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _boundary_tolerance(entry, default_atol):
    if entry.get("source") == "opengoddard":
        return 1e-10
    return default_atol


def _skip_if_no_compatible_entries(checked_entries: int, reason: str):
    if checked_entries == 0:
        pytest.skip(reason)


@pytest.mark.requires_saved_runs
def test_saved_runs_present(saved_run_dirs):
    if not saved_run_dirs:
        pytest.skip("No run dirs provided. Use --run-dir ... or RUN_DIRS=...")

    for run_dir in saved_run_dirs:
        assert run_dir.exists(), f"Run dir does not exist: {run_dir}"
        assert (run_dir / "manifest.json").exists(), f"manifest.json missing in: {run_dir}"


@pytest.mark.requires_saved_runs
def test_saved_run_gravity_matches_point_mass_model(saved_run_dirs):
    if not saved_run_dirs:
        pytest.skip("No run dirs provided. Use --run-dir ... or RUN_DIRS=...")

    checked_entries = 0
    for _run_dir, _entry, positions, gravity_sources, gravity, _acceleration, _thrust in _iter_compatible_entries(
        saved_run_dirs
    ):
        expected_gravity = _manual_gravity_acceleration(positions, gravity_sources)
        np.testing.assert_allclose(gravity, expected_gravity, rtol=2e-5, atol=2e-6)
        checked_entries += 1

    _skip_if_no_compatible_entries(checked_entries, "No compatible run entries found for gravity validation.")


@pytest.mark.requires_saved_runs
def test_saved_run_dynamics_balance(saved_run_dirs):
    if not saved_run_dirs:
        pytest.skip("No run dirs provided. Use --run-dir ... or RUN_DIRS=...")

    checked_entries = 0
    for _run_dir, _entry, _positions, _gravity_sources, gravity, acceleration, thrust in _iter_compatible_entries(
        saved_run_dirs
    ):
        np.testing.assert_allclose(acceleration - gravity, thrust, rtol=2e-5, atol=2e-6)
        checked_entries += 1

    _skip_if_no_compatible_entries(checked_entries, "No compatible run entries found for dynamics validation.")


@pytest.mark.requires_saved_runs
def test_saved_run_boundary_conditions(saved_run_dirs, boundary_abs_tolerance):
    if not saved_run_dirs:
        pytest.skip("No run dirs provided. Use --run-dir ... or RUN_DIRS=...")

    checked_entries = 0
    for _run_dir, entry, _positions, _gravity_sources, _gravity, _acceleration, _thrust in _iter_compatible_entries(
        saved_run_dirs
    ):
        positions_native, boundary_start, boundary_end = _native_boundary_data(entry)
        if boundary_start is None or boundary_end is None:
            continue
        if positions_native.ndim != 2:
            continue
        if boundary_start.shape[0] != positions_native.shape[1]:
            continue
        if boundary_end.shape[0] != positions_native.shape[1]:
            continue

        atol = _boundary_tolerance(entry, boundary_abs_tolerance)
        np.testing.assert_allclose(positions_native[0], boundary_start, rtol=0.0, atol=atol)
        np.testing.assert_allclose(positions_native[-1], boundary_end, rtol=0.0, atol=atol)
        checked_entries += 1

    _skip_if_no_compatible_entries(checked_entries, "No compatible run entries found for boundary validation.")

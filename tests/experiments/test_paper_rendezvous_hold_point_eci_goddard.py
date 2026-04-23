from __future__ import annotations

import importlib
import math


def test_spaceship_matches_rendezvous_boundary_setup():
    module = importlib.import_module("spacepinn.opengoddard.rendezvous_hold_point_eci_goddard")

    spaceship = module.Spaceship(time_final_guess=900.0)
    target_speed = math.sqrt(module.GM_EARTH / (6878.0**3)) * 6878.0

    assert math.isclose(float(spaceship.r0_cart[0]), 6877.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(spaceship.r0_cart[1]), -0.3, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(spaceship.v0_cart[0]), 0.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(spaceship.v0_cart[1]), target_speed, rel_tol=0.0, abs_tol=1e-12)
    expected_rho = math.hypot(6877.0, -0.3)
    expected_alpha = math.atan2(-0.3, 6877.0)
    expected_vr = ((6877.0 * 0.0) + (-0.3 * target_speed)) / expected_rho
    expected_vt = ((-(-0.3) * 0.0) + (6877.0 * target_speed)) / expected_rho
    assert math.isclose(float(spaceship.r0[0]), expected_rho, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(spaceship.r0[1]), expected_alpha, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(spaceship.v0[0]), expected_vr, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(spaceship.v0[1]), expected_vt, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(spaceship.target_radius), 6878.0, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(float(spaceship.target_speed), target_speed, rel_tol=0.0, abs_tol=1e-12)


def test_target_state_places_hold_point_radially_outward():
    module = importlib.import_module("spacepinn.opengoddard.rendezvous_hold_point_eci_goddard")

    spaceship = module.Spaceship(time_final_guess=900.0)
    terminal = spaceship.target_state(900.0)

    radial = terminal["target_position"] / module.np.linalg.norm(terminal["target_position"])
    expected_hold = terminal["target_position"] + spaceship.hold_point_radial_offset * radial

    assert module.np.allclose(terminal["hold_position"], expected_hold)
    assert math.isclose(float(terminal["hold_rho"]), spaceship.target_radius + spaceship.hold_point_radial_offset, rel_tol=0.0)
    assert math.isclose(float(terminal["hold_vr"]), 0.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        float(terminal["hold_vt"]),
        float(terminal["hold_rho"] * terminal["mean_motion"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_wrapper_returns_expected_metadata():
    module = importlib.import_module("spacepinn.opengoddard.rendezvous_hold_point_eci_goddard")
    captured = {}

    monkey_result = {}

    def fake_solve(prob, obj, display_func, *, ftol, maxiter, label=None, details=None):
        captured["time_final_before"] = prob.time_final(-1)
        captured["ftol"] = ftol
        captured["maxiter"] = maxiter
        captured["label"] = label
        captured["details"] = details
        display_func()
        return 0.0, {"converged": False, "status_code": 9, "message": "Iteration limit reached"}

    module.solve_with_diagnostics = fake_solve

    result = module.kinematic_rendezvous_hold_point_eci_goddard(
        label="Baseline (OpenGoddard)",
        max_iteration=3,
        ftol=1e-9,
        slsqp_maxiter=7,
        time_final_guess=900.0,
    )

    assert result["label"] == "Baseline (OpenGoddard)"
    assert result["config"]["problem"] == "rendezvous_hold_point_eci_polar"
    assert result["config"]["solver"]["max_iteration"] == 3
    assert result["config"]["solver"]["slsqp_maxiter"] == 7
    assert math.isclose(float(result["config"]["solver"]["ftol"]), 1e-9, rel_tol=0.0)
    assert result["result"].coordinate_system == "polar"
    assert captured["maxiter"] == 7
    assert captured["label"] == "Baseline (OpenGoddard)"
    assert captured["details"] is None
    assert math.isclose(float(captured["ftol"]), 1e-9, rel_tol=0.0)

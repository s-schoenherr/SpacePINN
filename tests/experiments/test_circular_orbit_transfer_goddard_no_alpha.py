from __future__ import annotations

import importlib
import math

import numpy as np


def test_integrate_alpha_matches_constant_angular_rate_case():
    module = importlib.import_module(
        "spacepinn.opengoddard.circular_orbit_transfer_goddard_no_alpha"
    )

    time = np.linspace(0.0, 2.0, 5)
    radius = np.full_like(time, 2.0)
    vt = np.full_like(time, 4.0)

    alpha = module._integrate_alpha(time, radius, vt, alpha0=0.5)

    expected = 0.5 + (vt[0] / radius[0]) * time
    assert np.allclose(alpha, expected, atol=1e-12)


def test_no_alpha_solver_uses_three_states():
    module = importlib.import_module(
        "spacepinn.opengoddard.circular_orbit_transfer_goddard_no_alpha"
    )

    result = module.kinematic_ot_goddard_free_final_angle_no_alpha(
        "Baseline (OpenGoddard)",
        max_iteration=1,
        transfer_bc=module.OrbitalTransferBC(module.Orbit.LEO, module.Orbit.HEO, alpha_T=math.pi, coordinate_system="polar"),
        time_final_guess=1.0,
        time_final_upper_bound=2.0,
    )

    assert result["config"]["solver"]["num_states"] == [3]

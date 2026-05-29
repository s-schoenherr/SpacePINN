from __future__ import annotations

import numpy as np
import pytest
import torch

from spacepinn.config.transform_functions import (
    kinematic_polar_fn,
    kinematic_rendezvous_hold_point_eci_polar_fn,
)
from spacepinn.optimization.dynamics import CartesianDynamics, PolarDynamics
from spacepinn.pinn import PINN
from spacepinn.result import TrajectoryResult


def _manual_gravity_acceleration(positions: np.ndarray, gravity_sources: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    gravity = np.zeros_like(positions, dtype=np.float64)
    for source in gravity_sources:
        source_position = source[:-1]
        gm = source[-1]
        r_diff = positions - source_position
        denominator = (np.linalg.norm(r_diff, axis=1) + eps) ** 3
        gravity -= gm * r_diff / denominator[:, None]
    return gravity


@pytest.mark.parametrize(
    ("dims", "gravity_sources"),
    [
        (
            2,
            np.array(
                [
                    [5.0, -4.0, 0.7],
                    [-6.0, 8.0, 1.2],
                    [3.0, 3.0, 0.4],
                ],
                dtype=np.float64,
            ),
        ),
        (
            3,
            np.array(
                [
                    [5.0, -4.0, 2.0, 0.7],
                    [-6.0, 8.0, -3.0, 1.2],
                    [3.0, 3.0, 4.0, 0.4],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_cartesian_gravity_matches_newtonian_point_mass_sum(dims: int, gravity_sources: np.ndarray):
    t = torch.linspace(0.0, 1.0, 9, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    coordinates = [1.5 + 0.2 * (idx + 1) * t + 0.1 * (idx + 1) * t**2 for idx in range(dims)]
    r = torch.cat(coordinates, dim=1)

    state = CartesianDynamics().compute(
        r=r,
        t=t,
        t_total=torch.tensor(2.5, dtype=torch.float64),
        gravity_sources=torch.as_tensor(gravity_sources, dtype=torch.float64),
        dims=dims,
        eps=1e-8,
    )

    expected = _manual_gravity_acceleration(r.detach().numpy(), gravity_sources, eps=1e-8)
    np.testing.assert_allclose(state.G.detach().numpy(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        (state.a - state.G).detach().numpy(),
        state.F.detach().numpy(),
        rtol=0.0,
        atol=0.0,
    )


def test_polar_circular_orbit_has_zero_thrust_residual():
    mu = 398600.0
    radius = 6878.0
    mean_motion = np.sqrt(mu / radius**3)
    t_total = torch.tensor(900.0, dtype=torch.float64)

    t = torch.linspace(0.0, 1.0, 21, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    rho = radius + 0.0 * t**3
    alpha = mean_motion * t_total * t
    r = torch.cat((rho, alpha), dim=1)

    state = PolarDynamics().compute(
        r=r,
        t=t,
        t_total=t_total,
        gravity_sources=torch.tensor([[0.0, 0.0, mu]], dtype=torch.float64),
        dims=2,
        eps=0.0,
    )

    np.testing.assert_allclose(state.F_rho.detach().numpy(), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state.F_alpha.detach().numpy(), 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(state.G_alpha.detach().numpy(), 0.0, rtol=0.0, atol=1e-12)


def test_kinematic_polar_transform_enforces_physical_boundary_velocities():
    model = PINN(1, 2, 4, 2)
    model.register_parameter("t_total", torch.nn.Parameter(torch.tensor(10.0, dtype=torch.float32)))

    x0 = torch.tensor([2.0, 0.0], dtype=torch.float32)
    xN = torch.tensor([4.0, 1.0], dtype=torch.float32)
    v0 = torch.tensor([0.0, 3.0], dtype=torch.float32)
    vN = torch.tensor([0.0, 5.0], dtype=torch.float32)

    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    x = torch.zeros((11, 2), dtype=torch.float32)
    r = kinematic_polar_fn(t, x, x0=x0, xN=xN, v0=v0, vN=vN, model=model)

    rho = r[:, 0:1]
    alpha = r[:, 1:2]
    rho_t = torch.autograd.grad(rho, t, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    alpha_t = torch.autograd.grad(alpha, t, grad_outputs=torch.ones_like(alpha), create_graph=True)[0]

    assert float((rho_t[0] / model.t_total).item()) == pytest.approx(0.0, abs=1e-6)
    assert float((rho_t[-1] / model.t_total).item()) == pytest.approx(0.0, abs=1e-6)
    assert float((rho[0] * alpha_t[0] / model.t_total).item()) == pytest.approx(3.0, abs=1e-6)
    assert float((rho[-1] * alpha_t[-1] / model.t_total).item()) == pytest.approx(5.0, abs=1e-6)


def test_rendezvous_polar_transform_tracks_moving_target_terminal_state():
    target_radius = 6878.0
    mu = 398600.0
    target_speed = np.sqrt(mu / target_radius)
    hold_offset = 0.03
    t_total = 900.0
    mean_motion = np.sqrt(mu / target_radius**3)

    model = PINN(1, 2, 4, 2)
    model.register_parameter("t_total", torch.nn.Parameter(torch.tensor(t_total, dtype=torch.float64)))

    x0 = torch.tensor([target_radius - 1.0, -0.3 / target_radius], dtype=torch.float64)
    v0 = torch.tensor([0.0, target_speed], dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 11, dtype=torch.float64).reshape(-1, 1).requires_grad_(True)
    x = torch.zeros((11, 2), dtype=torch.float64)

    r = kinematic_rendezvous_hold_point_eci_polar_fn(
        t,
        x,
        x0=x0,
        v0=v0,
        target_radius=target_radius,
        target_speed=target_speed,
        hold_point_radial_offset=hold_offset,
        model=model,
    )

    rho = r[:, 0:1]
    alpha = r[:, 1:2]
    rho_t = torch.autograd.grad(rho, t, grad_outputs=torch.ones_like(rho), create_graph=True)[0]
    alpha_t = torch.autograd.grad(alpha, t, grad_outputs=torch.ones_like(alpha), create_graph=True)[0]

    assert float(r[-1, 0].item()) == pytest.approx(target_radius + hold_offset, abs=1e-9)
    assert float(r[-1, 1].item()) == pytest.approx(mean_motion * t_total, abs=1e-12)
    assert float((rho_t[-1] / model.t_total).item()) == pytest.approx(0.0, abs=1e-12)
    assert float((rho[-1] * alpha_t[-1] / model.t_total).item()) == pytest.approx(
        (target_radius + hold_offset) * mean_motion,
        rel=1e-12,
    )


def test_open_goddard_polar_result_is_normalized_to_cartesian_outputs():
    t = np.array([0.0, 2.0], dtype=np.float64)
    r = np.array([[2.0, 0.0], [2.0, np.pi / 2]], dtype=np.float64)
    v = np.array([[1.0, 3.0], [1.0, 3.0]], dtype=np.float64)
    F = np.array([[0.5, 0.25], [0.5, 0.25]], dtype=np.float64)
    G = np.array([[-1.0, 0.0], [-1.0, 0.0]], dtype=np.float64)
    ao = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    result = TrajectoryResult.from_open_goddard(
        label="polar-opengoddard",
        t=t,
        r=r,
        v=v,
        F=F,
        G=G,
        r0=r[0],
        rN=r[-1],
        ao=ao,
        coordinate_system="polar",
    )

    np.testing.assert_allclose(result.r, [[2.0, 0.0], [0.0, 2.0]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.v, [[1.0, 3.0], [-3.0, 1.0]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.F, [[0.5, 0.25], [-0.25, 0.5]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.G, [[-1.0, 0.0], [0.0, -1.0]], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.r_polar, r, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.v_polar, v, rtol=0.0, atol=1e-12)

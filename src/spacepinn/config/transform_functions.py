import torch


def R(t, x0, xN):
    """Linear endpoint interpolant on normalized time tau in [0, 1]."""
    return t * (xN - x0) + x0


def R_dot(t, x0, xN):
    """Derivative of R with respect to normalized time tau."""
    return xN - x0


def V(t, v0, vN):
    """Linear interpolation of endpoint slopes with respect to tau."""
    return t * (vN - v0) + v0


def _physical_velocity_to_state_velocity(v_boundary_physical, model, *, t, dtype):
    """Convert physical velocity dr/dt to normalized-time slope dr/dtau."""
    v_boundary_tensor = torch.as_tensor(v_boundary_physical, device=t.device, dtype=dtype)
    t_total_tensor = model.t_total.to(device=t.device, dtype=dtype).reshape(())
    return t_total_tensor * v_boundary_tensor


def position_fn(t, x, x0, xN, model):
    """Position-only exact boundary transform for Cartesian states."""
    psi = t * (1 - t)
    return R(t, x0, xN) + psi * x


def kinematic_fn(t, x, x0, xN, v0, vN, model):
    """Cartesian kinematic transform with physical endpoint velocities."""
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    xN_tensor = torch.as_tensor(xN, device=t.device, dtype=t.dtype)
    v0_tensor = _physical_velocity_to_state_velocity(v0, model, t=t, dtype=t.dtype)
    vN_tensor = _physical_velocity_to_state_velocity(vN, model, t=t, dtype=t.dtype)
    psi = t**2 * (1 - t) ** 2
    phi = 2 * t**3 - 3 * t**2 + t
    return R(t, x0_tensor, xN_tensor) + psi * x + phi * (V(t, v0_tensor, vN_tensor) - R_dot(t, x0_tensor, xN_tensor))


def _polar_state_velocity_from_physical_velocity(x_boundary, v_boundary_physical, model, *, t, dtype):
    """Convert [v_r, v_t] to [rho_tau, alpha_tau]; v_t is rho * dalpha/dt."""
    x_boundary_tensor = torch.as_tensor(x_boundary, device=t.device, dtype=dtype)
    v_boundary_tensor = torch.as_tensor(v_boundary_physical, device=t.device, dtype=dtype)
    t_total_tensor = model.t_total.to(device=t.device, dtype=dtype).reshape(())

    rho = x_boundary_tensor[0]
    vr = v_boundary_tensor[0]
    vt = v_boundary_tensor[1]

    rho_t = t_total_tensor * vr
    alpha_t = t_total_tensor * vt / rho
    return torch.stack((rho_t, alpha_t))


def kinematic_polar_fn(t, x, x0, xN, v0, vN, model, transform_only_R=False):
    """Polar kinematic transform with physical radial/tangential endpoint velocities."""
    if transform_only_R:
        alpha = x[:, 1].detach()
        xN[1] = alpha[-1]

    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    xN_tensor = torch.as_tensor(xN, device=t.device, dtype=t.dtype)
    v0_tensor = _polar_state_velocity_from_physical_velocity(x0_tensor, v0, model, t=t, dtype=t.dtype)
    vN_tensor = _polar_state_velocity_from_physical_velocity(xN_tensor, vN, model, t=t, dtype=t.dtype)

    psi = t**2 * (1 - t) ** 2
    phi = 2 * t**3 - 3 * t**2 + t
    return R(t, x0_tensor, xN_tensor) + psi * x + phi * (V(t, v0_tensor, vN_tensor) - R_dot(t, x0_tensor, xN_tensor))


def kinematic_rendezvous_hold_point_eci_polar_fn(
    t,
    x,
    x0,
    v0,
    target_radius,
    target_speed,
    hold_point_radial_offset,
    model,
):
    """Polar rendezvous transform with terminal angle from target mean motion."""
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    v0_tensor = torch.as_tensor(v0, device=t.device, dtype=t.dtype)

    radius_tensor = torch.as_tensor(target_radius, device=t.device, dtype=t.dtype).reshape(())
    speed_tensor = torch.as_tensor(target_speed, device=t.device, dtype=t.dtype).reshape(())
    offset_tensor = torch.as_tensor(hold_point_radial_offset, device=t.device, dtype=t.dtype).reshape(())
    t_total_tensor = model.t_total.to(device=t.device, dtype=t.dtype).reshape(())

    mean_motion = torch.sqrt(torch.as_tensor(398600.0, device=t.device, dtype=t.dtype) / radius_tensor**3)
    theta = mean_motion * t_total_tensor
    xN_tensor = torch.stack((radius_tensor + offset_tensor, theta))
    vN_tensor = torch.stack((torch.zeros_like(speed_tensor), (radius_tensor + offset_tensor) * mean_motion))

    return kinematic_polar_fn(
        t,
        x,
        x0=x0_tensor,
        xN=xN_tensor,
        v0=v0_tensor,
        vN=vN_tensor,
        model=model,
    )


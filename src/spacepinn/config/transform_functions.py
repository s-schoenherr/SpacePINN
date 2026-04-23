import torch
import torch.nn.functional as F


def R(t, x0, xN):
    return t * (xN - x0) + x0


def R_dot(t, x0, xN):
    return xN - x0


def V(t, v0, vN):
    return t * (vN - v0) + v0


def _physical_velocity_to_state_velocity(v_boundary_physical, model, *, t, dtype):
    v_boundary_tensor = torch.as_tensor(v_boundary_physical, device=t.device, dtype=dtype)
    t_total_tensor = model.t_total.to(device=t.device, dtype=dtype).reshape(())
    return t_total_tensor * v_boundary_tensor


def position_fn(t, x, x0, xN, model):
    """
    Transform function for the 3D position transformed PINN.
    """
    psi = t * (1 - t)
    return R(t, x0, xN) + psi * x  # phi = 0


def kinematic_fn(t, x, x0, xN, v0, vN, model):
    """
    Transform function for the Cartesian kinematic transformed PINN.
    `v0` and `vN` are physical endpoint velocities and are converted
    to the model's normalized-time state velocities using the current `t_total`.
    """
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    xN_tensor = torch.as_tensor(xN, device=t.device, dtype=t.dtype)
    v0_tensor = _physical_velocity_to_state_velocity(v0, model, t=t, dtype=t.dtype)
    vN_tensor = _physical_velocity_to_state_velocity(vN, model, t=t, dtype=t.dtype)
    psi = t**2 * (1 - t) ** 2
    phi = 2 * t**3 - 3 * t**2 + t
    return R(t, x0_tensor, xN_tensor) + psi * x + phi * (V(t, v0_tensor, vN_tensor) - R_dot(t, x0_tensor, xN_tensor))


def kinematic_rendezvous_hold_point_eci_fn(
    t,
    x,
    x0,
    v0,
    target_radius,
    target_speed,
    hold_point_radial_offset,
    model,
):
    """
    Cartesian kinematic transform for a rendezvous problem with a trainable arrival
    time. The initial chaser state is fixed in ECI, while the terminal chaser state
    is defined relative to a target continuing on a circular orbit:
      - r_c(T) = r_t(T) + d_r * e_r(T)
      - v_c(T) = v_t(T)
    """
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    v0_tensor = _physical_velocity_to_state_velocity(v0, model, t=t, dtype=t.dtype)

    radius_tensor = torch.as_tensor(target_radius, device=t.device, dtype=t.dtype).reshape(())
    speed_tensor = torch.as_tensor(target_speed, device=t.device, dtype=t.dtype).reshape(())
    offset_tensor = torch.as_tensor(hold_point_radial_offset, device=t.device, dtype=t.dtype).reshape(())
    t_total_tensor = model.t_total.to(device=t.device, dtype=t.dtype).reshape(())

    mean_motion = torch.sqrt(torch.as_tensor(398600.0, device=t.device, dtype=t.dtype) / radius_tensor**3)
    theta = mean_motion * t_total_tensor

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    xN_tensor = torch.stack(((radius_tensor + offset_tensor) * cos_theta, (radius_tensor + offset_tensor) * sin_theta))
    vN_physical = torch.stack((-speed_tensor * sin_theta, speed_tensor * cos_theta))
    vN_tensor = _physical_velocity_to_state_velocity(vN_physical, model, t=t, dtype=t.dtype)

    psi = t**2 * (1 - t) ** 2
    phi = 2 * t**3 - 3 * t**2 + t
    return R(t, x0_tensor, xN_tensor) + psi * x + phi * (V(t, v0_tensor, vN_tensor) - R_dot(t, x0_tensor, xN_tensor))


def kinematic_moon_transfer_eci_fn(
    t,
    x,
    x0,
    v0,
    moon_provider,
    llo_radius,
    llo_speed,
    model,
):
    """
    Cartesian kinematic transform for an Earth-to-Moon transfer in ECI with a
    trainable arrival time and a free Moon-relative arrival phase.
    """
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    v0_tensor = _physical_velocity_to_state_velocity(v0, model, t=t, dtype=t.dtype)

    llo_radius_tensor = torch.as_tensor(llo_radius, device=t.device, dtype=t.dtype).reshape(())
    llo_speed_tensor = torch.as_tensor(llo_speed, device=t.device, dtype=t.dtype).reshape(())
    alpha_N_tensor = model.alpha_N.to(device=t.device, dtype=t.dtype).reshape(())
    t_total_tensor = model.t_total.to(device=t.device, dtype=t.dtype).reshape(())

    moon_position_tensor, moon_velocity_tensor = moon_provider.state_torch(
        t_total_tensor,
        dtype=t.dtype,
        device=t.device,
    )
    radial_unit = torch.stack((torch.cos(alpha_N_tensor), torch.sin(alpha_N_tensor)))
    tangential_unit = torch.stack((-torch.sin(alpha_N_tensor), torch.cos(alpha_N_tensor)))
    xN_tensor = moon_position_tensor + llo_radius_tensor * radial_unit
    vN_physical = moon_velocity_tensor + llo_speed_tensor * tangential_unit
    vN_tensor = _physical_velocity_to_state_velocity(vN_physical, model, t=t, dtype=t.dtype)

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
    """
    Polar kinematic transform for a rendezvous problem with a trainable arrival
    time. The initial state is prescribed in polar coordinates, while the
    terminal chaser state follows the target's circular motion:
      - rho_c(T) = r_target + d_r
      - alpha_c(T) = theta_target(T)
      - v_r(T) = 0
      - v_t(T) = v_target
    """
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


def kinematic_rendezvous_hold_point_eci_polar_min_radius_fn(
    t,
    x,
    x0,
    v0,
    target_radius,
    target_speed,
    hold_point_radial_offset,
    model,
):
    """
    Polar kinematic transform for the rendezvous problem with an interior radius
    guard. Because the chaser starts slightly inside the target orbit, a strict
    ``rho >= target_radius`` constraint is incompatible with the exact initial
    boundary condition. This transform therefore preserves the exact endpoint
    conditions while lifting the trajectory above the target-orbit radius across
    the interior of the horizon.
    """
    del target_speed
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    v0_tensor = torch.as_tensor(v0, device=t.device, dtype=t.dtype)

    radius_tensor = torch.as_tensor(target_radius, device=t.device, dtype=t.dtype).reshape(())
    offset_tensor = torch.as_tensor(hold_point_radial_offset, device=t.device, dtype=t.dtype).reshape(())
    t_total_tensor = model.t_total.to(device=t.device, dtype=t.dtype).reshape(())

    mean_motion = torch.sqrt(torch.as_tensor(398600.0, device=t.device, dtype=t.dtype) / radius_tensor**3)
    theta = mean_motion * t_total_tensor
    xN_tensor = torch.stack((radius_tensor + offset_tensor, theta))
    vN_tensor = torch.stack((torch.zeros_like(radius_tensor), (radius_tensor + offset_tensor) * mean_motion))

    tau = t.reshape(-1, 1)
    rho0 = x0_tensor[0].reshape(())
    rhoN = xN_tensor[0].reshape(())
    vr0 = v0_tensor[0].reshape(())
    vrN = vN_tensor[0].reshape(())
    rho_dot0 = (t_total_tensor * vr0).reshape(())
    rho_dotN = (t_total_tensor * vrN).reshape(())

    rho_base = _cubic_hermite_scalar(tau, rho0, rhoN, rho_dot0, rho_dotN)

    # Lift the interior of the trajectory toward rho >= target_radius while
    # leaving the exact endpoint values and endpoint derivatives untouched.
    psi = tau**2 * (1 - tau) ** 2
    beta = torch.as_tensor(0.02, device=t.device, dtype=t.dtype)
    floor_gap = F.softplus((radius_tensor - rho_base) / beta) * beta
    interior_scale = 16.0 * psi
    rho = rho_base + interior_scale * floor_gap + psi * F.softplus(x[:, 0:1])

    alpha0 = x0_tensor[1].reshape(())
    alphaN = xN_tensor[1].reshape(())
    alpha_dot0 = (t_total_tensor * v0_tensor[1] / rho0).reshape(())
    alpha_dotN = (t_total_tensor * vN_tensor[1] / rhoN).reshape(())
    alpha_base = _cubic_hermite_scalar(tau, alpha0, alphaN, alpha_dot0, alpha_dotN)
    alpha = alpha_base + psi * x[:, 1:2]

    return torch.cat((rho, alpha), dim=1)


def kinematic_rendezvous_hold_point_eci_polar_alpha_guard_fn(
    t,
    x,
    x0,
    v0,
    target_radius,
    target_speed,
    hold_point_radial_offset,
    model,
):
    """
    Polar kinematic transform for the rendezvous problem with a hard angular
    ordering relative to the target:

      alpha_chaser(t) <= alpha_target(t)  for all t in [0, 1]

    The initial state is slightly behind the target, while the terminal state
    meets the desired hold point exactly. The radius remains unconstrained
    beyond the exact boundary conditions.
    """
    del target_speed
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    v0_tensor = torch.as_tensor(v0, device=t.device, dtype=t.dtype)

    radius_tensor = torch.as_tensor(target_radius, device=t.device, dtype=t.dtype).reshape(())
    offset_tensor = torch.as_tensor(hold_point_radial_offset, device=t.device, dtype=t.dtype).reshape(())
    t_total_tensor = model.t_total.to(device=t.device, dtype=t.dtype).reshape(())

    mean_motion = torch.sqrt(torch.as_tensor(398600.0, device=t.device, dtype=t.dtype) / radius_tensor**3)
    tau = t.reshape(-1, 1)
    theta_target = mean_motion * t_total_tensor * tau

    rho0 = x0_tensor[0].reshape(())
    alpha0 = x0_tensor[1].reshape(())
    rhoN = (radius_tensor + offset_tensor).reshape(())
    alphaN = (mean_motion * t_total_tensor).reshape(())
    vr0 = v0_tensor[0].reshape(())
    vt0 = v0_tensor[1].reshape(())
    vrN = torch.zeros_like(rhoN)
    vtN = (radius_tensor + offset_tensor) * mean_motion

    rho_dot0 = (t_total_tensor * vr0).reshape(())
    rho_dotN = (t_total_tensor * vrN).reshape(())
    rho_base = _cubic_hermite_scalar(tau, rho0, rhoN, rho_dot0, rho_dotN)
    psi = tau**2 * (1 - tau) ** 2
    rho = rho_base + psi * x[:, 0:1]

    delta0 = (alpha0 - torch.zeros_like(alpha0)).reshape(())
    deltaN = torch.zeros_like(delta0)
    alpha_dot0 = (t_total_tensor * vt0 / rho0).reshape(())
    alpha_dotN = (t_total_tensor * vtN / rhoN).reshape(())
    theta_dot = (mean_motion * t_total_tensor).reshape(())
    delta_dot0 = alpha_dot0 - theta_dot
    delta_dotN = alpha_dotN - theta_dot

    delta_base = _cubic_hermite_scalar(tau, delta0, deltaN, delta_dot0, delta_dotN)
    beta = torch.as_tensor(0.02, device=t.device, dtype=t.dtype)
    positive_gap = F.softplus(delta_base / beta) * beta
    interior_scale = 16.0 * psi
    delta = delta_base - interior_scale * positive_gap - psi * F.softplus(x[:, 1:2])
    alpha = theta_target + delta

    return torch.cat((rho, alpha), dim=1)


def geometric_polar_fn(t, x, x0, xN, model, transform_only_R=False):
    if transform_only_R:
        alpha = x[:, 1].detach()
        xN[1] = alpha[-1]

    psi = t * (1 - t)
    return R(t, x0, xN) + psi * x  # phi = 0


def _polar_state_velocity_from_physical_velocity(x_boundary, v_boundary_physical, model, *, t, dtype):
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


def kinematic_polar_polar_transform_only_vr(
    t,
    x,
    x0,
    xN,
    v0,
    vN,
    model,
):
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    xN_tensor = torch.as_tensor(xN, device=t.device, dtype=t.dtype)
    v0_tensor = _polar_state_velocity_from_physical_velocity(x0_tensor, v0, model, t=t, dtype=t.dtype)
    vN_tensor = _polar_state_velocity_from_physical_velocity(xN_tensor, vN, model, t=t, dtype=t.dtype)

    psi = t**2 * (1 - t) ** 2
    phi = 2 * t**3 - 3 * t**2 + t
    r = (R(t, x0_tensor, xN_tensor) + psi * x + phi * (V(t, v0_tensor, vN_tensor) - R_dot(t, x0_tensor, xN_tensor)))[:, 0]
    return torch.stack([r, x[:, 1]], dim=1)


def _cubic_hermite_scalar(t, y0, y1, dy0, dy1):
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * y0 + h10 * dy0 + h01 * y1 + h11 * dy1


def kinematic_polar_positive_radius_landing_fixed_angle_fn(t, x, x0, xN, vt_0, vt_N, model):
    """
    Polar kinematic transform for a descent / landing problem with a fixed
    terminal angle. The radius is constructed to stay at or above the terminal
    landing radius for the whole trajectory, preventing unphysical negative or
    sub-surface radii.

    Boundary conditions:
      - rho(0) = x0[0]
      - alpha(0) = x0[1]
      - rho(1) = xN[0]
      - alpha(1) = xN[1]
      - v_r(0) = 0, v_r(1) = 0
      - v_t(0) = vt_0, v_t(1) = vt_N
    """
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    xN_tensor = torch.as_tensor(xN, device=t.device, dtype=t.dtype)
    vt_0_tensor = torch.as_tensor(vt_0, device=t.device, dtype=t.dtype).reshape(())
    vt_N_tensor = torch.as_tensor(vt_N, device=t.device, dtype=t.dtype).reshape(())
    t_total_tensor = model.t_total.to(device=t.device, dtype=t.dtype).reshape(())

    tau = t.reshape(-1, 1)
    rho0 = x0_tensor[0].reshape(())
    alpha0 = x0_tensor[1].reshape(())
    rho_N_tensor = xN_tensor[0].reshape(())
    alpha_N_tensor = xN_tensor[1].reshape(())

    rho_base = _cubic_hermite_scalar(
        tau,
        rho0,
        rho_N_tensor,
        torch.zeros_like(rho0),
        torch.zeros_like(rho_N_tensor),
    )

    psi = tau**2 * (1 - tau) ** 2
    rho = rho_base + psi * F.softplus(x[:, 0:1])

    alpha_dot0 = (t_total_tensor * vt_0_tensor / rho0).reshape(())
    alpha_dotN = (t_total_tensor * vt_N_tensor / rho_N_tensor).reshape(())
    alpha_base = _cubic_hermite_scalar(tau, alpha0, alpha_N_tensor, alpha_dot0, alpha_dotN)
    alpha = alpha_base + psi * x[:, 1:2]

    return torch.cat((rho, alpha), dim=1)



def kinematic_polar_positive_radius_landing_fn(t, x, x0, rho_N, vt_0, vt_N, model):
    """
    Polar kinematic transform for a descent / landing problem with a trainable
    final angle. The radius is constructed to stay at or above ``rho_N`` for the
    whole trajectory, preventing the model from exploiting negative radii.

    Boundary conditions:
      - rho(0) = x0[0]
      - alpha(0) = x0[1]
      - rho(1) = rho_N
      - alpha(1) = alpha_N (trainable)
      - v_r(0) = 0, v_r(1) = 0
      - v_t(0) = vt_0, v_t(1) = vt_N
    """
    x0_tensor = torch.as_tensor(x0, device=t.device, dtype=t.dtype)
    rho_N_tensor = torch.as_tensor(rho_N, device=t.device, dtype=t.dtype).reshape(())
    vt_0_tensor = torch.as_tensor(vt_0, device=t.device, dtype=t.dtype).reshape(())
    vt_N_tensor = torch.as_tensor(vt_N, device=t.device, dtype=t.dtype).reshape(())
    alpha_N_tensor = model.alpha_N.to(device=t.device, dtype=t.dtype).reshape(())
    t_total_tensor = model.t_total.to(device=t.device, dtype=t.dtype).reshape(())

    tau = t.reshape(-1, 1)
    rho0 = x0_tensor[0].reshape(())
    alpha0 = x0_tensor[1].reshape(())

    # Enforce a monotone, positive radius baseline between start and touchdown
    # with zero radial velocity at both ends.
    rho_base = _cubic_hermite_scalar(
        tau,
        rho0,
        rho_N_tensor,
        torch.zeros_like(rho0),
        torch.zeros_like(rho_N_tensor),
    )

    # Add only a non-negative interior bump so the radius can never fall below
    # the touchdown radius while still allowing shape flexibility.
    psi = tau**2 * (1 - tau) ** 2
    rho = rho_base + psi * F.softplus(x[:, 0:1])

    alpha_dot0 = (t_total_tensor * vt_0_tensor / rho0).reshape(())
    alpha_dotN = (t_total_tensor * vt_N_tensor / rho_N_tensor).reshape(())
    alpha_base = _cubic_hermite_scalar(tau, alpha0, alpha_N_tensor, alpha_dot0, alpha_dotN)
    alpha = alpha_base + psi * x[:, 1:2]

    return torch.cat((rho, alpha), dim=1)


def scaled_tanh(m, m0, mN):
    return (m0 + mN) / 2 + (m0 - mN) / 2 * torch.tanh(m / m.max())


def m_sigmoid(t, m, m0, a=10, t0=0):
    return m0 - (1 - torch.exp(-a * (t - t0))) * m0 * torch.sigmoid(m)


def mass_pinn_fn(t, x, x0, xN, model, transform_only_r=False, transform_only_m0=False, **kwargs):
    # model = kwargs["model"]
    # mN = model.mN
    # mN = kwargs["mN"]
    # mN = torch.nn.functional.softplus(mN.detach())
    # xN_clone = xN.clone()
    # xN_clone[-1] = mN

    if transform_only_r:
        alpha = x[:, 1].detach()
        xN[1] = alpha[-1]
    elif transform_only_m0:
        m = x[:, -1].detach()
        xN[-1] = m[-1]

    # phi = 0

    # m = x0[-1] - (1 - torch.exp(-10 * t)) * x0[-1] * torch.sigmoid(x[:, -2:-1])
    # return torch.cat([r[:, 0:2], x[:, 1:2]], dim=1)

    psi = t * (1 - t)
    r = R(t, x0, xN) + psi * x
    m = r[:, 1:2]
    m = R(t, x0[-1], model.mN) + psi * m
    return torch.cat([r[:, 0:2], m], dim=1)


def mass_pinn_kinematic_fn(t, x, x0, xN, v0, vN, model, transform_only_r=False, transform_only_m0=False):

    psi = t**2 * (1 - t) ** 2
    phi = 2 * t**3 - 3 * t**2 + t
    r = R(t, x0[:-1], xN[:-1]) + psi * x[:, :-1] + phi * (V(t, v0, vN) - R_dot(t, x0[:-1], xN[:-1]))

    psi_m = t * (1 - t)
    m = R(t, x0[-1], xN[-1]) + psi_m * x[:, -2:-1]  # Geometric transform for m
    return torch.cat([r, m], dim=1)

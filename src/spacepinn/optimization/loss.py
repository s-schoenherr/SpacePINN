from __future__ import annotations

from dataclasses import dataclass

import torch

from .dynamics import DynamicsState


@dataclass
class LossBreakdown:
    total: torch.Tensor
    physics: torch.Tensor
    boundary: torch.Tensor | None
    thrust_cap: torch.Tensor | None
    tangential_thrust_smoothness: torch.Tensor | None


def compute_physics_loss(
    *,
    state: DynamicsState,
    coordinate_system: str,
    physics_loss_weight: float,
) -> torch.Tensor:
    if coordinate_system == "cartesian":
        return torch.mean(state.F.norm(dim=1) ** 2) * physics_loss_weight
    if coordinate_system == "polar":
        return (torch.mean(state.F_rho**2) + torch.mean(state.F_alpha**2)) * physics_loss_weight
    raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")


def compute_boundary_loss(
    *,
    state: DynamicsState,
    coordinate_system: str,
    r0: torch.Tensor,
    rN: torch.Tensor,
    boundary_loss_weight: float,
) -> torch.Tensor:
    if coordinate_system == "cartesian":
        return (
            (torch.sum((state.r[0] - r0) ** 2) + torch.sum((state.r[-1] - rN) ** 2))
            / 2
            * boundary_loss_weight
        )

    if coordinate_system == "polar":
        r0_cart = _polar_to_cart(r0)
        rN_cart = _polar_to_cart(rN)
        return (
            (torch.sum((state.r_cart[0] - r0_cart) ** 2) + torch.sum((state.r_cart[-1] - rN_cart) ** 2))
            / 2
            * boundary_loss_weight
        )

    raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")


def compute_thrust_cap_loss(
    *,
    state: DynamicsState,
    coordinate_system: str,
    thrust_cap: float,
    thrust_cap_weight: float,
) -> torch.Tensor:
    if coordinate_system == "cartesian":
        thrust_mag = state.F.norm(dim=1)
    elif coordinate_system == "polar":
        thrust_mag = torch.sqrt(state.F_rho.square().squeeze(-1) + state.F_alpha.square().squeeze(-1))
    else:
        raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")

    cap_excess = torch.relu(thrust_mag / thrust_cap - 1.0)
    return torch.mean(cap_excess.square()) * thrust_cap_weight


def compute_tangential_thrust_smoothness_loss(
    *,
    state: DynamicsState,
    coordinate_system: str,
    t_colloc: torch.Tensor,
    tangential_thrust_smoothness_weight: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if tangential_thrust_smoothness_weight <= 0:
        return torch.zeros((), dtype=t_colloc.dtype, device=t_colloc.device)

    if coordinate_system != "cartesian":
        raise ValueError(
            "Tangential thrust smoothness is currently only supported for cartesian coordinate systems."
        )
    if state.F is None or state.v is None:
        raise ValueError("Tangential thrust smoothness requires cartesian thrust and velocity state.")
    if t_colloc.shape[0] < 4:
        return torch.zeros((), dtype=state.F.dtype, device=state.F.device)

    speed = state.v.norm(dim=1, keepdim=True).clamp_min(eps)
    tangential_thrust = torch.sum(state.F * (state.v / speed), dim=1)
    dt = torch.diff(t_colloc.squeeze(-1)).clamp_min(eps)
    tangential_gradient = torch.diff(tangential_thrust) / dt
    gradient_midpoints = 0.5 * (dt[1:] + dt[:-1]).clamp_min(eps)
    tangential_curvature = torch.diff(tangential_gradient) / gradient_midpoints
    return torch.mean(tangential_curvature.square()) * tangential_thrust_smoothness_weight


def compute_total_loss(
    *,
    state: DynamicsState,
    coordinate_system: str,
    r0: torch.Tensor,
    rN: torch.Tensor,
    physics_loss_weight: float,
    boundary_loss_weight: float,
    t_colloc: torch.Tensor,
    thrust_cap: float | None,
    thrust_cap_weight: float,
    tangential_thrust_smoothness_weight: float,
) -> LossBreakdown:
    physics_loss = compute_physics_loss(
        state=state,
        coordinate_system=coordinate_system,
        physics_loss_weight=physics_loss_weight,
    )
    total_loss = physics_loss
    thrust_cap_loss = None
    tangential_thrust_smoothness_loss = None

    if thrust_cap is not None and thrust_cap_weight > 0:
        thrust_cap_loss = compute_thrust_cap_loss(
            state=state,
            coordinate_system=coordinate_system,
            thrust_cap=thrust_cap,
            thrust_cap_weight=thrust_cap_weight,
        )
        total_loss = total_loss + thrust_cap_loss

    if tangential_thrust_smoothness_weight > 0:
        tangential_thrust_smoothness_loss = compute_tangential_thrust_smoothness_loss(
            state=state,
            coordinate_system=coordinate_system,
            t_colloc=t_colloc,
            tangential_thrust_smoothness_weight=tangential_thrust_smoothness_weight,
        )
        total_loss = total_loss + tangential_thrust_smoothness_loss

    if boundary_loss_weight <= 0:
        return LossBreakdown(
            total=total_loss,
            physics=physics_loss,
            boundary=None,
            thrust_cap=thrust_cap_loss,
            tangential_thrust_smoothness=tangential_thrust_smoothness_loss,
        )

    boundary_loss = compute_boundary_loss(
        state=state,
        coordinate_system=coordinate_system,
        r0=r0,
        rN=rN,
        boundary_loss_weight=boundary_loss_weight,
    )
    total_loss = total_loss + boundary_loss
    return LossBreakdown(
        total=total_loss,
        physics=physics_loss,
        boundary=boundary_loss,
        thrust_cap=thrust_cap_loss,
        tangential_thrust_smoothness=tangential_thrust_smoothness_loss,
    )


def _polar_to_cart(point: torch.Tensor) -> torch.Tensor:
    radius, angle = point
    return radius * torch.stack([torch.cos(angle), torch.sin(angle)])

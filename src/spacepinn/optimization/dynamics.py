from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class DynamicsState:
    r: torch.Tensor
    r_cart: torch.Tensor
    v: torch.Tensor | None
    a: torch.Tensor | None
    G: torch.Tensor
    F: torch.Tensor | None
    rho: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    rho_t: torch.Tensor | None = None
    alpha_t: torch.Tensor | None = None
    a_rho: torch.Tensor | None = None
    a_alpha: torch.Tensor | None = None
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None
    G_rho: torch.Tensor | None = None
    G_alpha: torch.Tensor | None = None
    F_rho: torch.Tensor | None = None
    F_alpha: torch.Tensor | None = None


class DynamicsStrategy(Protocol):
    def compute(
        self,
        *,
        r: torch.Tensor,
        t: torch.Tensor,
        t_total,
        gravity_sources: torch.Tensor,
        dims: int,
        eps: float,
        external_acceleration_fn=None,
    ) -> DynamicsState: ...


def _compute_cartesian_gravity_acceleration(
    *,
    r_cart: torch.Tensor,
    gravity_sources: torch.Tensor,
    dims: int,
    eps: float,
) -> torch.Tensor:
    gravity = torch.zeros((r_cart.shape[0], dims), dtype=r_cart.dtype, device=r_cart.device)
    for gravity_source in gravity_sources:
        r_diff = r_cart - gravity_source[:-1]
        denominator = (torch.linalg.norm(r_diff, dim=1) + eps) ** 3
        for idx in range(dims):
            gravity[:, idx] -= gravity_source[-1] * r_diff[:, idx] / denominator
    return gravity


class CartesianDynamics:
    def compute(
        self,
        *,
        r: torch.Tensor,
        t: torch.Tensor,
        t_total,
        gravity_sources: torch.Tensor,
        dims: int,
        eps: float,
        external_acceleration_fn=None,
    ) -> DynamicsState:
        velocity = torch.stack(
            [
                torch.autograd.grad(
                    r[:, idx],
                    t,
                    grad_outputs=torch.ones_like(r[:, idx]),
                    create_graph=True,
                )[
                    0
                ].squeeze(-1)
                for idx in range(dims)
            ],
            dim=1,
        )
        acceleration = (
            torch.stack(
                [
                    torch.autograd.grad(
                        velocity[:, idx],
                        t,
                        grad_outputs=torch.ones_like(velocity[:, idx]),
                        create_graph=True,
                    )[0].squeeze(-1)
                    for idx in range(dims)
                ],
                dim=1,
            )
            / t_total**2
        )

        gravity = _compute_cartesian_gravity_acceleration(
            r_cart=r,
            gravity_sources=gravity_sources,
            dims=dims,
            eps=eps,
        )
        extra_acceleration = torch.zeros_like(acceleration)
        if external_acceleration_fn is not None:
            extra_acceleration = external_acceleration_fn(
                r=r,
                r_cart=r,
                v=velocity,
                a=acceleration,
                t=t,
                t_total=t_total,
            )
        thrust = acceleration - gravity - extra_acceleration

        return DynamicsState(
            r=r,
            r_cart=r,
            v=velocity,
            a=acceleration,
            G=gravity,
            F=thrust,
        )


class PolarDynamics:
    def compute(
        self,
        *,
        r: torch.Tensor,
        t: torch.Tensor,
        t_total,
        gravity_sources: torch.Tensor,
        dims: int,
        eps: float,
        external_acceleration_fn=None,
    ) -> DynamicsState:
        rho = r[:, 0:1]
        alpha = r[:, 1:2]

        rho_t = torch.autograd.grad(
            rho,
            t,
            grad_outputs=torch.ones_like(rho),
            create_graph=True,
        )[0]
        alpha_t = torch.autograd.grad(
            alpha,
            t,
            grad_outputs=torch.ones_like(alpha),
            create_graph=True,
        )[0]
        rho_tt = torch.autograd.grad(
            rho_t,
            t,
            grad_outputs=torch.ones_like(rho_t),
            create_graph=True,
        )[0]
        alpha_tt = torch.autograd.grad(
            alpha_t,
            t,
            grad_outputs=torch.ones_like(alpha_t),
            create_graph=True,
        )[0]

        a_rho = (rho_tt - rho * alpha_t**2) / t_total**2
        a_alpha = (2 * rho_t * alpha_t + rho * alpha_tt) / t_total**2

        x = rho * torch.cos(alpha)
        y = rho * torch.sin(alpha)
        r_cart = torch.cat((x, y), dim=1)

        gravity = _compute_cartesian_gravity_acceleration(
            r_cart=r_cart,
            gravity_sources=gravity_sources,
            dims=dims,
            eps=eps,
        )
        cos = torch.cos(alpha)
        sin = torch.sin(alpha)

        gravity_x = gravity[:, 0:1]
        gravity_y = gravity[:, 1:2]
        gravity_rho = gravity_x * cos + gravity_y * sin
        gravity_alpha = -gravity_x * sin + gravity_y * cos

        extra_rho = torch.zeros_like(rho)
        extra_alpha = torch.zeros_like(alpha)
        if external_acceleration_fn is not None:
            v_rho = rho_t / t_total
            v_alpha = rho * alpha_t / t_total
            extra_rho, extra_alpha = external_acceleration_fn(
                rho=rho,
                alpha=alpha,
                v_rho=v_rho,
                v_alpha=v_alpha,
                r_cart=r_cart,
                t=t,
                t_total=t_total,
            )

        thrust_rho = a_rho - gravity_rho - extra_rho
        thrust_alpha = a_alpha - gravity_alpha - extra_alpha

        return DynamicsState(
            r=r,
            r_cart=r_cart,
            v=None,
            a=None,
            G=gravity,
            F=None,
            rho=rho,
            alpha=alpha,
            rho_t=rho_t,
            alpha_t=alpha_t,
            a_rho=a_rho,
            a_alpha=a_alpha,
            cos=cos,
            sin=sin,
            G_rho=gravity_rho,
            G_alpha=gravity_alpha,
            F_rho=thrust_rho,
            F_alpha=thrust_alpha,
        )


DYNAMICS_REGISTRY: dict[str, DynamicsStrategy] = {
    "cartesian": CartesianDynamics(),
    "polar": PolarDynamics(),
}


def get_dynamics_strategy(coordinate_system: str) -> DynamicsStrategy:
    try:
        return DYNAMICS_REGISTRY[coordinate_system]
    except KeyError as error:
        supported = ", ".join(sorted(DYNAMICS_REGISTRY))
        raise ValueError(f"Unsupported coordinate_system '{coordinate_system}'. Supported: {supported}.") from error

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .optimization.dynamics import get_dynamics_strategy
from .optimization.engine import OptimizationRun


@dataclass
class TrajectoryDynamics:
    t: np.ndarray
    t_total: float
    gravity_sources: np.ndarray
    r0: np.ndarray
    rN: np.ndarray
    coordinate_system: str
    r: np.ndarray
    v: np.ndarray
    a: np.ndarray
    G: np.ndarray
    F: np.ndarray
    a_mag: np.ndarray
    G_mag: np.ndarray
    F_mag: np.ndarray
    r_polar: Optional[np.ndarray] = None
    v_polar: Optional[np.ndarray] = None
    F_rho: Optional[np.ndarray] = None
    F_alpha: Optional[np.ndarray] = None
    m: Optional[np.ndarray] = None
    m_monotonic: Optional[bool] = None
    residual: Optional[np.ndarray] = None


@dataclass
class OptimizationHistory:
    loss: list[float]
    loss_physics: list[float]
    loss_bc: list[float]


@dataclass
class RunMetadata:
    label: str
    coordinate_system: str
    delta_v: float
    runtime_seconds: float | None = None
    solver: dict[str, object] | None = None


class TrajectoryResult:
    """Class to extract and store the results of the trajectory optimization."""

    def __init__(
        self,
        label: str,
        run_or_optimizer,
        *,
        model=None,
        output_t: Optional[torch.Tensor] = None,
        output_points: Optional[int] = None,
        runtime_seconds: float | None = None,
        solver: dict[str, object] | None = None,
    ):
        run = _resolve_optimization_run(run_or_optimizer)
        t_eval, state = _resolve_output_state(run, model=model, output_t=output_t, output_points=output_points)

        coordinate_system = run.coordinate_system
        boundary_start = _to_numpy(run.r0).flatten()
        boundary_end = _to_numpy(run.rN).flatten()

        t = _to_numpy(t_eval)
        t_total = _to_scalar(run.t_total)
        gravity_sources = _to_numpy(run.gravity_sources)

        if coordinate_system == "cartesian":
            r = _to_numpy(state.r)
            v = _to_numpy(state.v) / t_total
            a = _to_numpy(state.a)
            G = _to_numpy(state.G)
            F = _to_numpy(state.F)
            r_polar = None
            v_polar = None
            F_rho = None
            F_alpha = None

        elif coordinate_system == "polar":
            r = _to_numpy(state.r_cart)
            r_polar = _to_numpy(state.r[:, : run.dims])

            vx = state.rho_t * state.cos + state.rho * state.alpha_t * (-state.sin)
            vy = state.rho_t * state.sin + state.rho * state.alpha_t * state.cos
            v = torch.cat([vx, vy], dim=1).detach().cpu().numpy() / t_total
            v_polar = torch.cat([state.rho_t, state.rho * state.alpha_t], dim=1).detach().cpu().numpy() / t_total

            ax = state.a_rho * state.cos + state.a_alpha * (-state.sin)
            ay = state.a_rho * state.sin + state.a_alpha * state.cos
            a = torch.cat([ax, ay], dim=1).detach().cpu().numpy()

            Gx = state.G_rho * state.cos + state.G_alpha * (-state.sin)
            Gy = state.G_rho * state.sin + state.G_alpha * state.cos
            G = torch.cat([Gx, Gy], dim=1).detach().cpu().numpy()

            Fx = state.F_rho * state.cos + state.F_alpha * (-state.sin)
            Fy = state.F_rho * state.sin + state.F_alpha * state.cos
            F = torch.cat([Fx, Fy], dim=1).detach().cpu().numpy()
            F_rho = _to_numpy(state.F_rho)
            F_alpha = _to_numpy(state.F_alpha)

            rho_0, alpha_0 = run.r0[0].item(), run.r0[1].item()
            rho_N, alpha_N = run.rN[0].item(), run.rN[1].item()
            boundary_start = np.array([rho_0 * np.cos(alpha_0), rho_0 * np.sin(alpha_0)])
            boundary_end = np.array([rho_N * np.cos(alpha_N), rho_N * np.sin(alpha_N)])
        else:
            raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")

        a_mag = np.linalg.norm(a, axis=1)
        G_mag = np.linalg.norm(G, axis=1)
        F_mag = np.linalg.norm(F, axis=1)

        residual = None
        if run.residual is not None:
            residual = np.array([value for value in run.residual.detach().cpu().numpy()])

        mass = None
        mass_monotonic = None
        if run.massPINN:
            mass = _to_numpy(state.r[:, -1])
            mass_monotonic = bool(np.all(np.diff(mass) < 0))

        self.dynamics = TrajectoryDynamics(
            t=t,
            t_total=t_total,
            gravity_sources=gravity_sources,
            r0=boundary_start,
            rN=boundary_end,
            coordinate_system=coordinate_system,
            r=r,
            v=v,
            a=a,
            G=G,
            F=F,
            a_mag=a_mag,
            G_mag=G_mag,
            F_mag=F_mag,
            r_polar=r_polar,
            v_polar=v_polar,
            F_rho=F_rho,
            F_alpha=F_alpha,
            m=mass,
            m_monotonic=mass_monotonic,
            residual=residual,
        )
        self.history = OptimizationHistory(
            loss=[value for value in run.history.loss],
            loss_physics=[value for value in run.history.loss_physics],
            loss_bc=[value for value in run.history.loss_bc],
        )
        self.metadata = RunMetadata(
            label=label,
            coordinate_system=coordinate_system,
            delta_v=float(np.trapezoid(F_mag, t.squeeze()) * t_total),
            runtime_seconds=runtime_seconds,
            solver=solver,
        )
        self._sync_legacy_attributes()

    def _sync_legacy_attributes(self):
        # Preserve the previous flat API used by experiment/plot scripts.
        self.label = self.metadata.label
        self.coordinate_system = self.metadata.coordinate_system
        self.delta_v = self.metadata.delta_v
        self.runtime_seconds = getattr(self.metadata, "runtime_seconds", None)
        self.solver = getattr(self.metadata, "solver", None)
        self.solver_metadata = getattr(self.metadata, "solver", None)

        self.t = self.dynamics.t
        self.t_total = self.dynamics.t_total
        self.ao = self.dynamics.gravity_sources
        self.gravity_sources = self.dynamics.gravity_sources
        self.r0 = self.dynamics.r0
        self.rN = self.dynamics.rN
        self.r = self.dynamics.r
        self.v = self.dynamics.v
        self.a = self.dynamics.a
        self.G = self.dynamics.G
        self.F = self.dynamics.F
        self.a_mag = self.dynamics.a_mag
        self.G_mag = self.dynamics.G_mag
        self.F_mag = self.dynamics.F_mag

        self.loss = self.history.loss
        self.loss_physics = self.history.loss_physics
        self.loss_bc = self.history.loss_bc

        if self.dynamics.r_polar is not None:
            self.r_polar = self.dynamics.r_polar
        if self.dynamics.v_polar is not None:
            self.v_polar = self.dynamics.v_polar
        if self.dynamics.F_rho is not None:
            self.F_rho = self.dynamics.F_rho
        if self.dynamics.F_alpha is not None:
            self.F_alpha = self.dynamics.F_alpha
        if self.dynamics.residual is not None:
            self.residual = self.dynamics.residual
        if self.dynamics.m is not None:
            self.m = self.dynamics.m
        if self.dynamics.m_monotonic is not None:
            self.m_monotonic = self.dynamics.m_monotonic

    @classmethod
    def from_open_goddard(
        cls,
        label: str,
        t,
        r,
        v,
        F,
        G,
        r0,
        rN,
        ao,
        coordinate_system,
        total_cost: float | None = None,
        runtime_seconds: float | None = None,
        solver: dict[str, object] | None = None,
    ):
        """Build a TrajectoryResult from the OpenGoddard output."""
        self = cls.__new__(cls)  # bypass __init__
        t_total = t[-1]
        t_norm = np.asarray(t / t_total)
        r_array = np.asarray(r)
        v_array = np.asarray(v)
        F_array = np.asarray(F)
        G_array = np.asarray(G)
        r0_array = np.asarray(r0)
        rN_array = np.asarray(rN)

        r_polar = None
        v_polar = None
        F_rho = None
        F_alpha = None

        if coordinate_system == "polar":
            rho = r_array[:, 0:1]
            alpha = r_array[:, 1:2]
            vr = v_array[:, 0:1]
            vt = v_array[:, 1:2]
            G_rho = G_array[:, 0:1]
            G_alpha = G_array[:, 1:2]
            thrust_rho = F_array[:, 0:1]
            thrust_alpha = F_array[:, 1:2]

            cos = np.cos(alpha)
            sin = np.sin(alpha)

            r_array = np.concatenate([rho * cos, rho * sin], axis=1)
            v_array = np.concatenate([vr * cos - vt * sin, vr * sin + vt * cos], axis=1)
            G_array = np.concatenate([G_rho * cos - G_alpha * sin, G_rho * sin + G_alpha * cos], axis=1)
            F_array = np.concatenate(
                [thrust_rho * cos - thrust_alpha * sin, thrust_rho * sin + thrust_alpha * cos],
                axis=1,
            )
            r0_array = _polar_point_to_cart(r0_array)
            rN_array = _polar_point_to_cart(rN_array)

            r_polar = np.asarray(r)
            v_polar = np.asarray(v)
            F_rho = np.asarray(thrust_rho)
            F_alpha = np.asarray(thrust_alpha)

        acceleration = F_array + G_array

        self.dynamics = TrajectoryDynamics(
            t=t_norm,
            t_total=float(t_total),
            gravity_sources=np.asarray(ao),
            r0=r0_array,
            rN=rN_array,
            coordinate_system=coordinate_system,
            r=r_array,
            v=v_array,
            a=acceleration,
            G=G_array,
            F=F_array,
            a_mag=np.linalg.norm(acceleration, axis=1),
            G_mag=np.linalg.norm(G_array, axis=1),
            F_mag=np.linalg.norm(F_array, axis=1),
            r_polar=r_polar,
            v_polar=v_polar,
            F_rho=F_rho,
            F_alpha=F_alpha,
        )
        history_loss = [float(total_cost)] if total_cost is not None else []
        self.history = OptimizationHistory(loss=history_loss, loss_physics=[], loss_bc=[])
        self.metadata = RunMetadata(
            label=label,
            coordinate_system=coordinate_system,
            delta_v=float(np.trapezoid(self.dynamics.F_mag, t_norm.squeeze()) * t_total),
            runtime_seconds=runtime_seconds,
            solver=solver,
        )
        self._sync_legacy_attributes()
        return self


def _resolve_optimization_run(run_or_optimizer) -> OptimizationRun:
    if isinstance(run_or_optimizer, OptimizationRun):
        return run_or_optimizer
    run = getattr(run_or_optimizer, "last_run", None)
    if run is None:
        raise ValueError("TrajectoryOptimizer has no run. Call fit() before creating TrajectoryResult.")
    return run


def _resolve_output_state(
    run: OptimizationRun,
    *,
    model=None,
    output_t: Optional[torch.Tensor] = None,
    output_points: Optional[int] = None,
):
    if model is None and output_t is None and output_points is None:
        return run.t, run.state

    if model is None:
        raise ValueError("Dense output evaluation requires `model` to be provided.")

    if output_t is not None:
        t_eval = output_t
    else:
        num_points = 1000 if output_points is None else int(output_points)
        if num_points < 2:
            raise ValueError(f"output_points must be >= 2, got {num_points}.")
        t_eval = torch.linspace(
            0.0,
            1.0,
            num_points,
            dtype=run.t.dtype,
            device=run.t.device,
        ).view(-1, 1)
        t_eval.requires_grad_(True)

    dynamics = get_dynamics_strategy(run.coordinate_system)
    predicted_trajectory = model(t_eval)
    state = dynamics.compute(
        r=predicted_trajectory,
        t=t_eval,
        t_total=run.t_total,
        gravity_sources=run.gravity_sources,
        dims=run.dims,
        eps=1e-8,
    )
    return t_eval, state


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _polar_point_to_cart(point) -> np.ndarray:
    point_array = np.asarray(point, dtype=np.float64).reshape(-1)
    radius, angle = point_array[:2]
    return np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float64)

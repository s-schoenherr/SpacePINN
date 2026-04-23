from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from .config import OptimizerConfig
from .dynamics import DynamicsState, DynamicsStrategy
from .loss import LossBreakdown, compute_total_loss


@dataclass
class OptimizationHistory:
    loss: list[float]
    loss_physics: list[float]
    loss_bc: list[float]


@dataclass
class OptimizationRun:
    t: torch.Tensor
    t_total: torch.Tensor | float
    r0: torch.Tensor
    rN: torch.Tensor
    gravity_sources: torch.Tensor
    coordinate_system: str
    dims: int
    state: DynamicsState
    history: OptimizationHistory
    loss: torch.Tensor
    physics_loss: torch.Tensor
    boundary_loss: torch.Tensor | None
    massPINN: bool = False
    residual: torch.Tensor | None = None


class OptimizationEngine:
    def __init__(
        self,
        *,
        model,
        config: OptimizerConfig,
        dynamics: DynamicsStrategy,
    ):
        self.model = model
        self.config = config
        self.dynamics = dynamics

        self.eps = 1e-8
        self.t = config.t_colloc
        self.t_total = config.t_total
        self.r0 = config.r0
        self.rN = config.rN
        self.dims = self.r0.shape[-1]

        self.gravity_sources = torch.as_tensor(
            config.ao_rgm,
            dtype=self.r0.dtype,
            device=self.r0.device,
        )

        self._track_component_history = config.boundary_loss_weight > 0
        self.history = OptimizationHistory(loss=[], loss_physics=[], loss_bc=[])
        self.state: DynamicsState | None = None
        self.losses: LossBreakdown | None = None

    def _current_t_total(self):
        model_t_total = getattr(self.model, "t_total", None)
        if model_t_total is not None:
            return model_t_total
        return self.t_total

    def fit(self) -> OptimizationRun:
        adam_optimizer = _instantiate_optimizer(self.config.opt_adam, self.model.parameters())
        lbfgs_optimizer = _instantiate_optimizer(self.config.opt_lbfgs, self.model.parameters())

        prev_loss = float("inf")
        total_iterations = self.config.n_adam + self.config.n_lbfgs

        with tqdm(range(total_iterations), disable=not self.config.show_progress, leave=False) as process_bar:
            for iteration in process_bar:
                if iteration < self.config.n_adam:
                    current_loss = self._adam_step(adam_optimizer)
                else:
                    current_loss = self._lbfgs_step(lbfgs_optimizer)

                loss_change = abs((current_loss - prev_loss) / (prev_loss + self.eps))
                prev_loss = current_loss

                if self.config.show_progress:
                    process_bar.set_postfix(
                        {
                            "current loss": f"{current_loss:.4e}",
                            "rel loss change": f"{loss_change:.4e}",
                        }
                    )
                elif self.config.progress_print_interval > 0 and (
                    iteration == 0 or (iteration + 1) % self.config.progress_print_interval == 0
                ):
                    phase = "adam" if iteration < self.config.n_adam else "lbfgs"
                    print(
                        f"[progress] iter={iteration + 1}/{total_iterations} "
                        f"phase={phase} loss={current_loss:.6e} rel_change={loss_change:.6e}"
                    )

                if self._should_stop_for_convergence(iteration=iteration, loss_change=loss_change):
                    print(f"Convergence reached at iteration {iteration} with loss change {loss_change:.6f}.")
                    _print_training_separator("converged", iteration, current_loss, loss_change)
                    break
                if np.isnan(current_loss):
                    print(f"Training ended at iteration {iteration} because loss is nan.")
                    _print_training_separator("nan", iteration, current_loss)
                    break

        if self.state is None or self.losses is None:
            raise RuntimeError("Training did not execute any optimization step.")

        self._refresh_final_state()

        return OptimizationRun(
            t=self.t,
            t_total=self._current_t_total(),
            r0=self.r0,
            rN=self.rN,
            gravity_sources=self.gravity_sources,
            coordinate_system=self.config.coordinate_system,
            dims=self.dims,
            state=self.state,
            history=self.history,
            loss=self.losses.total,
            physics_loss=self.losses.physics,
            boundary_loss=self.losses.boundary,
            massPINN=self.config.massPINN,
            residual=getattr(self.model, "residual", None),
        )

    def _should_stop_for_convergence(self, *, iteration: int, loss_change: float) -> bool:
        if not np.isfinite(loss_change) or loss_change >= self.config.convergence_threshold:
            return False

        completed_iterations = iteration + 1
        if self.config.n_lbfgs > 0:
            return completed_iterations > self.config.n_adam

        minimum_adam_iterations = min(self.config.n_adam, max(250, self.config.progress_print_interval))
        return completed_iterations >= max(1, minimum_adam_iterations)

    def _refresh_final_state(self) -> None:
        with torch.enable_grad():
            predicted_trajectory = self.model(self.t)
            self.state = self.dynamics.compute(
                r=predicted_trajectory,
                t=self.t,
                t_total=self._current_t_total(),
                gravity_sources=self.gravity_sources,
                dims=self.dims,
                eps=self.eps,
                external_acceleration_fn=self.config.external_acceleration_fn,
            )
            self.losses = compute_total_loss(
                state=self.state,
                coordinate_system=self.config.coordinate_system,
                r0=self.r0,
                rN=self.rN,
                physics_loss_weight=self.config.physics_loss_weight,
                boundary_loss_weight=self.config.boundary_loss_weight,
                t_colloc=self.t,
                thrust_cap=self.config.thrust_cap,
                thrust_cap_weight=self.config.thrust_cap_weight,
                tangential_thrust_smoothness_weight=self.config.tangential_thrust_smoothness_weight,
            )

        if self.history.loss:
            self.history.loss[-1] = self.losses.total.item()
        if self._track_component_history:
            if self.history.loss_physics:
                self.history.loss_physics[-1] = self.losses.physics.item()
            if self.history.loss_bc and self.losses.boundary is not None:
                self.history.loss_bc[-1] = self.losses.boundary.item()

    def _adam_step(self, optimizer) -> float:
        self._optimization_step(optimizer)
        optimizer.step()
        return self.history.loss[-1]

    def _lbfgs_step(self, optimizer) -> float:
        optimizer.step(lambda: self._optimization_step(optimizer))
        return self.history.loss[-1]

    def _optimization_step(self, optimizer):
        optimizer.zero_grad()
        predicted_trajectory = self.model(self.t)

        self.state = self.dynamics.compute(
            r=predicted_trajectory,
            t=self.t,
            t_total=self._current_t_total(),
            gravity_sources=self.gravity_sources,
            dims=self.dims,
            eps=self.eps,
            external_acceleration_fn=self.config.external_acceleration_fn,
        )
        self.losses = compute_total_loss(
            state=self.state,
            coordinate_system=self.config.coordinate_system,
            r0=self.r0,
            rN=self.rN,
            physics_loss_weight=self.config.physics_loss_weight,
            boundary_loss_weight=self.config.boundary_loss_weight,
            t_colloc=self.t,
            thrust_cap=self.config.thrust_cap,
            thrust_cap_weight=self.config.thrust_cap_weight,
            tangential_thrust_smoothness_weight=self.config.tangential_thrust_smoothness_weight,
        )

        if self._track_component_history:
            self.history.loss_physics.append(self.losses.physics.item())
            self.history.loss_bc.append(self.losses.boundary.item())
        self.history.loss.append(self.losses.total.item())

        total_loss = self.losses.total
        total_loss.backward()
        # LBFGS converts the closure result to float(), so return a detached tensor
        # after backprop to avoid noisy autograd warnings.
        return total_loss.detach()


def _instantiate_optimizer(factory, parameters):
    if isinstance(factory, torch.optim.Optimizer):
        return factory
    if isinstance(factory, functools.partial):
        return factory(parameters, **(factory.keywords or {}))
    if callable(factory):
        return factory(parameters)
    raise TypeError(f"Unsupported optimizer factory type: {type(factory)}")


def _print_training_separator(reason, iteration, current_loss, loss_change=None):
    print()
    print("*" * 92)
    if loss_change is None:
        print(
            f"[SWINGBY] Training finished ({reason}) at iteration {iteration} | "
            f"final_loss={current_loss:.6e}"
        )
    else:
        print(
            f"[SWINGBY] Training finished ({reason}) at iteration {iteration} | "
            f"final_loss={current_loss:.6e} | rel_loss_change={loss_change:.6e}"
        )
    print("*" * 92)
    print()

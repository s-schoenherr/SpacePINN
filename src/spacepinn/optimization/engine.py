from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

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
        self.best_loss = float("inf")
        self.best_model_state: dict[str, torch.Tensor] | None = None

    def _current_t_total(self):
        model_t_total = getattr(self.model, "t_total", None)
        if model_t_total is not None:
            return model_t_total
        return self.t_total

    def fit(self) -> OptimizationRun:
        adam_optimizer = _instantiate_optimizer(self.config.opt_adam, self.model.parameters())
        lbfgs_optimizer = _instantiate_optimizer(self.config.opt_lbfgs, self.model.parameters())

        total_iterations = self.config.n_adam + self.config.n_lbfgs
        completed_iterations = 0
        final_phase = "adam"
        final_loss = float("nan")
        final_loss_change = float("nan")
        final_reason = "completed"

        with tqdm(total=total_iterations, disable=not self.config.show_progress, leave=False) as process_bar:
            _, completed_iterations, final_loss, final_loss_change, final_reason = self._run_phase(
                optimizer=adam_optimizer,
                optimizer_name="adam",
                phase_iterations=self.config.n_adam,
                total_iterations=total_iterations,
                process_bar=process_bar,
                completed_iterations=completed_iterations,
            )
            final_phase = "adam"

            if final_reason == "nan":
                _print_training_separator("nan", completed_iterations - 1, final_loss)
            elif self.config.n_lbfgs > 0:
                _, completed_iterations, final_loss, final_loss_change, final_reason = self._run_phase(
                    optimizer=lbfgs_optimizer,
                    optimizer_name="lbfgs",
                    phase_iterations=self.config.n_lbfgs,
                    total_iterations=total_iterations,
                    process_bar=process_bar,
                    completed_iterations=completed_iterations,
                )
                final_phase = "lbfgs"
                if final_reason == "nan":
                    _print_training_separator("nan", completed_iterations - 1, final_loss)
                elif final_reason == "converged":
                    print(
                        f"Convergence reached in {final_phase} at iteration {completed_iterations - 1} "
                        f"with loss change {final_loss_change:.6f}."
                    )
                    _print_training_separator("converged", completed_iterations - 1, final_loss, final_loss_change)
            elif final_reason == "converged":
                print(
                    f"Convergence reached in {final_phase} at iteration {completed_iterations - 1} "
                    f"with loss change {final_loss_change:.6f}."
                )
                _print_training_separator("converged", completed_iterations - 1, final_loss, final_loss_change)

        if self.state is None or self.losses is None:
            raise RuntimeError("Training did not execute any optimization step.")

        self._restore_best_model_state()
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

    def _capture_best_model_state(self, current_loss: float) -> None:
        if not np.isfinite(current_loss) or current_loss >= self.best_loss:
            return

        self.best_loss = float(current_loss)
        self.best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _restore_best_model_state(self) -> None:
        if self.best_model_state is None:
            return
        restored_state: dict[str, Any] = {
            key: value.to(device=param.device, dtype=param.dtype) if (param := self.model.state_dict()[key]).is_floating_point() else value.to(device=param.device)
            for key, value in self.best_model_state.items()
        }
        self.model.load_state_dict(restored_state, strict=True)

    def _should_stop_for_convergence(self, *, phase_iteration: int, phase_iterations: int, loss_change: float) -> bool:
        if not np.isfinite(loss_change) or loss_change >= self.config.convergence_threshold:
            return False

        minimum_phase_iterations = max(1, min(phase_iterations, max(250, self.config.progress_print_interval)))
        return (phase_iteration + 1) >= minimum_phase_iterations

    def _run_phase(
        self,
        *,
        optimizer,
        optimizer_name: str,
        phase_iterations: int,
        total_iterations: int,
        process_bar,
        completed_iterations: int,
    ) -> tuple[bool, int, float, float, str]:
        if phase_iterations <= 0:
            return True, completed_iterations, float("nan"), float("nan"), "skipped"

        prev_loss = float("inf")
        current_loss = float("nan")
        loss_change = float("nan")

        for phase_iteration in range(phase_iterations):
            if optimizer_name == "adam":
                current_loss = self._adam_step(optimizer)
            else:
                current_loss = self._lbfgs_step(optimizer)

            loss_change = abs((current_loss - prev_loss) / (prev_loss + self.eps))
            prev_loss = current_loss
            completed_iterations += 1

            if self.config.show_progress:
                process_bar.set_postfix(
                    {
                        "current loss": f"{current_loss:.4e}",
                        "rel loss change": f"{loss_change:.4e}",
                    }
                )
            elif self.config.progress_print_interval > 0 and (
                phase_iteration == 0 or completed_iterations % self.config.progress_print_interval == 0
            ):
                print(
                    f"[progress] iter={completed_iterations}/{total_iterations} "
                    f"phase={optimizer_name} loss={current_loss:.6e} rel_change={loss_change:.6e}"
                )

            process_bar.update(1)

            if np.isnan(current_loss):
                print(f"Training ended in {optimizer_name} at iteration {completed_iterations - 1} because loss is nan.")
                return False, completed_iterations, current_loss, loss_change, "nan"

            self._capture_best_model_state(current_loss)

            if self._should_stop_for_convergence(
                phase_iteration=phase_iteration,
                phase_iterations=phase_iterations,
                loss_change=loss_change,
            ):
                return False, completed_iterations, current_loss, loss_change, "converged"

        return True, completed_iterations, current_loss, loss_change, "completed"

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

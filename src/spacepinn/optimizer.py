from __future__ import annotations

from .optimization.config import OptimizerConfig, normalize_optimizer_kwargs
from .optimization.dynamics import get_dynamics_strategy
from .optimization.engine import OptimizationRun, OptimizationEngine


class TrajectoryOptimizer:
    def __init__(self, model, **kwargs):
        self.model = model
        self.config: OptimizerConfig = normalize_optimizer_kwargs(kwargs)
        self.last_run: OptimizationRun | None = None

    def fit(self) -> OptimizationRun:
        dynamics = get_dynamics_strategy(self.config.coordinate_system)
        run = OptimizationEngine(model=self.model, config=self.config, dynamics=dynamics).fit()
        self.last_run = run
        return run

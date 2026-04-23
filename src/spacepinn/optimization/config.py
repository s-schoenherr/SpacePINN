from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any


_ALIASES = {
    "w_physics": "physics_loss_weight",
    "w_bc": "boundary_loss_weight",
}

_REQUIRED_KEYS = {
    "ao_rgm",
    "t_colloc",
    "t_total",
    "r0",
    "rN",
    "opt_adam",
    "opt_lbfgs",
}

_OPTIONAL_DEFAULTS = {
    "coordinate_system": "cartesian",
    "dimensions": None,
    "phi_trainable": None,
    "n_adam": 0,
    "n_lbfgs": 100,
    "physics_loss_weight": 1.0,
    "boundary_loss_weight": 0.0,
    "thrust_cap": None,
    "thrust_cap_weight": 0.0,
    "tangential_thrust_smoothness_weight": 0.0,
    "convergence_threshold": 1e-5,
    "show_progress": None,
    "progress_print_interval": 250,
    "massPINN": False,
    "external_acceleration_fn": None,
    "kwargs": None,
}

_SUPPORTED_COORDINATE_SYSTEMS = {"cartesian", "polar"}


@dataclass(frozen=True)
class OptimizerConfig:
    ao_rgm: Any
    t_colloc: Any
    t_total: Any
    r0: Any
    rN: Any
    opt_adam: Any
    opt_lbfgs: Any
    coordinate_system: str
    dimensions: int | None
    phi_trainable: Any
    n_adam: int
    n_lbfgs: int
    physics_loss_weight: float
    boundary_loss_weight: float
    thrust_cap: float | None
    thrust_cap_weight: float
    tangential_thrust_smoothness_weight: float
    convergence_threshold: float
    show_progress: bool
    progress_print_interval: int
    massPINN: bool
    external_acceleration_fn: Any
    kwargs: dict[str, Any] | None

    @property
    def w_physics(self) -> float:
        return self.physics_loss_weight

    @property
    def w_bc(self) -> float:
        return self.boundary_loss_weight


def _resolve_show_progress(show_progress: Any) -> bool:
    if show_progress is None:
        ci_mode = str(os.getenv("CI", "")).lower() in {"1", "true", "yes"}
        show_progress = (not ci_mode) and sys.stderr.isatty()
    return bool(show_progress)


def normalize_optimizer_kwargs(kwargs: dict[str, Any]) -> OptimizerConfig:
    normalized = dict(kwargs)

    for alias, canonical in _ALIASES.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized[alias]
        normalized.pop(alias, None)

    allowed_keys = _REQUIRED_KEYS.union(_OPTIONAL_DEFAULTS)
    unknown_keys = sorted(set(normalized) - allowed_keys)
    if unknown_keys:
        allowed = ", ".join(sorted(allowed_keys.union(_ALIASES)))
        raise TypeError(f"Unknown optimizer kwargs: {unknown_keys}. Allowed keys: {allowed}")

    missing_keys = sorted(key for key in _REQUIRED_KEYS if key not in normalized)
    if missing_keys:
        raise TypeError(f"Missing required optimizer kwargs: {missing_keys}")

    for key, default_value in _OPTIONAL_DEFAULTS.items():
        normalized.setdefault(key, default_value)

    coordinate_system = str(normalized["coordinate_system"]).strip().lower()
    if coordinate_system not in _SUPPORTED_COORDINATE_SYSTEMS:
        allowed_systems = ", ".join(sorted(_SUPPORTED_COORDINATE_SYSTEMS))
        raise ValueError(f"Unsupported coordinate_system '{coordinate_system}'. Allowed: {allowed_systems}.")
    normalized["coordinate_system"] = coordinate_system

    if normalized["dimensions"] is not None:
        normalized["dimensions"] = int(normalized["dimensions"])
    normalized["n_adam"] = int(normalized["n_adam"])
    normalized["n_lbfgs"] = int(normalized["n_lbfgs"])
    normalized["physics_loss_weight"] = float(normalized["physics_loss_weight"])
    normalized["boundary_loss_weight"] = float(normalized["boundary_loss_weight"])
    normalized["thrust_cap"] = None if normalized["thrust_cap"] is None else float(normalized["thrust_cap"])
    normalized["thrust_cap_weight"] = float(normalized["thrust_cap_weight"])
    normalized["tangential_thrust_smoothness_weight"] = float(normalized["tangential_thrust_smoothness_weight"])
    normalized["convergence_threshold"] = float(normalized["convergence_threshold"])
    normalized["show_progress"] = _resolve_show_progress(normalized["show_progress"])
    normalized["progress_print_interval"] = (
        int(normalized["progress_print_interval"]) if normalized["progress_print_interval"] else 0
    )
    normalized["massPINN"] = bool(normalized["massPINN"])

    return OptimizerConfig(**normalized)

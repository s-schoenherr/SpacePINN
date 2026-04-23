from __future__ import annotations

import time
from typing import Any

from ..optimizer import TrajectoryOptimizer
from ..pinn import PINN
from ..result import TrajectoryResult
from .runtime import _configure_runtime_determinism, _prepared_runtime_dtype


def _print_experiment_start(label: str, *, source: str = "pinn") -> None:
    print()
    print("*" * 92)
    print(f"[SWINGBY] Starting experiment | label={label} | source={source}")
    print("*" * 92)
    print()


def _format_trainable_parameter_summary(param) -> str:
    tensor = param.detach()
    shape = tuple(tensor.shape)
    numel = int(tensor.numel())
    if numel == 1:
        value_repr = f"{float(tensor.item()):.9g}"
    elif numel <= 6:
        flat_values = [f"{float(v):.6g}" for v in tensor.reshape(-1).tolist()]
        value_repr = "[" + ", ".join(flat_values) + "]"
    else:
        value_repr = f"numel={numel}"
    return f"shape={shape} | value={value_repr} | requires_grad={bool(param.requires_grad)}"


def _print_trainable_parameter_registration(name: str, param) -> None:
    print()
    print("*" * 92)
    print(f"[SWINGBY] Trainable parameter registered: {name}")
    print(_format_trainable_parameter_summary(param))
    print("*" * 92)
    print()


def execute_single_experiment(config_runtime: dict[str, Any], model=None):
    _print_experiment_start(config_runtime.get("label", "experiment"), source="pinn")
    _configure_runtime_determinism(config_runtime)
    started = time.process_time()
    runtime_dtype = _prepared_runtime_dtype(config_runtime)
    if model is None:
        model = PINN(**config_runtime["pinn"])  # Initialize the model
    else:
        model = model.to(dtype=runtime_dtype)

    # Register extra parameters if provided.
    if extra_parameters := config_runtime.get("extra_parameters", {}):
        for name, param in extra_parameters.items():
            model.register_parameter(str(name), param)
            if name in config_runtime["optimizer"]:
                config_runtime["optimizer"][name] = param
            _print_trainable_parameter_registration(name, param)

    optimizer = TrajectoryOptimizer(model, **config_runtime["optimizer"])
    run = optimizer.fit()
    result = TrajectoryResult(
        config_runtime["label"],
        run,
        model=model,
        output_points=int(config_runtime.get("result_output_points", 1000)),
        runtime_seconds=time.process_time() - started,
    )
    return model, result

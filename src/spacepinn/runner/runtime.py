from __future__ import annotations

import functools
import os
import random
from typing import Any

import numpy as np
import torch

_FLOAT_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _resolve_runtime_dtype(config_runtime: dict[str, Any]) -> torch.dtype:
    dtype_override = os.getenv("SWINGBY_DTYPE")
    dtype_value = dtype_override if dtype_override is not None else config_runtime.get("numeric_dtype", "float32")

    if isinstance(dtype_value, torch.dtype):
        if dtype_value not in (torch.float32, torch.float64):
            raise ValueError(f"Unsupported floating dtype: {dtype_value}. Use torch.float32 or torch.float64.")
        return dtype_value

    dtype_key = str(dtype_value).strip().lower()
    if dtype_key not in _FLOAT_DTYPE_BY_NAME:
        allowed = ", ".join(sorted(_FLOAT_DTYPE_BY_NAME))
        raise ValueError(f"Unsupported numeric_dtype '{dtype_value}'. Allowed values: {allowed}.")
    return _FLOAT_DTYPE_BY_NAME[dtype_key]


def _prepared_runtime_dtype(config_runtime: dict[str, Any]) -> torch.dtype:
    dtype_key = str(config_runtime.get("numeric_dtype", "float32")).strip().lower()
    dtype = _FLOAT_DTYPE_BY_NAME.get(dtype_key)
    if dtype is None:
        allowed = ", ".join(sorted(_FLOAT_DTYPE_BY_NAME))
        raise ValueError(f"Prepared config has unsupported numeric_dtype '{dtype_key}'. Allowed values: {allowed}.")
    return dtype


def _cast_runtime_value(value: Any, dtype: torch.dtype) -> Any:
    if isinstance(value, torch.nn.Parameter):
        casted_data = _cast_runtime_value(value.detach(), dtype)
        return torch.nn.Parameter(casted_data, requires_grad=value.requires_grad)
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and value.dtype != dtype:
            return value.to(dtype=dtype)
        return value
    if isinstance(value, dict):
        return {key: _cast_runtime_value(val, dtype) for key, val in value.items()}
    if isinstance(value, list):
        return [_cast_runtime_value(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_runtime_value(item, dtype) for item in value)
    if isinstance(value, functools.partial):
        return functools.partial(
            value.func,
            *(_cast_runtime_value(arg, dtype) for arg in value.args),
            **{key: _cast_runtime_value(val, dtype) for key, val in (value.keywords or {}).items()},
        )
    return value


def _prepare_runtime_config(config_runtime: dict[str, Any]) -> dict[str, Any]:
    dtype = _resolve_runtime_dtype(config_runtime)
    torch.set_default_dtype(dtype)
    prepared = _cast_runtime_value(config_runtime, dtype)
    prepared["numeric_dtype"] = _dtype_name(dtype)
    return prepared


def _configure_runtime_determinism(config_runtime: dict[str, Any]) -> None:
    """Apply optional deterministic runtime settings shared by local and CI runs."""
    thread_env = os.getenv("SWINGBY_NUM_THREADS")
    if thread_env:
        try:
            num_threads = int(thread_env)
            if num_threads > 0:
                torch.set_num_threads(num_threads)
                try:
                    torch.set_num_interop_threads(num_threads)
                except RuntimeError:
                    # set_num_interop_threads can only be called once per process.
                    pass
        except ValueError:
            pass

    if "seed" in config_runtime:
        seed = int(config_runtime["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


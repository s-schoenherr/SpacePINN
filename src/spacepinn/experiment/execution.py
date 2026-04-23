from __future__ import annotations

import os
from typing import Any

from ..pinn import PINN
from ..runner import execute_single_experiment
from ..runner.runtime import _prepare_runtime_config
from .specs import ExternalEntrySpec, PinnEntrySpec, PreparedEntry


def prepare_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_runtime_config(config)
    if os.getenv("FAST_SMOKE", "0") == "1":
        optimizer_cfg = prepared.get("optimizer")
        if isinstance(optimizer_cfg, dict):
            optimizer_cfg["n_adam"] = 1
            optimizer_cfg["n_lbfgs"] = 0
    return prepared


def build_prepared_entry(
    *,
    label: str,
    result: Any,
    model: Any | None,
    config: dict[str, Any] | None,
    plotting: dict[str, Any] | None = None,
    source: str = "pinn",
    log_text: str | None = None,
    log_filename: str | None = None,
) -> PreparedEntry:
    return PreparedEntry(
        label=label,
        result=result,
        model=model,
        config=config,
        plotting={} if plotting is None else dict(plotting),
        source=source,
        log_text=log_text,
        log_filename=log_filename,
    )


def prepare_external_entry(spec: ExternalEntrySpec) -> PreparedEntry:
    return build_prepared_entry(
        label=spec.label,
        result=spec.result,
        model=spec.model,
        config=spec.config,
        plotting=spec.plotting,
        source=spec.source,
        log_text=spec.log_text,
        log_filename=spec.log_filename,
    )


def build_pretrained_model(config_runtime: dict[str, Any], source_model, *, model_cls: type[PINN] = PINN):
    model = model_cls(**config_runtime["pinn"])
    model.load_state_dict(
        {
            key: value
            for key, value in source_model.state_dict().items()
            if key in model.state_dict()
        },
        strict=False,
    )
    return model


def run_pinn_entry(spec: PinnEntrySpec) -> PreparedEntry:
    config = spec.config_builder()
    config_runtime = prepare_runtime_config(config)
    if spec.runtime_mutator is not None:
        spec.runtime_mutator(config_runtime)
    model = spec.model_factory(config_runtime) if spec.model_factory is not None else None
    model, result = execute_single_experiment(config_runtime, model=model)
    return build_prepared_entry(
        label=config_runtime["label"],
        result=result,
        model=model,
        config=config_runtime,
        plotting=config_runtime.get("plotting", {}),
        source=spec.source,
    )

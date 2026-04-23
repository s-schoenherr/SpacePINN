from __future__ import annotations

import copy
import pickle
from typing import Any

import torch

from .context import RunCollectionContext, RunContext
from .execution import execute_single_experiment
from .loading import load_run
from .runtime import _prepare_runtime_config
from .summary import format_collection_run_summary, print_collection_run_summary


_PLOTTING_ENTRY_KEYS = {
    "color",
    "linestyle",
    "trajectory_linestyle",
    "quiver_scale",
    "quiver_step",
    "quiver_count",
    "zorder",
}


def run_experiment(config: dict[str, Any], model=None, run_root: str = "runs") -> dict[str, Any]:
    config_runtime = _prepare_runtime_config(copy.deepcopy(config))
    run_context = RunContext(
        config=config_runtime,
        label=config_runtime.get("label", "experiment"),
        run_root=run_root,
    )
    run_context.start()

    try:
        model, result = execute_single_experiment(config_runtime, model=model)

        model_path = run_context.model_dir / "model_state_dict.pt"
        torch.save(model.state_dict(), model_path)
        run_context.register_artifact(model_path, kind="model_state_dict")

        result_pickle_path = run_context.result_dir / "trajectory_result.pkl"
        with result_pickle_path.open("wb") as fh:
            pickle.dump(result, fh)
        run_context.register_artifact(result_pickle_path, kind="trajectory_result_pickle")

        summary = {
            "label": config_runtime["label"],
            "delta_v": result.delta_v,
            "coordinate_system": result.coordinate_system,
            "t_total": result.t_total,
            "runtime_seconds": getattr(result, "runtime_seconds", None),
            "solver": getattr(result, "solver_metadata", None),
            "final_loss": result.loss[-1] if result.loss else None,
            "final_loss_physics": result.loss_physics[-1] if result.loss_physics else None,
            "final_loss_bc": result.loss_bc[-1] if result.loss_bc else None,
            "epochs_total": len(result.loss),
        }
        run_context.finalize_success(summary)

        return {
            "label": config_runtime["label"],
            "result": result,
            **config_runtime.get("plotting", {}),
            "model": model,
            "run_id": run_context.run_id,
            "run_dir": str(run_context.run_dir),
            "plot_output_dir": str(run_context.plot_dir),
            "summary_path": str(run_context.summary_path),
        }
    except Exception as error:
        run_context.finalize_failure(error)
        raise


def run_experiment_collection(
    configs: list[dict[str, Any]],
    label: str,
    additional_entries: list[dict[str, Any]] | None = None,
    run_root: str = "runs",
) -> dict[str, Any]:
    collection_context = RunCollectionContext(label=label, run_root=run_root)
    collection_context.start()
    collection_results = []

    try:
        for config in configs:
            config_runtime = _prepare_runtime_config(copy.deepcopy(config))
            model, result = execute_single_experiment(config_runtime)
            collection_context.add_entry(
                label=config_runtime["label"],
                result=result,
                config=config_runtime,
                model=model,
                source="pinn",
            )
            collection_results.append(
                {
                    "label": config_runtime["label"],
                    "source": "pinn",
                    "result": result,
                    **config_runtime.get("plotting", {}),
                    "model": model,
                    "run_id": collection_context.run_id,
                    "run_dir": str(collection_context.run_dir),
                    "plot_output_dir": str(collection_context.plot_dir),
                    "summary_path": str(collection_context.summary_path),
                }
            )

        for entry in additional_entries or []:
            additional_label = entry["label"]
            additional_result = entry["result"]
            additional_model = entry.get("model")
            additional_config = copy.deepcopy(entry.get("config")) if entry.get("config") is not None else None
            additional_source = entry.get("source", "external")
            additional_plotting = dict(entry.get("plotting", {}))
            for key in _PLOTTING_ENTRY_KEYS:
                if key in entry:
                    additional_plotting[key] = entry[key]
            collection_context.add_entry(
                label=additional_label,
                result=additional_result,
                config=additional_config,
                model=additional_model,
                source=additional_source,
                log_text=entry.get("log_text"),
                log_filename=entry.get("log_filename"),
            )
            collection_results.append(
                {
                    "label": additional_label,
                    "source": additional_source,
                    "result": additional_result,
                    **additional_plotting,
                    "model": additional_model,
                    "run_id": collection_context.run_id,
                    "run_dir": str(collection_context.run_dir),
                    "plot_output_dir": str(collection_context.plot_dir),
                    "summary_path": str(collection_context.summary_path),
                }
            )

        collection_context.finalize_success()

        return {
            "label": label,
            "entries": collection_results,
            "run_id": collection_context.run_id,
            "run_dir": str(collection_context.run_dir),
            "plot_output_dir": str(collection_context.plot_dir),
            "summary_path": str(collection_context.summary_path),
            "manifest_path": str(collection_context.manifest_path),
            "config_path": str(collection_context.config_path),
        }
    except Exception as error:
        collection_context.finalize_failure(error)
        raise


def export_results(results, filename: str) -> None:
    with open(filename, "wb") as fh:
        pickle.dump(results, fh)


def load_results(filename: str):
    with open(filename, "rb") as fh:
        return pickle.load(fh)


__all__ = [
    "run_experiment",
    "run_experiment_collection",
    "load_run",
    "format_collection_run_summary",
    "print_collection_run_summary",
    "export_results",
    "load_results",
    "execute_single_experiment",
]

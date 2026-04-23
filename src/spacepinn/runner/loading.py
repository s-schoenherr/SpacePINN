from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import torch

from ..plotting.style import resolve_plotting_style
from .types import EntryPathsDict, LoadedEntryDict, LoadedRunDict


def _resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path

    run_dir_path = Path(run_dir)
    candidates = [run_dir_path / path]

    parts_lower = [part.lower() for part in path.parts]
    if parts_lower and parts_lower[0] == "runs":
        ancestor = run_dir_path
        while ancestor.name.lower() != "runs" and ancestor.parent != ancestor:
            ancestor = ancestor.parent
        if ancestor.name.lower() == "runs":
            candidates.insert(0, ancestor.parent / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_run_dir(run_dir: str | Path) -> Path:
    run_dir_input = Path(run_dir).expanduser()
    repo_root = Path(__file__).resolve().parents[2]

    candidates = []
    if run_dir_input.is_absolute():
        candidates.append(run_dir_input.resolve())
    else:
        candidates.append((Path.cwd() / run_dir_input).resolve())
        candidates.append((repo_root / run_dir_input).resolve())
        if "runs" not in [part.lower() for part in run_dir_input.parts]:
            candidates.append((repo_root / "runs" / run_dir_input).resolve())

    for candidate in candidates:
        if (candidate / "manifest.json").exists():
            return candidate

    return candidates[0]


def _entry_paths(
    result_path: Path,
    config_path: Path | None,
    model_path: Path | None,
    log_path: Path | None = None,
) -> EntryPathsDict:
    return {
        "result_pickle": str(result_path),
        "config_json": str(config_path) if config_path is not None else None,
        "model_state_dict": str(model_path) if model_path is not None else None,
        "log_file": str(log_path) if log_path is not None else None,
    }


def load_run(run_dir: str | Path, load_model_state_dict: bool = False, map_location: str = "cpu") -> LoadedRunDict:
    run_dir_path = _resolve_run_dir(run_dir)
    manifest_path = run_dir_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    config = None
    summary = None
    aggregate_summary = None
    manifest_paths = manifest.get("paths", {})

    config_json_path = _resolve_run_path(run_dir_path, manifest_paths.get("config_json"))
    if config_json_path is None:
        config_json_path = run_dir_path / "config.json"
    if config_json_path.exists():
        config = json.loads(config_json_path.read_text(encoding="utf-8"))

    summary_path = _resolve_run_path(run_dir_path, manifest_paths.get("summary"))
    if summary_path is None:
        summary_path = run_dir_path / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    aggregate_summary_path = _resolve_run_path(run_dir_path, manifest_paths.get("aggregate_summary"))
    if aggregate_summary_path is None:
        aggregate_summary_path = run_dir_path / "aggregate_summary.json"
    if aggregate_summary_path.exists():
        aggregate_summary = json.loads(aggregate_summary_path.read_text(encoding="utf-8"))

    loaded_entries: list[LoadedEntryDict] = []

    if manifest.get("entries"):
        for entry in manifest["entries"]:
            entry_paths = entry.get("paths", {})
            result_path = _resolve_run_path(run_dir_path, entry_paths.get("result_pickle"))
            if result_path is None:
                raise FileNotFoundError(f"Missing result_pickle path for entry {entry.get('entry_id')}")

            with result_path.open("rb") as fh:
                result = pickle.load(fh)

            entry_config = None
            entry_config_path = _resolve_run_path(run_dir_path, entry_paths.get("config_json"))
            if entry_config_path is not None and entry_config_path.exists():
                entry_config = json.loads(entry_config_path.read_text(encoding="utf-8"))

            model_state_dict: dict[str, Any] | None = None
            model_state_path = _resolve_run_path(run_dir_path, entry_paths.get("model_state_dict"))
            log_path = _resolve_run_path(run_dir_path, entry_paths.get("log_file"))
            if load_model_state_dict and model_state_path is not None and model_state_path.exists():
                model_state_dict = torch.load(model_state_path, map_location=map_location)

            loaded_entries.append(
                {
                    "entry_id": entry.get("entry_id"),
                    "label": entry.get("label"),
                    "source": entry.get("source"),
                    "result": result,
                    "config": entry_config,
                    **resolve_plotting_style(
                        label=entry.get("label"),
                        source=entry.get("source"),
                        existing_plotting=(entry_config or {}).get("plotting", {}),
                    ),
                    "model_state_dict": model_state_dict,
                    "paths": _entry_paths(result_path, entry_config_path, model_state_path, log_path),
                }
            )
    else:
        artifact_index_path = _resolve_run_path(run_dir_path, manifest_paths.get("artifact_index"))
        if artifact_index_path is None:
            artifact_index_path = run_dir_path / "artifacts" / "index.json"

        artifact_index = json.loads(artifact_index_path.read_text(encoding="utf-8"))
        result_artifact = next(
            (artifact for artifact in artifact_index.get("artifacts", []) if artifact.get("kind") == "trajectory_result_pickle"),
            None,
        )
        if result_artifact is None:
            raise FileNotFoundError("No trajectory_result_pickle found in artifact index.")

        result_path = _resolve_run_path(run_dir_path, result_artifact.get("path"))
        if result_path is None:
            raise FileNotFoundError("Unable to resolve trajectory_result_pickle path from artifact index.")
        with result_path.open("rb") as fh:
            result = pickle.load(fh)

        model_state_dict: dict[str, Any] | None = None
        model_artifact = next(
            (artifact for artifact in artifact_index.get("artifacts", []) if artifact.get("kind") == "model_state_dict"),
            None,
        )
        model_state_path = _resolve_run_path(run_dir_path, model_artifact.get("path")) if model_artifact else None
        if load_model_state_dict and model_state_path is not None and model_state_path.exists():
            model_state_dict = torch.load(model_state_path, map_location=map_location)

        loaded_entries.append(
            {
                "entry_id": "00",
                "label": manifest.get("label"),
                "source": "pinn",
                "result": result,
                "config": config,
                **resolve_plotting_style(
                    label=manifest.get("label"),
                    source="pinn",
                    existing_plotting=(config or {}).get("plotting", {}),
                ),
                "model_state_dict": model_state_dict,
                "paths": _entry_paths(result_path, config_json_path, model_state_path, None),
            }
        )

    return {
        "label": manifest.get("label"),
        "run_id": manifest.get("run_id"),
        "run_dir": str(run_dir_path),
        "manifest": manifest,
        "config": config,
        "summary": summary,
        "aggregate_summary": aggregate_summary,
        "entries": loaded_entries,
    }

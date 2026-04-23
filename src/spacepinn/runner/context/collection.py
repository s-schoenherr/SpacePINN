from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

import torch

from .common import _config_hash, _slugify, _to_jsonable, capture_environment, capture_git_state


@dataclass
class RunCollectionContext:
    label: str
    run_root: str = "runs"
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        run_root_path = Path(self.run_root).expanduser().resolve()
        year = self.timestamp_utc.strftime("%Y")
        month = self.timestamp_utc.strftime("%m")
        stamp = self.timestamp_utc.strftime("%Y%m%d_%H%M%S")
        slug = _slugify(self.label)
        self.run_id = f"{stamp}_{slug}"
        self.run_dir = run_root_path / year / month / self.run_id

        self.artifact_dir = self.run_dir / "artifacts"
        self.plot_dir = self.artifact_dir / "plots"
        self.log_dir = self.artifact_dir / "logs"
        self.model_dir = self.artifact_dir / "model"
        self.result_dir = self.artifact_dir / "result"
        self.config_dir = self.run_dir / "configs"
        self.system_dir = self.run_dir / "system"

        self.manifest_path = self.run_dir / "manifest.json"
        self.config_path = self.run_dir / "config.json"
        self.summary_path = self.run_dir / "summary.json"
        self.aggregate_summary_path = self.run_dir / "aggregate_summary.json"
        self.artifact_index_path = self.artifact_dir / "index.json"
        self.environment_path = self.system_dir / "environment.json"
        self.git_path = self.system_dir / "git.json"
        self.error_path = self.run_dir / "error.json"

        self.started_at = self.timestamp_utc
        self.finished_at: datetime | None = None

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def start(self) -> None:
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(self.environment_path, capture_environment())
        self._write_json(self.git_path, capture_git_state())
        self._write_json(self.artifact_index_path, {"artifacts": []})
        self._write_json(
            self.config_path,
            {
                "label": self.label,
                "run_id": self.run_id,
                "entries": [],
            },
        )
        self._write_json(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "label": self.label,
                "status": "running",
                "started_at_utc": self.started_at.isoformat(),
                "finished_at_utc": None,
                "duration_seconds": None,
                "num_entries": 0,
                "entries": [],
                "paths": {
                    "config_json": str(self.config_path),
                    "summary": str(self.summary_path),
                    "aggregate_summary": str(self.aggregate_summary_path),
                    "artifact_index": str(self.artifact_index_path),
                    "environment": str(self.environment_path),
                    "git": str(self.git_path),
                },
            },
        )

    def register_artifact(self, path: Path, kind: str, entry_id: str | None = None) -> None:
        self.artifact_index_path.parent.mkdir(parents=True, exist_ok=True)
        rel_path = str(path.relative_to(self.run_dir))
        artifact = {"kind": kind, "path": rel_path}
        if entry_id is not None:
            artifact["entry_id"] = entry_id
        self.artifacts.append(artifact)
        self._write_json(self.artifact_index_path, {"artifacts": self.artifacts})

    def add_entry(
        self,
        *,
        label: str,
        result: Any,
        config: dict[str, Any] | None = None,
        model: Any | None = None,
        source: str = "pinn",
        log_text: str | None = None,
        log_filename: str | None = None,
    ) -> dict[str, Any]:
        idx = len(self.entries)
        entry_id = f"{idx:02d}_{_slugify(label)}"

        entry_config_payload = None
        config_sha256 = None
        config_json_path = None
        config_pickle_path = None
        if config is not None:
            entry_config_payload = _to_jsonable(config)
            config_sha256 = _config_hash(entry_config_payload)
            config_json_path = self.config_dir / f"{entry_id}.json"
            config_pickle_path = self.config_dir / f"{entry_id}.pkl"
            config_json_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(config_json_path, entry_config_payload)
            config_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with config_pickle_path.open("wb") as fh:
                import pickle

                pickle.dump(config, fh)
            self.register_artifact(config_json_path, kind="entry_config_json", entry_id=entry_id)
            self.register_artifact(config_pickle_path, kind="entry_config_pickle", entry_id=entry_id)

        result_pickle_path = self.result_dir / f"{entry_id}_trajectory_result.pkl"
        result_pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with result_pickle_path.open("wb") as fh:
            import pickle

            pickle.dump(result, fh)
        self.register_artifact(result_pickle_path, kind="trajectory_result_pickle", entry_id=entry_id)

        model_path = None
        if model is not None and hasattr(model, "state_dict"):
            model_path = self.model_dir / f"{entry_id}_model_state_dict.pt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            self.register_artifact(model_path, kind="model_state_dict", entry_id=entry_id)

        log_path = None
        if log_text is not None:
            resolved_log_filename = log_filename or f"{entry_id}_log.txt"
            log_path = self.log_dir / resolved_log_filename
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(log_text, encoding="utf-8")
            self.register_artifact(log_path, kind="entry_log", entry_id=entry_id)

        summary = {
            "entry_id": entry_id,
            "label": label,
            "source": source,
            "delta_v": getattr(result, "delta_v", None),
            "coordinate_system": getattr(result, "coordinate_system", None),
            "t_total": getattr(result, "t_total", None),
            "runtime_seconds": getattr(result, "runtime_seconds", None),
            "solver": getattr(result, "solver_metadata", None),
            "final_loss": result.loss[-1] if getattr(result, "loss", None) else None,
            "final_loss_physics": result.loss_physics[-1] if getattr(result, "loss_physics", None) else None,
            "final_loss_bc": result.loss_bc[-1] if getattr(result, "loss_bc", None) else None,
            "epochs_total": len(result.loss) if getattr(result, "loss", None) else 0,
        }

        entry = {
            "entry_id": entry_id,
            "label": label,
            "source": source,
            "config_sha256": config_sha256,
            "config": entry_config_payload,
            "paths": {
                "config_json": str(config_json_path) if config_json_path is not None else None,
                "config_pickle": str(config_pickle_path) if config_pickle_path is not None else None,
                "result_pickle": str(result_pickle_path),
                "model_state_dict": str(model_path) if model_path is not None else None,
                "log_file": str(log_path) if log_path is not None else None,
            },
            "summary": summary,
        }
        self.entries.append(entry)
        return entry

    def finalize_success(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

        config_payload = {
            "label": self.label,
            "run_id": self.run_id,
            "entries": [
                {
                    "entry_id": entry["entry_id"],
                    "label": entry["label"],
                    "source": entry["source"],
                    "config_sha256": entry["config_sha256"],
                    "config": entry["config"],
                }
                for entry in self.entries
            ],
        }
        summary_payload = {
            "label": self.label,
            "run_id": self.run_id,
            "num_entries": len(self.entries),
            "entries": [entry["summary"] for entry in self.entries],
        }
        manifest_payload = {
            "run_id": self.run_id,
            "label": self.label,
            "status": "completed",
            "started_at_utc": self.started_at.isoformat(),
            "finished_at_utc": self.finished_at.isoformat(),
            "duration_seconds": self._duration_seconds(),
            "num_entries": len(self.entries),
            "entries": [
                {
                    "entry_id": entry["entry_id"],
                    "label": entry["label"],
                    "source": entry["source"],
                    "paths": entry["paths"],
                }
                for entry in self.entries
            ],
            "paths": {
                "config_json": str(self.config_path),
                "summary": str(self.summary_path),
                "aggregate_summary": str(self.aggregate_summary_path),
                "artifact_index": str(self.artifact_index_path),
                "environment": str(self.environment_path),
                "git": str(self.git_path),
            },
        }

        self._write_json(self.config_path, config_payload)
        self._write_json(self.summary_path, summary_payload)
        self._write_json(self.manifest_path, manifest_payload)

    def finalize_failure(self, error: Exception) -> None:
        self.finished_at = datetime.now(timezone.utc)
        self._write_json(
            self.error_path,
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        self._write_json(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "label": self.label,
                "status": "failed",
                "started_at_utc": self.started_at.isoformat(),
                "finished_at_utc": self.finished_at.isoformat(),
                "duration_seconds": self._duration_seconds(),
                "num_entries": len(self.entries),
                "entries": [
                    {
                        "entry_id": entry["entry_id"],
                        "label": entry["label"],
                        "source": entry["source"],
                        "paths": entry["paths"],
                    }
                    for entry in self.entries
                ],
            "paths": {
                "config_json": str(self.config_path),
                "summary": str(self.summary_path),
                "aggregate_summary": str(self.aggregate_summary_path),
                "artifact_index": str(self.artifact_index_path),
                "environment": str(self.environment_path),
                "git": str(self.git_path),
                    "error": str(self.error_path),
                },
            },
        )

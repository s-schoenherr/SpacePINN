from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any

from .common import _config_hash, _slugify, _to_jsonable, capture_environment, capture_git_state


@dataclass
class RunContext:
    config: dict[str, Any]
    label: str
    run_root: str = "runs"
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    artifacts: list[dict[str, str]] = field(default_factory=list)

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
        self.model_dir = self.artifact_dir / "model"
        self.result_dir = self.artifact_dir / "result"
        self.system_dir = self.run_dir / "system"

        self.manifest_path = self.run_dir / "manifest.json"
        self.config_path = self.run_dir / "config.json"
        self.config_pickle_path = self.run_dir / "config.pkl"
        self.summary_path = self.run_dir / "summary.json"
        self.artifact_index_path = self.artifact_dir / "index.json"
        self.environment_path = self.system_dir / "environment.json"
        self.git_path = self.system_dir / "git.json"
        self.error_path = self.run_dir / "error.json"

        self.config_jsonable = _to_jsonable(self.config)
        self.config_sha256 = _config_hash(self.config_jsonable)
        self.started_at = self.timestamp_utc
        self.finished_at: datetime | None = None

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _duration_seconds(self) -> float | None:
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def update_config(self, config: dict[str, Any]) -> None:
        self.config = config
        self.config_jsonable = _to_jsonable(self.config)
        self.config_sha256 = _config_hash(self.config_jsonable)
        self._write_json(self.config_path, self.config_jsonable)
        with self.config_pickle_path.open("wb") as fh:
            import pickle

            pickle.dump(self.config, fh)

    def start(self) -> None:
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.system_dir.mkdir(parents=True, exist_ok=True)

        self.update_config(self.config)

        self._write_json(self.environment_path, capture_environment())
        self._write_json(self.git_path, capture_git_state())
        self._write_json(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "label": self.label,
                "status": "running",
                "started_at_utc": self.started_at.isoformat(),
                "finished_at_utc": None,
                "duration_seconds": None,
                "config_sha256": self.config_sha256,
                "paths": {
                    "config_json": str(self.config_path),
                    "config_pickle": str(self.config_pickle_path),
                    "summary": str(self.summary_path),
                    "artifact_index": str(self.artifact_index_path),
                    "environment": str(self.environment_path),
                    "git": str(self.git_path),
                },
            },
        )
        self._write_json(self.artifact_index_path, {"artifacts": []})

    def register_artifact(self, path: Path, kind: str) -> None:
        rel_path = str(path.relative_to(self.run_dir))
        self.artifacts.append({"kind": kind, "path": rel_path})
        self._write_json(self.artifact_index_path, {"artifacts": self.artifacts})

    def finalize_success(self, summary: dict[str, Any]) -> None:
        self.finished_at = datetime.now(timezone.utc)
        self._write_json(self.summary_path, _to_jsonable(summary))
        self._write_json(
            self.manifest_path,
            {
                "run_id": self.run_id,
                "label": self.label,
                "status": "completed",
                "started_at_utc": self.started_at.isoformat(),
                "finished_at_utc": self.finished_at.isoformat(),
                "duration_seconds": self._duration_seconds(),
                "config_sha256": self.config_sha256,
                "paths": {
                    "config_json": str(self.config_path),
                    "config_pickle": str(self.config_pickle_path),
                    "summary": str(self.summary_path),
                    "artifact_index": str(self.artifact_index_path),
                    "environment": str(self.environment_path),
                    "git": str(self.git_path),
                },
            },
        )

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
                "config_sha256": self.config_sha256,
                "paths": {
                    "config_json": str(self.config_path),
                    "config_pickle": str(self.config_pickle_path),
                    "summary": str(self.summary_path),
                    "artifact_index": str(self.artifact_index_path),
                    "environment": str(self.environment_path),
                    "git": str(self.git_path),
                    "error": str(self.error_path),
                },
            },
        )

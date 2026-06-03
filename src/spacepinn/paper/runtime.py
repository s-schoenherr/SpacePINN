from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_PLACEHOLDER = Path("runs") / "YYYY" / "MM" / "<run_id>"


def smoke_mode_enabled() -> bool:
    return os.getenv("FAST_SMOKE", "0") == "1"


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def resolve_run_dir(run_dir: str | Path = DEFAULT_RUN_PLACEHOLDER) -> Path:
    run_path = Path(run_dir).expanduser()
    if "<run_id>" in str(run_path):
        raise ValueError("Set run_dir to a concrete run folder instead of the <run_id> placeholder before executing the example.")
    return resolve_repo_path(run_path)

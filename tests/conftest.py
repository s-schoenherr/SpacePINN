from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    parser.addoption(
        "--run-dir",
        action="append",
        default=[],
        help="Path to a saved run directory with manifest.json; can be passed multiple times.",
    )
    parser.addoption(
        "--bc-atol",
        action="store",
        type=float,
        default=1e-3,
        help="Absolute tolerance for boundary-condition checks on saved runs.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_saved_runs: test consumes saved run directories from --run-dir or RUN_DIRS.",
    )


@pytest.fixture
def saved_run_dirs(pytestconfig):
    run_dirs = list(pytestconfig.getoption("--run-dir") or [])
    run_dirs.extend(entry.strip() for entry in os.getenv("RUN_DIRS", "").split(",") if entry.strip())

    resolved = []
    for run_dir in run_dirs:
        path = Path(run_dir).expanduser()
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        resolved.append(path)
    return resolved


@pytest.fixture
def boundary_abs_tolerance(pytestconfig):
    return float(pytestconfig.getoption("--bc-atol"))

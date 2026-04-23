import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def pytest_addoption(parser):
    parser.addoption(
        "--run-dir",
        action="append",
        default=[],
        help="Path to a saved run directory (can be passed multiple times).",
    )
    parser.addoption(
        "--bc-atol",
        action="store",
        type=float,
        default=1e-3,
        help="Absolute tolerance for boundary-condition checks on saved runs.",
    )
    parser.addoption(
        "--golden-record",
        action="store",
        default=None,
        help="Golden record id under docs/golden_records/<record_id>.",
    )
    parser.addoption(
        "--golden-root",
        action="store",
        default="docs/golden_records",
        help="Root directory that contains golden records and manifest.json.",
    )
    parser.addoption(
        "--golden-entry-id",
        action="append",
        default=[],
        help="Optional golden record entry id to compare (can be passed multiple times).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_saved_runs: test consumes saved run directories from --run-dir or RUN_DIRS env var.",
    )
    config.addinivalue_line(
        "markers",
        "requires_golden_record: test consumes a golden record id from --golden-record.",
    )


@pytest.fixture
def saved_run_dirs(pytestconfig):
    run_dirs = []
    cli_dirs = pytestconfig.getoption("--run-dir") or []
    run_dirs.extend(cli_dirs)

    env_dirs = [entry.strip() for entry in (os.getenv("RUN_DIRS", "")).split(",") if entry.strip()]
    run_dirs.extend(env_dirs)

    resolved = []
    for run_dir in run_dirs:
        path = Path(run_dir).expanduser()
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        resolved.append(path)

    return resolved


@pytest.fixture
def boundary_abs_tolerance(pytestconfig):
    return float(pytestconfig.getoption("--bc-atol"))


@pytest.fixture
def golden_record_id(pytestconfig):
    return pytestconfig.getoption("--golden-record")


@pytest.fixture
def golden_root_dir(pytestconfig):
    golden_root = Path(pytestconfig.getoption("--golden-root")).expanduser()
    if not golden_root.is_absolute():
        golden_root = (REPO_ROOT / golden_root).resolve()
    return golden_root


@pytest.fixture
def golden_entry_ids(pytestconfig):
    return list(pytestconfig.getoption("--golden-entry-id") or [])

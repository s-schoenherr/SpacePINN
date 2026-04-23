from __future__ import annotations

import functools
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _slugify(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
    normalized = "".join(ch.lower() if ch.isascii() else "-" for ch in value)
    normalized = "".join(ch if ch in allowed else "-" for ch in normalized)
    normalized = normalized.strip("-")
    return normalized or "run"


def _callable_name(fn: Any) -> str:
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return repr(fn)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.nn.Parameter):
        return {
            "__type__": "torch.Parameter",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": bool(value.requires_grad),
            "value": _to_jsonable(value.detach().cpu().numpy()),
        }
    if isinstance(value, torch.Tensor):
        return {
            "__type__": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "requires_grad": bool(value.requires_grad),
            "value": _to_jsonable(value.detach().cpu().numpy()),
        }
    if isinstance(value, functools.partial):
        return {
            "__type__": "functools.partial",
            "func": _callable_name(value.func),
            "args": _to_jsonable(value.args),
            "keywords": _to_jsonable(value.keywords or {}),
        }
    if callable(value):
        return {"__type__": "callable", "name": _callable_name(value)}
    return {"__type__": type(value).__name__, "repr": repr(value)}


def _config_hash(config_jsonable: dict[str, Any]) -> str:
    payload = json.dumps(config_jsonable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_git_state() -> dict[str, Any]:
    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def capture_environment() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "torch_version": getattr(torch, "__version__", None),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "numpy_version": getattr(np, "__version__", None),
    }


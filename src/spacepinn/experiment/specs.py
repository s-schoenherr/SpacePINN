from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PreparedEntry:
    label: str
    result: Any
    model: Any | None
    config: dict[str, Any] | None
    plotting: dict[str, Any] = field(default_factory=dict)
    source: str = "pinn"

    def as_collection_entry(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "result": self.result,
            "model": self.model,
            "config": self.config,
            "plotting": dict(self.plotting),
            "source": self.source,
        }


@dataclass
class PinnEntrySpec:
    config_builder: Callable[[], dict[str, Any]]
    runtime_mutator: Callable[[dict[str, Any]], None] | None = None
    model_factory: Callable[[dict[str, Any]], Any] | None = None
    source: str = "pinn"


@dataclass
class ExternalEntrySpec:
    label: str
    result: Any
    model: Any | None = None
    config: dict[str, Any] | None = None
    plotting: dict[str, Any] = field(default_factory=dict)
    source: str = "external"


@dataclass
class CollectionSpec:
    label: str
    run_root: str
    entries: list[PreparedEntry]
    summary_fn: Callable[[dict[str, Any]], None] | None = None
    plot_fn: Callable[[dict[str, Any]], None] | None = None

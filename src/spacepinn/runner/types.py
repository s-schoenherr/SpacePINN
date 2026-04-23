from __future__ import annotations

from typing import Any, TypedDict


class EntryPathsDict(TypedDict):
    result_pickle: str
    config_json: str | None
    model_state_dict: str | None
    log_file: str | None


class LoadedEntryDict(TypedDict):
    entry_id: str | None
    label: str | None
    source: str | None
    result: Any
    config: dict[str, Any] | None
    model_state_dict: dict[str, Any] | None
    paths: EntryPathsDict


class LoadedRunDict(TypedDict):
    label: str | None
    run_id: str | None
    run_dir: str
    manifest: dict[str, Any]
    config: dict[str, Any] | None
    summary: dict[str, Any] | None
    aggregate_summary: dict[str, Any] | None
    entries: list[LoadedEntryDict]

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import sys
from typing import Callable, TextIO


class _Tee:
    def __init__(self, *streams: TextIO):
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def capture_baseline_entry(builder: Callable[[], dict], *, log_filename: str) -> dict:
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    with redirect_stdout(_Tee(sys.stdout, stdout_buffer)), redirect_stderr(_Tee(sys.stderr, stderr_buffer)):
        entry = builder()

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    combined_log = stdout_text
    if stderr_text:
        combined_log = combined_log + ("\n" if combined_log and not combined_log.endswith("\n") else "")
        combined_log += "[stderr]\n" + stderr_text

    captured_entry = dict(entry)
    if combined_log:
        captured_entry["log_text"] = combined_log
        captured_entry["log_filename"] = log_filename
    return captured_entry

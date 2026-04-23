from __future__ import annotations

from spacepinn.paper._baseline_capture import capture_baseline_entry


class DummyResult:
    def __init__(self):
        self.solver_metadata = {"status_code": 9}
        self.solver = self.solver_metadata


def test_capture_baseline_entry_tees_terminal_output_and_returns_log_text(capsys):
    result = DummyResult()

    def builder():
        print("OpenGoddard iteration log")
        return {"label": "Baseline", "result": result}

    entry = capture_baseline_entry(builder, log_filename="baseline.log")
    captured = capsys.readouterr()

    assert captured.out == "OpenGoddard iteration log\n"
    assert entry["log_text"] == "OpenGoddard iteration log\n"
    assert entry["log_filename"] == "baseline.log"
    assert result.solver_metadata == {"status_code": 9}
    assert result.solver is result.solver_metadata

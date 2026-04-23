from __future__ import annotations

from spacepinn.opengoddard._solve import print_opengoddard_start


class _DummyProblem:
    number_of_section = 2
    number_of_states = [4, 4]
    maxIterator = 7


def test_print_opengoddard_start_includes_run_context(capsys):
    print_opengoddard_start(
        label="Baseline (OpenGoddard)",
        prob=_DummyProblem(),
        ftol=1e-8,
        maxiter=25,
        details={"problem": "test transfer"},
    )

    out = capsys.readouterr().out
    assert "Starting experiment | label=Baseline (OpenGoddard) | source=opengoddard" in out
    assert "sections=2" in out
    assert "states=[4, 4]" in out
    assert "outer_iterations=7" in out
    assert "slsqp_maxiter=25" in out
    assert "ftol=1e-08" in out
    assert "problem: test transfer" in out

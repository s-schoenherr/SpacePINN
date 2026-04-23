from types import SimpleNamespace

from spacepinn.optimization.engine import OptimizationEngine


def test_pure_adam_convergence_check_is_not_disabled():
    engine = OptimizationEngine.__new__(OptimizationEngine)
    engine.config = SimpleNamespace(
        convergence_threshold=1e-6,
        n_adam=100_000,
        n_lbfgs=0,
        progress_print_interval=250,
    )

    assert engine._should_stop_for_convergence(iteration=249, loss_change=1e-7)
    assert not engine._should_stop_for_convergence(iteration=10, loss_change=1e-7)
    assert not engine._should_stop_for_convergence(iteration=400, loss_change=1e-5)


def test_lbfgs_convergence_waits_until_after_adam_phase():
    engine = OptimizationEngine.__new__(OptimizationEngine)
    engine.config = SimpleNamespace(
        convergence_threshold=1e-6,
        n_adam=1000,
        n_lbfgs=100,
        progress_print_interval=250,
    )

    assert not engine._should_stop_for_convergence(iteration=999, loss_change=1e-7)
    assert engine._should_stop_for_convergence(iteration=1000, loss_change=1e-7)

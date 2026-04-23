import numpy as np


def patch_opengoddard_legendre(problem_cls):
    """Patch OpenGoddard Legendre callbacks with NumPy-based implementations."""

    if getattr(problem_cls, "_swingby_legendre_patched", False):
        return

    def _LegendreFunction_fixed(self, x, n):
        basis = np.polynomial.legendre.Legendre.basis(n)
        return basis(np.asarray(x, dtype=float))

    def _LegendreDerivative_fixed(self, x, n):
        basis = np.polynomial.legendre.Legendre.basis(n)
        return basis.deriv()(np.asarray(x, dtype=float))

    problem_cls._LegendreFunction = _LegendreFunction_fixed
    problem_cls._LegendreDerivative = _LegendreDerivative_fixed
    problem_cls._swingby_legendre_patched = True

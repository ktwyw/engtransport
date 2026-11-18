"""Two root finders, written out so that they can be read: bisection and Newton's method.

Both return the root together with the history of iterates, so that the convergence
can be plotted (Chapter 1).  Only the Python standard library is used.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["bisect", "newton"]


def bisect(f: Callable[[float], float], a: float, b: float, xtol: float = 1e-12, maxiter: int = 200):
    """Find a root of ``f`` in ``[a, b]`` by bisection.

    ``f(a)`` and ``f(b)`` must have opposite signs.  Each step halves the bracket, so the
    error after n steps is at most ``(b - a) / 2**n``.  Returns ``(root, history)``, where
    ``history`` lists the midpoints.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a, [a]
    if fb == 0.0:
        return b, [b]
    if fa * fb > 0.0:
        raise ValueError("bisect needs f(a) and f(b) of opposite signs.")
    history = []
    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = f(m)
        history.append(m)
        if fm == 0.0 or 0.5 * (b - a) < xtol:
            return m, history
        if fa * fm < 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    raise RuntimeError("bisect did not converge.")


def newton(f: Callable[[float], float], dfdx: Callable[[float], float], x0: float,
           xtol: float = 1e-12, maxiter: int = 50):
    """Find a root of ``f`` by Newton's method from the starting guess ``x0``.

    The update is x <- x - f(x)/f'(x).  Near a simple root the error is squared at
    every step.  Returns ``(root, history)``, where ``history`` starts with ``x0``.
    """
    x = float(x0)
    history = [x]
    for _ in range(maxiter):
        slope = dfdx(x)
        if slope == 0.0:
            raise ZeroDivisionError("newton: zero derivative.")
        step = f(x) / slope
        x -= step
        history.append(x)
        if abs(step) < xtol * max(1.0, abs(x)):
            return x, history
    raise RuntimeError("newton did not converge.")

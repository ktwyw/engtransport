"""Transient conduction: the lumped model, series solutions for the slab, cylinder and sphere,
semi-infinite solids, the contact temperature, periodic surface waves, and freezing fronts."""

import functools
import math
import numpy as np


# =============================================================================
# Pure-Python / NumPy Bessel J0 and J1 and Error Function helpers
# =============================================================================
def _bessel_j0_scalar(x: float) -> float:
    """High-precision Bessel function of the first kind J0(x) via power series / Hankel asymptotic expansion."""
    ax = abs(x)
    if ax < 14.0:
        y = x * x
        term = 1.0
        total = 1.0
        for k in range(1, 60):
            term *= -y / (4.0 * k * k)
            total += term
            if abs(term) < 1e-16 * abs(total):
                break
        return total
    z = 8.0 / ax
    y = z * z
    xx = ax - 0.7853981633974483
    p0 = 1.0 - 0.1098628627e-2 * y + 0.2734510407e-4 * (y ** 2) - 0.2073370639e-5 * (y ** 3)
    q0 = -0.1562499995e-1 * z + 0.1430488765e-3 * (z ** 3) - 0.6911147651e-5 * (z ** 5)
    return math.sqrt(0.6366197723675814 / ax) * (math.cos(xx) * p0 - math.sin(xx) * q0)


def _bessel_j1_scalar(x: float) -> float:
    """High-precision Bessel function of the first kind J1(x)."""
    ax = abs(x)
    if ax < 14.0:
        y = x * x
        term = 0.5 * x
        total = term
        for k in range(1, 60):
            term *= -y / (4.0 * k * (k + 1.0))
            total += term
            if abs(term) < 1e-16 * abs(total):
                break
        return total
    z = 8.0 / ax
    y = z * z
    xx = ax - 2.356194490192345
    p1 = 1.0 + 0.183105e-2 * y - 0.3516396496e-4 * (y ** 2) + 0.2457520174e-5 * (y ** 3)
    q1 = 0.04687499995 * z - 0.2002690873e-3 * (z ** 3) + 0.8449199096e-5 * (z ** 5)
    ans = math.sqrt(0.6366197723675814 / ax) * (math.cos(xx) * p1 - math.sin(xx) * q1)
    return ans if x >= 0.0 else -ans


def bessel_j0(x: float | np.ndarray):
    """Bessel function J0(x), for a number or an array."""
    if np.ndim(x) == 0:
        return _bessel_j0_scalar(float(x))
    return np.vectorize(_bessel_j0_scalar, otypes=[float])(x)


def bessel_j1(x: float | np.ndarray):
    """Bessel function J1(x), for a number or an array."""
    if np.ndim(x) == 0:
        return _bessel_j1_scalar(float(x))
    return np.vectorize(_bessel_j1_scalar, otypes=[float])(x)


def erf_array(x: float | np.ndarray):
    """Error function erf(x), for a number or an array."""
    if np.ndim(x) == 0:
        return math.erf(float(x))
    return np.vectorize(math.erf, otypes=[float])(x)


def erfc_array(x: float | np.ndarray):
    """Complementary error function erfc(x), for a number or an array."""
    if np.ndim(x) == 0:
        return math.erfc(float(x))
    return np.vectorize(math.erfc, otypes=[float])(x)


def _bisect_scalar(fn, a: float, b: float, tol: float = 1e-13, max_iter: int = 80) -> float:
    fa = fn(a)
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if b - a < tol:
            return mid
        fm = fn(mid)
        if abs(fm) < 1e-15:
            return mid
        if fa * fm <= 0.0:
            b = mid
        else:
            a, fa = mid, fm
    return 0.5 * (a + b)


# =============================================================================
# 1. Lumped Capacitance Model
# =============================================================================
def lumped_capacitance(t: float | np.ndarray, h: float, As: float, rho: float, V: float, cp: float, T_i: float, T_inf: float) -> dict:
    """Lumped capacitance exponential decay theta(t)/theta_i = exp(-t / tau_th)."""
    tau_th = (rho * V * cp) / (h * As)
    theta_ratio = np.exp(-np.asarray(t) / tau_th)
    T_t = T_inf + (T_i - T_inf) * theta_ratio
    Q_total = rho * V * cp * (T_i - T_inf) * (1.0 - theta_ratio)
    return {"tau_th": float(tau_th), "theta_ratio": theta_ratio, "T": T_t, "Q": Q_total}


# =============================================================================
# 2. Exact Heisler Eigenfunction Series (Slab, Cylinder, Sphere)
# =============================================================================
def _slab_eigenvalues_uncached(Bi: float, n_modes: int = 20) -> np.ndarray:
    """
    Solve xi_n * tan(xi_n) = Bi on ((n-1)*pi, (n-1)*pi + pi/2) for n = 1..n_modes.
    """
    if Bi <= 0.0:
        raise ValueError("Biot number Bi must be positive.")
    roots = np.zeros(n_modes, dtype=float)
    for k in range(n_modes):
        low = k * np.pi + 1e-12
        high = k * np.pi + 0.5 * np.pi - 1e-12
        # Rewrite xi * sin(xi) - Bi * cos(xi) = 0 to avoid tan(pi/2) blowup
        roots[k] = _bisect_scalar(lambda xi: xi * math.sin(xi) - Bi * math.cos(xi), low, high)
    return roots


def _sphere_eigenvalues_uncached(Bi: float, n_modes: int = 20) -> np.ndarray:
    """
    Solve 1 - xi_n * cot(xi_n) = Bi  <=>  sin(xi_n) - xi_n * cos(xi_n) - Bi * sin(xi_n) = 0
    on ((n-1)*pi, n*pi) for n = 1..n_modes.
    """
    if Bi <= 0.0:
        raise ValueError("Biot number Bi must be positive.")
    roots = np.zeros(n_modes, dtype=float)
    for k in range(n_modes):
        low = k * np.pi + 1e-10
        high = (k + 1) * np.pi - 1e-10
        roots[k] = _bisect_scalar(lambda xi: (1.0 - Bi) * math.sin(xi) - xi * math.cos(xi), low, high)
    return roots


# First 15 positive zeros of J0(x) for bracketing cylinder roots
_J0_ZEROS = [
    2.404825557695773, 5.520078110286311, 8.653727912911012, 11.791534439014281, 14.930917708487787,
    18.071063967910924, 21.21163662987926, 24.352471530749302, 27.493479132040253, 30.634606468431976,
    33.77582021357357, 36.91709835366404, 40.05842576462824, 43.19979171317673, 46.34118837166181,
]


def _cylinder_eigenvalues_uncached(Bi: float, n_modes: int = 15) -> np.ndarray:
    """Solve xi_n * J1(xi_n) - Bi * J0(xi_n) = 0 for infinite cylinder transient conduction."""
    if Bi <= 0.0:
        raise ValueError("Biot number Bi must be positive.")
    n_modes = min(n_modes, len(_J0_ZEROS))
    roots = np.zeros(n_modes, dtype=float)
    for k in range(n_modes):
        low = 1e-10 if k == 0 else _J0_ZEROS[k - 1] + 1e-10
        high = _J0_ZEROS[k] - 1e-10
        roots[k] = _bisect_scalar(lambda xi: xi * _bessel_j1_scalar(xi) - Bi * _bessel_j0_scalar(xi), low, high)
    return roots


def transient_1d_series(
    x_star: float | np.ndarray, Fo: float, Bi: float, geometry: str = "slab", n_modes: int = 15
) -> dict:
    """
    Exact multi-term eigenfunction series solution for dimensionless temperature
    theta*(x*, Fo) = (T(x, t) - T_inf) / (T_i - T_inf) and fractional energy loss Q / Q_max.
    geometry in {'slab', 'cylinder', 'sphere'}.
    """
    if Fo < 0.0 or Bi <= 0.0:
        raise ValueError("Require Fo >= 0 and Bi > 0.")

    xs = np.asarray(x_star, dtype=float)
    theta = np.zeros_like(xs, dtype=float)
    Q_ratio = 0.0

    if geometry == "slab":
        eigs = slab_eigenvalues(Bi, n_modes=n_modes)
        for xi in eigs:
            Cn = 4.0 * np.sin(xi) / (2.0 * xi + np.sin(2.0 * xi))
            decay = np.exp(-(xi ** 2) * Fo)
            theta += Cn * decay * np.cos(xi * xs)
            Q_ratio += Cn * (np.sin(xi) / xi) * (1.0 - decay)
    elif geometry == "sphere":
        eigs = sphere_eigenvalues(Bi, n_modes=n_modes)
        for xi in eigs:
            Cn = 4.0 * (np.sin(xi) - xi * np.cos(xi)) / (2.0 * xi - np.sin(2.0 * xi))
            decay = np.exp(-(xi ** 2) * Fo)
            spatial = np.where(np.abs(xs) < 1e-12, 1.0, np.sin(xi * xs) / (xi * np.where(np.abs(xs) < 1e-12, 1.0, xs)))
            theta += Cn * decay * spatial
            j1_sphere = (np.sin(xi) - xi * np.cos(xi)) / (xi ** 3)
            Q_ratio += 3.0 * Cn * j1_sphere * (1.0 - decay)
    elif geometry == "cylinder":
        eigs = cylinder_eigenvalues(Bi, n_modes=min(n_modes, 15))
        for xi in eigs:
            j0_xi = _bessel_j0_scalar(xi)
            j1_xi = _bessel_j1_scalar(xi)
            Cn = (2.0 / xi) * j1_xi / (j0_xi ** 2 + j1_xi ** 2)
            decay = np.exp(-(xi ** 2) * Fo)
            theta += Cn * decay * bessel_j0(xi * xs)
            Q_ratio += 2.0 * Cn * (j1_xi / xi) * (1.0 - decay)
    else:
        raise ValueError("geometry must be 'slab', 'cylinder', or 'sphere'.")

    return {"theta_star": theta, "Q_over_Qmax": float(Q_ratio), "eigenvalues": eigs}


# =============================================================================
# 3. Semi-Infinite Media & Neumann-Stefan Phase Change
# =============================================================================
def semi_infinite_const_surface_T(x: float | np.ndarray, t: float, alpha: float, T_i: float, T_s: float) -> dict:
    """Exact similarity solution for semi-infinite solid with sudden surface temperature T_s."""
    if t <= 0.0 or alpha <= 0.0:
        raise ValueError("Require t > 0 and alpha > 0.")
    eta = np.asarray(x, dtype=float) / (2.0 * np.sqrt(alpha * t))
    T_x = T_s + (T_i - T_s) * erf_array(eta)
    return {"eta": eta, "T": T_x}


def stefan_melting_root(Ste: float) -> float:
    """
    Solve the exact 1-phase Neumann-Stefan transcendental equation:
      lambda * exp(lambda^2) * erf(lambda) = Ste / sqrt(pi)
    where s(t) = 2 * lambda * sqrt(alpha * t) is the phase-change front position.
    """
    if Ste <= 0.0:
        raise ValueError("Stefan number Ste = cp * dT / h_sf must be positive.")
    target = Ste / math.sqrt(math.pi)
    return _bisect_scalar(lambda lam: lam * math.exp(lam * lam) * math.erf(lam) - target, 0.0, 5.0)


@functools.lru_cache(maxsize=256)
def _slab_eigenvalues_cached(Bi: float, n_modes: int) -> tuple:
    return tuple(_slab_eigenvalues_uncached(Bi, n_modes))


def slab_eigenvalues(Bi: float, n_modes: int = 20) -> np.ndarray:
    """Roots of xi tan(xi) = Bi (slab), one in each interval (n pi, n pi + pi/2); cached."""
    return np.array(_slab_eigenvalues_cached(float(Bi), int(n_modes)))


@functools.lru_cache(maxsize=256)
def _sphere_eigenvalues_cached(Bi: float, n_modes: int) -> tuple:
    return tuple(_sphere_eigenvalues_uncached(Bi, n_modes))


def sphere_eigenvalues(Bi: float, n_modes: int = 20) -> np.ndarray:
    """Roots of 1 - xi cot(xi) = Bi (sphere), one in each interval (n pi, (n + 1) pi); cached."""
    return np.array(_sphere_eigenvalues_cached(float(Bi), int(n_modes)))


@functools.lru_cache(maxsize=256)
def _cylinder_eigenvalues_cached(Bi: float, n_modes: int) -> tuple:
    return tuple(_cylinder_eigenvalues_uncached(Bi, n_modes))


def cylinder_eigenvalues(Bi: float, n_modes: int = 15) -> np.ndarray:
    """Roots of xi J1(xi) = Bi J0(xi) (long cylinder), bracketed by the zeros of J0; cached."""
    return np.array(_cylinder_eigenvalues_cached(float(Bi), int(n_modes)))


# =============================================================================
# 4. Semi-infinite solids: contact temperature, periodic surface, freezing ground
# =============================================================================
def contact_temperature(k1: float, rho_c1: float, T1: float, k2: float, rho_c2: float, T2: float) -> float:
    """Interface temperature of two semi-infinite bodies brought into contact.

    Weighted by the thermal effusivities e = sqrt(k rho c); constant in time.
    """
    e1, e2 = math.sqrt(k1 * rho_c1), math.sqrt(k2 * rho_c2)
    return (e1 * T1 + e2 * T2) / (e1 + e2)


def periodic_surface(x, t, alpha: float, T_mean: float, amplitude: float, period: float) -> np.ndarray:
    """Steady periodic temperature below a surface held at T_mean + amplitude cos(2 pi t / period).

    T = T_mean + amplitude exp(-x/d) cos(omega t - x/d), with the damping depth d = sqrt(2 alpha/omega).
    """
    omega = 2.0 * math.pi / period
    d = math.sqrt(2.0 * alpha / omega)
    x = np.asarray(x, dtype=float)
    return T_mean + amplitude * np.exp(-x / d) * np.cos(omega * t - x / d)


def neumann_freezing(k_f: float, C_f: float, k_u: float, C_u: float, latent: float,
                     T_s: float, T_freeze: float, T_i: float) -> dict:
    """Two-phase Neumann solution for freezing of a semi-infinite moist medium.

    The medium, initially unfrozen at ``T_i`` > ``T_freeze``, has its surface held at
    ``T_s`` < ``T_freeze`` from t = 0.  ``k`` and ``C`` (volumetric heat capacity) are those of
    the frozen (f) and unfrozen (u) phases; ``latent`` is the latent heat per unit volume.
    The front is at X = 2 lam sqrt(alpha_f t), where lam solves the Stefan condition.

    Returns a dict with ``lam``, ``alpha_f``, ``alpha_u``, a function ``front(t)`` and a
    function ``T(x, t)``.
    """
    a_f, a_u = k_f / C_f, k_u / C_u
    nu = math.sqrt(a_f / a_u)

    def stefan(lam):
        frozen = k_f * (T_freeze - T_s) * math.exp(-lam * lam) / (math.erf(lam) * math.sqrt(math.pi * a_f))
        unfrozen = (k_u * (T_i - T_freeze) * math.exp(-(lam * nu) ** 2)
                    / (math.erfc(lam * nu) * math.sqrt(math.pi * a_u))) if T_i > T_freeze else 0.0
        return frozen - unfrozen - latent * lam * math.sqrt(a_f)

    hi = 1e-6
    while stefan(hi) > 0:
        hi *= 2.0
    lam = _bisect_scalar(stefan, 1e-12, hi)

    def front(t):
        return 2.0 * lam * np.sqrt(a_f * np.asarray(t, dtype=float))

    def T(x, t):
        x = np.asarray(x, dtype=float)
        X = front(t)
        eta_f = x / (2.0 * math.sqrt(a_f * t))
        eta_u = x / (2.0 * math.sqrt(a_u * t))
        frozen = T_s + (T_freeze - T_s) * erf_array(eta_f) / math.erf(lam)
        unfrozen = T_i - (T_i - T_freeze) * erfc_array(eta_u) / math.erfc(lam * nu)
        return np.where(x < X, frozen, unfrozen)

    return {"lam": lam, "alpha_f": a_f, "alpha_u": a_u, "front": front, "T": T}

"""Molecular diffusion: films with drift (Stefan), the Stefan tube, porous media; porous catalysts."""

import math

import numpy as np


def stefan_stagnant_film_flux(
    c_total: float, D_AB: float, delta: float, x_A1: float, x_A2: float
) -> dict:
    """
    Molar flux N_A [mol/(m^2*s)] of species A diffusing through a stagnant, non-diffusing gas B (N_B = 0)
    across film thickness delta:
      N_A = (c * D_AB / delta) * ln((1 - x_A2) / (1 - x_A1)) = (c * D_AB / delta) * (x_A1 - x_A2) / x_Bm
    Also returns the dilute equimolar Fickian flux J_A = (c * D_AB / delta) * (x_A1 - x_A2) and drift enhancement factor.
    """
    if c_total <= 0.0 or D_AB <= 0.0 or delta <= 0.0:
        raise ValueError("c_total, D_AB, and delta must be positive.")
    if not (0.0 <= x_A1 < 1.0 and 0.0 <= x_A2 < 1.0):
        raise ValueError("Mole fractions x_A1, x_A2 must lie in [0, 1) for a stagnant gas film.")

    fick_flux = (c_total * D_AB / delta) * (x_A1 - x_A2)
    if abs(x_A1 - x_A2) < 1e-12:
        stefan_flux = fick_flux
        x_Bm = 1.0 - 0.5 * (x_A1 + x_A2)
    else:
        stefan_flux = (c_total * D_AB / delta) * np.log((1.0 - x_A2) / (1.0 - x_A1))
        x_Bm = ((1.0 - x_A2) - (1.0 - x_A1)) / np.log((1.0 - x_A2) / (1.0 - x_A1))

    return {
        "N_A": float(stefan_flux),
        "J_A_equimolar": float(fick_flux),
        "x_Bm": float(x_Bm),
        "drift_factor": float(1.0 / x_Bm),
    }


def film_flux(c_total: float, D_AB: float, L: float, x1: float, x2: float, sigma: float = 1.0) -> float:
    """Molar flux of A [mol/(m^2 s)] across a film of thickness L, from x1 at z = 0 to x2 at z = L.

    ``sigma = 1 + N_B/N_A`` fixes how B moves with A: sigma = 1 for B stagnant (Stefan's law),
    sigma = 0 for equimolar counter-diffusion (Fick's law, linear profile), sigma = -1 for a
    surface reaction A -> 2B.  From N_A (1 - sigma x) = -c D dx/dz, Eq. (14.5) of the book,
    N_A = (c D / (sigma L)) ln[(1 - sigma x2)/(1 - sigma x1)].
    """
    if c_total <= 0.0 or D_AB <= 0.0 or L <= 0.0:
        raise ValueError("c_total, D_AB and L must be positive.")
    if abs(sigma) < 1e-12:
        return c_total * D_AB * (x1 - x2) / L
    if 1.0 - sigma * x1 <= 0.0 or 1.0 - sigma * x2 <= 0.0:
        raise ValueError("1 - sigma x must stay positive across the film.")
    return c_total * D_AB / (sigma * L) * math.log((1.0 - sigma * x2) / (1.0 - sigma * x1))


def film_profile(zeta, x1: float, x2: float, sigma: float = 1.0):
    """Mole fraction across the film of ``film_flux`` at zeta = z/L: (1 - sigma x) varies
    geometrically, 1 - sigma x = (1 - sigma x1)^(1 - zeta) (1 - sigma x2)^zeta."""
    zeta = np.asarray(zeta, dtype=float)
    if abs(sigma) < 1e-12:
        return x1 + (x2 - x1) * zeta
    return (1.0 - (1.0 - sigma * x1) ** (1.0 - zeta) * (1.0 - sigma * x2) ** zeta) / sigma


def stefan_tube_level(t, s0: float, D_AB: float, x_s: float, x_top: float, r: float):
    """Depth s(t) [m] of the liquid surface below the mouth of a Stefan tube, quasi-steady.

    r is the molar density of the evaporating phase over that of the gas, c_L/c (for a
    liquid, rho_L/(M c); for water held in a porous layer at volume fraction theta,
    rho_w theta/(M c)).  Eq. (14.8): s^2 = s0^2 + (2 D / r) ln[(1 - x_top)/(1 - x_s)] t.
    """
    t = np.asarray(t, dtype=float)
    K = 2.0 * D_AB / r * math.log((1.0 - x_top) / (1.0 - x_s))
    s = np.sqrt(s0 * s0 + K * t)
    return float(s) if np.ndim(s) == 0 else s


def millington_quirk(porosity: float, air_filled: float | None = None) -> float:
    """Ratio of the effective to the free-air gas diffusivity in a porous medium,
    D_eff/D = theta_a^(10/3)/phi^2 (Millington and Quirk 1961); for a dry layer theta_a = phi
    and the ratio is phi^(4/3)."""
    theta_a = porosity if air_filled is None else air_filled
    if not (0.0 < porosity < 1.0) or not (0.0 <= theta_a <= porosity):
        raise ValueError("need 0 < porosity < 1 and 0 <= air-filled porosity <= porosity.")
    return theta_a ** (10.0 / 3.0) / porosity ** 2


SHAPE_INDEX = {"slab": 0, "cylinder": 1, "sphere": 2}


def i1_over_i0(x):
    """Ratio of modified Bessel functions I1(x)/I0(x), x >= 0, by Gauss's continued fraction
    I1/I0 = x/(2 + x^2/(4 + x^2/(6 + ...))), evaluated backwards (NumPy only)."""
    x = np.asarray(x, dtype=float)
    n_terms = int(np.max(x, initial=0.0)) + 40
    t = np.zeros_like(x)
    for m in range(n_terms, 0, -1):
        t = x * x / (2.0 * m + t)
    out = np.divide(t, x, out=np.full_like(x, 0.0), where=x > 0)
    return float(out) if out.ndim == 0 else out


def effectiveness_first_order(phi, shape: str = "slab"):
    """Effectiveness factor of an isothermal first-order reaction in a pellet, Eqs. (15.4)-(15.6).

    ``phi`` is the Thiele modulus based on the volume-to-surface ratio of the pellet,
    phi = (V/A) sqrt(k/D_eff) (Aris 1957): half-thickness L for a slab, R/2 for a long
    cylinder, R/3 for a sphere.  slab: tanh(phi)/phi; cylinder: I1(2 phi)/(phi I0(2 phi));
    sphere: (1/phi) [1/tanh(3 phi) - 1/(3 phi)].  All tend to 1 at small phi and 1/phi at large.
    """
    p = np.asarray(phi, dtype=float)
    if np.any(p < 0.0):
        raise ValueError("phi must be non-negative.")
    small = p < 1e-4
    q = np.where(small, 1.0, p)
    if shape == "slab":
        eta = np.tanh(q) / q
        series = 1.0 - p ** 2 / 3.0
    elif shape == "cylinder":
        eta = i1_over_i0(2.0 * q) / q
        series = 1.0 - p ** 2 / 2.0
    elif shape == "sphere":
        eta = (1.0 / np.tanh(3.0 * q) - 1.0 / (3.0 * q)) / q
        series = 1.0 - 3.0 * p ** 2 / 5.0
    else:
        raise ValueError("shape must be 'slab', 'cylinder' or 'sphere'.")
    eta = np.where(small, series, eta)
    return float(eta) if eta.ndim == 0 else eta


def generalized_modulus(phi, order: float):
    """Bischoff's (1965) modulus for an n-th order reaction, phi_G = phi sqrt((n + 1)/2), with
    phi = (V/A) sqrt(k c_s^(n-1)/D_eff); using it in the first-order formula gives the right
    large-phi limit for any order n > -1."""
    return np.asarray(phi, dtype=float) * math.sqrt((order + 1.0) / 2.0)


def pellet_fd(phi, shape: str = "slab", order: float = 1.0, n: int = 200, activity=None):
    """Concentration in a pellet with an n-th order reaction, by finite differences and Newton.

    Solves u'' + (s/x) u' = Phi^2 a(x) u^order on 0 < x < 1 with u'(0) = 0 and u(1) = 1, where
    s = 0, 1, 2 for slab, cylinder, sphere, Phi = (s + 1) phi is the modulus on the pellet's
    half-thickness or radius, and a(x) an optional activity profile (default 1).  Conservative
    second-order differences on n cells and Newton's method, for orders >= 0.5.
    Returns x, u and the effectiveness factor from the surface flux, u'(1)/(Phi^2 int a x^s dx),
    the actual rate over the rate the same catalyst would give at the surface concentration.
    """
    s = SHAPE_INDEX[shape]
    Phi = (s + 1.0) * float(phi)
    x = np.linspace(0.0, 1.0, n + 1)
    h = 1.0 / n
    a = np.ones(n + 1) if activity is None else np.asarray(activity(x), dtype=float)
    xf = 0.5 * (x[1:] + x[:-1])                     # cell faces between nodes
    wf = xf ** s                                      # face weights x^s
    vol = np.empty(n + 1)                             # control volumes, integral of x^s dx
    vol[1:-1] = ((x[1:-1] + h / 2) ** (s + 1) - (x[1:-1] - h / 2) ** (s + 1)) / (s + 1)
    vol[0] = (h / 2) ** (s + 1) / (s + 1)
    vol[-1] = (1.0 - (1.0 - h / 2) ** (s + 1)) / (s + 1)
    if order < 0.5:
        raise ValueError("pellet_fd handles orders >= 0.5; lower orders leave a dead core.")
    u = np.ones(n + 1)
    idx = np.arange(n)
    for _ in range(100):
        up = np.maximum(u, 0.0)
        g = up ** order
        dg = order * np.maximum(up, 1e-300) ** (order - 1.0) * (u > 0)
        r = np.zeros(n + 1)
        flux = wf * (u[1:] - u[:-1]) / h             # x^s du/dx on faces
        r[:-1] += flux
        r[1:] -= flux
        r -= Phi ** 2 * a * vol * g
        J = np.zeros((n + 1, n + 1))
        J[idx, idx] -= wf / h
        J[idx, idx + 1] += wf / h
        J[idx + 1, idx + 1] -= wf / h
        J[idx + 1, idx] += wf / h
        J[np.arange(n + 1), np.arange(n + 1)] -= Phi ** 2 * a * vol * dg
        r[-1], J[-1, :] = u[-1] - 1.0, 0.0
        J[-1, -1] = 1.0
        du = np.linalg.solve(J, -r)
        u = u + du
        if np.max(np.abs(du)) < 1e-13:
            break
    g_last = max(u[-1], 0.0) ** order
    # surface flux from the balance of the last half cell: x^s u'(1) = flux into it + reaction in it
    flux_last = wf[-1] * (u[-1] - u[-2]) / h
    grad = flux_last + Phi ** 2 * a[-1] * vol[-1] * g_last
    eta = grad / (Phi ** 2 * np.sum(a * vol))       # actual rate over the rate at surface concentration
    return {"x": x, "u": u, "eta": float(eta)}


def observable_modulus(rate_obs: float, L: float, D_eff: float, c_s: float) -> float:
    """Weisz-Prater observable modulus Phi = r_obs L^2/(D_eff c_s) = eta phi^2 (L = V/A)."""
    return rate_obs * L * L / (D_eff * c_s)


def shrinking_core_depth(t, D_eff: float, c0: float, a: float, x0: float = 0.0, R_outer: float = 0.0):
    """Depth of a sharp reaction front advancing into a solid by diffusion through the reacted
    layer, quasi-steady (Eq. 15.11): a dx/dt = c0/(R_outer + x/D_eff), so
    x^2/(2 D_eff) + R_outer x = c0 t/a + x0^2/(2 D_eff) + R_outer x0.  ``R_outer`` is an extra
    resistance in series at the surface (a coating), s/m."""
    t = np.asarray(t, dtype=float)
    C = c0 * t / a + x0 * x0 / (2.0 * D_eff) + R_outer * x0
    x = D_eff * (-R_outer + np.sqrt(R_outer ** 2 + 2.0 * C / D_eff))
    return float(x) if np.ndim(x) == 0 else x


def catalyst_effectiveness_factor(phi: float | np.ndarray, geometry: str = "sphere") -> float | np.ndarray:
    """
    Exact isothermal first-order reaction-diffusion effectiveness factor eta(phi) in a porous catalyst:
      - 'slab' (half-thickness L, phi = L * sqrt(k / D_eff)):
          eta = tanh(phi) / phi
      - 'sphere' (radius R, phi = R * sqrt(k / D_eff), or Aris generalized phi_G = (R/3)*sqrt(k/D_eff)):
          Using standard spherical Thiele modulus phi = R * sqrt(k / D_eff):
          eta = (3 / phi^2) * (phi * coth(phi) - 1)
    """
    p = np.asarray(phi, dtype=float)
    if np.any(p < 0.0):
        raise ValueError("Thiele modulus phi must be non-negative.")

    if geometry == "slab":
        return np.where(p < 1e-6, 1.0 - (p ** 2) / 3.0, np.tanh(p) / np.where(p < 1e-6, 1.0, p))
    if geometry == "sphere":
        safe_p = np.where(p < 1e-4, 1.0, p)
        coth_p = 1.0 / np.tanh(safe_p)
        exact = (3.0 / (safe_p ** 2)) * (safe_p * coth_p - 1.0)
        series = 1.0 - (p ** 2) / 15.0 + 2.0 * (p ** 4) / 315.0
        return np.where(p < 1e-4, series, exact)
    raise ValueError("geometry must be 'slab' or 'sphere'.")


def solve_spherical_catalyst_fd(phi: float, N: int = 401) -> dict:
    """
    From-scratch finite-difference BVP solver for spherical reaction-diffusion:
      (1 / xi^2) d/dxi (xi^2 dc/dxi) - phi^2 * c = 0,   dc/dxi(0) = 0,  c(1) = 1.
    At xi = 0, L'Hopital's rule gives (1/xi^2) d/dxi(xi^2 dc/dxi) -> 3 d^2c/dxi^2.
    """
    xi = np.linspace(0.0, 1.0, N)
    h = xi[1] - xi[0]
    A = np.zeros((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    # Center symmetry node i = 0: 3 * (2*c_1 - 2*c_0) / h^2 - phi^2 * c_0 = 0
    A[0, 0] = -6.0 / (h ** 2) - (phi ** 2)
    A[0, 1] = 6.0 / (h ** 2)

    # Interior nodes i = 1 .. N - 2:
    for i in range(1, N - 1):
        r_i = xi[i]
        A[i, i - 1] = 1.0 / (h ** 2) - 1.0 / (r_i * h)
        A[i, i] = -2.0 / (h ** 2) - (phi ** 2)
        A[i, i + 1] = 1.0 / (h ** 2) + 1.0 / (r_i * h)

    # Surface Dirichlet node i = N - 1: c(1) = 1
    A[-1, -1] = 1.0
    b[-1] = 1.0

    c_num = np.linalg.solve(A, b)
    # Volume-averaged effectiveness factor eta = 3 * int_0^1 c(xi) * xi^2 dxi
    eta_num = 3.0 * float(np.trapezoid(c_num * (xi ** 2), xi))
    return {"xi": xi, "c": c_num, "effectiveness": eta_num}

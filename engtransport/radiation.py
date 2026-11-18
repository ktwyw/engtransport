"""Thermal radiation: Planck's law and band fractions, selective surfaces, view factors and
their algebra, the crossed-string rule, radiosity networks and radiation shields."""

import math

import numpy as np

H_PLANCK = 6.62607015e-34     # J s, exact (SI 2019)
C_LIGHT = 299792458.0         # m/s, exact
K_BOLTZMANN = 1.380649e-23    # J/K, exact
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m^2*K^4)]
C1_PLANCK = 2.0 * math.pi * H_PLANCK * C_LIGHT**2 * 1e24   # first radiation constant 2 pi h c^2 [W um^4/m^2]
C2_PLANCK = H_PLANCK * C_LIGHT / K_BOLTZMANN * 1e6          # second radiation constant h c / k [um K]
WIEN_B = 2897.771955       # Wien displacement constant (lambda_max * T) [um*K]
T_SUN = 5772.0             # K, nominal effective temperature of the Sun (IAU 2015 Resolution B3)
R_SUN = 6.957e8            # m, nominal solar radius (IAU 2015 Resolution B3)
AU = 1.495978707e11        # m, astronomical unit (IAU 2012)


def planck_spectral_emissive_power(lam_um: float | np.ndarray, T: float) -> float | np.ndarray:
    """Spectral blackbody emissive power E_{lambda,b}(lambda, T) [W/(m^2*um)]; lambda and T may be arrays."""
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("Temperature T [K] must be positive.")
    lam = np.asarray(lam_um, dtype=float)
    if np.any(lam <= 0.0):
        raise ValueError("Wavelength lam_um [um] must be positive.")
    exponent = np.clip(C2_PLANCK / (lam * T), 1e-12, 500.0)
    return C1_PLANCK / ((lam ** 5) * np.expm1(exponent))


def blackbody_band_fraction(lam_T_um_K, n_terms: int | None = None):
    """Fraction of blackbody emission below the wavelength lambda: F(lambda T), lambda T in um K.

    With x = C2/(lambda T), F = (15/pi^4) int_x^inf z^3/(e^z - 1) dz.  For x >= 0.5 the integral
    is the exponential series sum_m e^{-m x} (x^3/m + 3x^2/m^2 + 6x/m^3 + 6/m^4), summed until
    the terms are below 1e-17 (about 37/x terms, or ``n_terms`` if given).  For x < 0.5 it is
    pi^4/15 minus the integral from 0 to x, from the Bernoulli-number series of z^3/(e^z - 1),
    to the x^15 term.  Both are accurate to about 1e-15.  ``lam_T_um_K`` may be an array.
    (An earlier version switched series at x = 2 with only six Bernoulli terms and was wrong
    by up to 1.5e-6 near lambda T = 7200 um K; the check in Chapter 12 found it.)
    """
    lt = np.atleast_1d(np.asarray(lam_T_um_K, dtype=float))
    out = np.zeros_like(lt)
    pos = lt > 0.0
    x = np.full_like(lt, np.inf)
    x[pos] = C2_PLANCK / lt[pos]
    big = pos & (x >= 0.5) & (x < 700.0)
    if np.any(big):
        xb = x[big]
        m_max = int(np.ceil(37.0 / xb.min())) + 1 if n_terms is None else n_terms
        m = np.arange(1, m_max + 1)[:, None]
        mx = m * xb
        out[big] = (15.0 / np.pi**4) * np.sum(np.exp(-mx) / m**4 * (mx**3 + 3 * mx**2 + 6 * mx + 6), axis=0)
    small = pos & (x < 0.5)
    if np.any(small):
        xs = x[small]
        # int_0^x z^3/(e^z - 1) dz = sum_k B_{2k} x^{2k+3} / ((2k)! (2k+3)) - x^4/8, with B_0 = 1
        B = (1.0, 1.0 / 6, -1.0 / 30, 1.0 / 42, -1.0 / 30, 5.0 / 66, -691.0 / 2730, 7.0 / 6)
        total = -xs**4 / 8.0
        for k, b in enumerate(B):
            total = total + b * xs ** (2 * k + 3) / (math.factorial(2 * k) * (2 * k + 3))
        out[small] = 1.0 - (15.0 / np.pi**4) * total
    out = np.clip(out, 0.0, 1.0)
    return float(out[0]) if np.ndim(lam_T_um_K) == 0 else out


def band_fraction(lam1_um: float, lam2_um: float, T: float) -> float:
    """Fraction of blackbody emission at temperature T between wavelengths lam1 and lam2 [um]."""
    return float(blackbody_band_fraction(lam2_um * T) - blackbody_band_fraction(lam1_um * T))


def wien_constant() -> float:
    """Wien's displacement constant b = lambda_max T [um K], from the root of x = 5 (1 - e^{-x}), x = C2/(lambda T)."""
    x = 5.0
    for _ in range(50):
        f = x - 5.0 * (1.0 - math.exp(-x))
        step = f / (1.0 - 5.0 * math.exp(-x))
        x -= step
        if abs(step) < 1e-15:
            break
    return C2_PLANCK / x


def stefan_boltzmann_from_constants() -> float:
    """sigma = 2 pi^5 k^4 / (15 h^3 c^2) from the exact SI values of h, c and k."""
    return 2.0 * math.pi**5 * K_BOLTZMANN**4 / (15.0 * H_PLANCK**3 * C_LIGHT**2)


def solar_spectrum(lam_um, G: float = 1000.0, T_sun: float = T_SUN):
    """Spectral solar irradiance [W/(m^2 um)] modelled as a blackbody at T_sun, scaled to a total G [W/m^2]."""
    return planck_spectral_emissive_power(lam_um, T_sun) * G / (SIGMA_SB * T_sun**4)


def step_surface(lam_c_um: float, short: float, long: float) -> dict:
    """A spectrally selective surface: emissivity (= absorptivity) ``short`` below lam_c and ``long`` above.

    Returns functions: ``emissivity(T)``, the total emissivity at surface temperature T, and
    ``absorptance(T_source)``, the total absorptance for blackbody radiation from a source at
    T_source (the sun by default).
    """
    def total(T):
        F = blackbody_band_fraction(lam_c_um * T)
        return short * F + long * (1.0 - F)

    return {"emissivity": total, "absorptance": lambda T_source=T_SUN: total(T_source)}


def view_factor_coaxial_disks(r1: float, r2: float, L: float) -> float:
    """View factor F_12 from a disk of radius r1 to a coaxial parallel disk of radius r2 a distance L away."""
    if r1 <= 0.0 or r2 <= 0.0 or L <= 0.0:
        raise ValueError("Disk radii r1, r2 and spacing L must be positive.")
    R1 = r1 / L
    R2 = r2 / L
    S = 1.0 + (1.0 + R2 * R2) / (R1 * R1)
    return float(0.5 * (S - np.sqrt((S ** 2) - 4.0 * ((r2 / r1) ** 2))))


def view_factor_parallel_rectangles(a: float, b: float, c: float) -> float:
    """View factor between two equal, directly opposed parallel rectangles a x b a distance c apart.

    Exact result tabulated by Hamilton and Morgan (NACA TN 2836, 1952), Eq. (13.5) of the book.
    """
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise ValueError("Rectangle sides a, b and spacing c must be positive.")
    X, Y = a / c, b / c
    sx, sy = math.sqrt(1.0 + X * X), math.sqrt(1.0 + Y * Y)
    return 2.0 / (math.pi * X * Y) * (
        0.5 * math.log((1.0 + X * X) * (1.0 + Y * Y) / (1.0 + X * X + Y * Y))
        + X * sy * math.atan(X / sy) + Y * sx * math.atan(Y / sx)
        - X * math.atan(X) - Y * math.atan(Y))


def view_factor_perpendicular_rectangles(l: float, w1: float, w2: float) -> float:
    """View factor F_12 between perpendicular rectangles sharing an edge of length l.

    Surface 1 extends a width w1 from the common edge and surface 2 a width w2
    (Hamilton and Morgan 1952), Eq. (13.6) of the book.
    """
    if l <= 0.0 or w1 <= 0.0 or w2 <= 0.0:
        raise ValueError("Edge length l and widths w1, w2 must be positive.")
    W, H = w1 / l, w2 / l
    W2, H2 = W * W, H * H
    s = math.sqrt(W2 + H2)
    log_term = (math.log((1.0 + W2) * (1.0 + H2) / (1.0 + W2 + H2))
                + W2 * math.log(W2 * (1.0 + W2 + H2) / ((1.0 + W2) * (W2 + H2)))
                + H2 * math.log(H2 * (1.0 + W2 + H2) / ((1.0 + H2) * (W2 + H2))))
    return (W * math.atan(1.0 / W) + H * math.atan(1.0 / H) - s * math.atan(1.0 / s)
            + 0.25 * log_term) / (math.pi * W)


BOX_FACES = ("floor", "roof", "x=0", "x=Lx", "y=0", "y=Ly")


def box_view_factors(Lx: float, Ly: float, Lz: float) -> tuple[np.ndarray, np.ndarray]:
    """Areas and 6 x 6 view-factor matrix of the inside faces of a rectangular box.

    Faces are ordered as in ``BOX_FACES``: floor (z = 0), roof (z = Lz), then the walls
    x = 0, x = Lx, y = 0, y = Ly.  Opposite faces use Eq. (13.5), adjacent faces Eq. (13.6).
    """
    dims = {0: (Lx, Ly, "z"), 1: (Lx, Ly, "z"), 2: (Ly, Lz, "x"), 3: (Ly, Lz, "x"),
            4: (Lx, Lz, "y"), 5: (Lx, Lz, "y")}
    normal = {0: "z", 1: "z", 2: "x", 3: "x", 4: "y", 5: "y"}
    length = {"x": Lx, "y": Ly, "z": Lz}
    A = np.array([dims[i][0] * dims[i][1] for i in range(6)])
    F = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            ni, nj = normal[i], normal[j]
            if ni == nj:                                   # opposite faces
                a, b, _ = dims[i]
                F[i, j] = view_factor_parallel_rectangles(a, b, length[ni])
            else:                                          # adjacent faces share an edge
                edge_axis = ({"x", "y", "z"} - {ni, nj}).pop()
                F[i, j] = view_factor_perpendicular_rectangles(length[edge_axis], length[nj], length[ni])
    return A, F


def _offset_exchange(G, a1, a2, b1, b2):
    """Exchange area between strips [a1, a2] and [b1, b2] along a common direction.

    For two surfaces whose geometry does not change along one direction, the exchange
    area A F between the parts over [a1, a2] and [b1, b2] follows from the exchange
    area G(x) of the aligned parts of length x: (G(a2-b1) + G(a1-b2) - G(a2-b2) - G(a1-b1))/2.
    """
    return 0.5 * (G(a2 - b1) + G(a1 - b2) - G(a2 - b2) - G(a1 - b1))


def box_zone_view_factors(Lx: float, Ly: float, Lz: float, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """Areas and view factors of a box whose four walls are cut into horizontal bands.

    Zones: 0 floor, 1 roof, then for each wall (x=0, x=Lx, y=0, y=Ly) ``n_bands`` bands
    of equal height from the floor up.  Exchange areas between bands follow exactly from
    Eqs. (13.5) and (13.6) by view-factor algebra (``_offset_exchange``).
    """
    n = int(n_bands)
    z = np.linspace(0.0, Lz, n + 1)
    walls = [("x", Ly), ("x", Ly), ("y", Lx), ("y", Lx)]           # normal axis, horizontal width
    N = 2 + 4 * n
    AF = np.zeros((N, N))
    A = np.zeros(N)
    A[0] = A[1] = Lx * Ly
    AF[0, 1] = AF[1, 0] = Lx * Ly * view_factor_parallel_rectangles(Lx, Ly, Lz)
    for w, (ax, width) in enumerate(walls):
        depth = Ly if ax == "y" else Lx
        for k in range(n):
            i = 2 + w * n + k
            A[i] = width * (z[k + 1] - z[k])
            # floor (common edge at z = 0) and roof (common edge at z = Lz)
            g = lambda h: 0.0 if h <= 0 else width * h * view_factor_perpendicular_rectangles(width, h, depth)
            AF[i, 0] = g(z[k + 1]) - g(z[k])
            AF[i, 1] = g(Lz - z[k]) - g(Lz - z[k + 1])
            AF[0, i], AF[1, i] = AF[i, 0], AF[i, 1]
    for w1, (ax1, width1) in enumerate(walls):
        for w2, (ax2, width2) in enumerate(walls):
            if w2 <= w1:
                continue
            if ax1 == ax2:      # opposite walls, a distance apart equal to the other box side
                gap = Lx if ax1 == "x" else Ly
                G = lambda h, wd=width1, gp=gap: 0.0 if abs(h) <= 0 else wd * abs(h) * view_factor_parallel_rectangles(wd, abs(h), gp)
            else:               # adjacent walls share a vertical edge
                G = lambda h, a=width1, b=width2: 0.0 if abs(h) <= 0 else abs(h) * a * view_factor_perpendicular_rectangles(abs(h), a, b)
            for k in range(n):
                for m in range(n):
                    i, j = 2 + w1 * n + k, 2 + w2 * n + m
                    AF[i, j] = AF[j, i] = _offset_exchange(G, z[k], z[k + 1], z[m], z[m + 1])
    F = AF / A[:, None]
    return A, F


def view_factor_crossed_strings(a1, b1, a2, b2) -> float:
    """Hottel's crossed-string rule for two long surfaces in two dimensions.

    The surfaces are the straight strips from point a1 to b1 and from a2 to b2 (2-D
    coordinates).  F_12 = |(crossed strings) - (uncrossed strings)| / (2 L_1), Eq. (13.7);
    the absolute value makes the result independent of the order of the end points.
    Valid when nothing blocks the view between the strips.
    """
    a1, b1, a2, b2 = (np.asarray(p, dtype=float) for p in (a1, b1, a2, b2))
    d = lambda p, q: float(np.hypot(*(p - q)))
    L1 = d(a1, b1)
    if L1 <= 0.0:
        raise ValueError("Strip 1 must have positive length.")
    return abs((d(a1, b2) + d(b1, a2)) - (d(a1, a2) + d(b1, b2))) / (2.0 * L1)


def view_factor_hottel_crossed_strings(
    w1: float, d_diag1: float, d_diag2: float, d_side1: float, d_side2: float
) -> float:
    """Crossed-string rule from string lengths: F_12 = (crossed - uncrossed) / (2 w1)."""
    if w1 <= 0.0:
        raise ValueError("Surface width w1 must be positive.")
    return float(((d_diag1 + d_diag2) - (d_side1 + d_side2)) / (2.0 * w1))


def verify_view_factor_matrix(areas: np.ndarray, F: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Verify that an N x N view-factor matrix F for an enclosure with surface areas A_i satisfies:
      1. Non-negativity: F_ij >= 0
      2. Summation rule: sum_j F_ij == 1 for every surface i
      3. Reciprocity rule: A_i * F_ij == A_j * F_ji for all pairs (i, j)
    """
    A = np.asarray(areas, dtype=float)
    F_mat = np.asarray(F, dtype=float)
    if F_mat.shape != (len(A), len(A)):
        raise ValueError("View factor matrix F must have shape (N, N) matching len(areas).")
    if np.any(F_mat < -tol):
        raise ValueError("View factors cannot be negative.")
    row_sums = np.sum(F_mat, axis=1)
    if not np.allclose(row_sums, 1.0, atol=tol):
        raise ValueError(f"Enclosure summation rule violated: row sums = {row_sums}")
    AF = A[:, None] * F_mat
    if not np.allclose(AF, AF.T, rtol=tol, atol=tol):
        raise ValueError("View-factor reciprocity A_i * F_ij == A_j * F_ji violated.")
    return True


def solve_radiosity_enclosure(areas, emissivities, F, T, reradiating_mask=None, Q=None) -> dict:
    """Diffuse gray enclosure by the radiosity (net-radiation) method, Eqs. (13.8)-(13.11).

    Each surface has either a prescribed temperature or a prescribed net rate of heat
    leaving it, Q_i [W]:
      prescribed T:  J_i - (1 - eps_i) sum_j F_ij J_j = eps_i sigma T_i^4
      prescribed Q:  J_i - sum_j F_ij J_j = Q_i / A_i
    A reradiating (adiabatic) surface is a prescribed Q of zero; it may be marked with
    ``reradiating_mask`` or with Q_i = 0.  ``Q`` is an array with NaN where T is prescribed.
    Returns radiosities J [W/m^2], net rates Q [W] (positive leaving the surface) and the
    surface temperatures T [K] (computed where Q was prescribed).
    """
    A = np.asarray(areas, dtype=float)
    eps = np.asarray(emissivities, dtype=float)
    F_mat = np.asarray(F, dtype=float)
    N = len(A)
    T_arr = np.broadcast_to(np.asarray(T, dtype=float), (N,)).copy()
    if np.any((eps <= 0.0) | (eps > 1.0)):
        raise ValueError("Surface emissivities must lie in (0, 1].")
    verify_view_factor_matrix(A, F_mat)

    Q_known = np.full(N, np.nan) if Q is None else np.asarray(Q, dtype=float).copy()
    if reradiating_mask is not None:
        Q_known[np.asarray(reradiating_mask, dtype=bool)] = 0.0
    fixed_Q = ~np.isnan(Q_known)

    M = np.eye(N)
    rhs = np.zeros(N)
    for i in range(N):
        if fixed_Q[i]:
            M[i, :] -= F_mat[i, :]
            rhs[i] = Q_known[i] / A[i]
        else:
            M[i, :] -= (1.0 - eps[i]) * F_mat[i, :]
            rhs[i] = eps[i] * SIGMA_SB * T_arr[i] ** 4
    J = np.linalg.solve(M, rhs)

    G = F_mat @ J                       # irradiation: A_i G_i = sum_j A_j F_ji J_j = A_i sum_j F_ij J_j
    Q_out = A * (J - G)                 # net rate leaving each surface
    Eb = J + (1.0 - eps) / eps * (J - G)
    T_out = T_arr.copy()
    T_out[fixed_Q] = (Eb[fixed_Q] / SIGMA_SB) ** 0.25
    Q_out[fixed_Q] = Q_known[fixed_Q]
    return {"J": J, "Q": Q_out, "T": T_out, "T_eff": T_out,
            "energy_balance_residual": float(np.sum(A * (J - G)))}


def radiation_shields(T_hot: float, T_cold: float, areas, eps_inner, eps_outer) -> dict:
    """Heat flow through nested surfaces: a hot surface, shields, and a cold surface.

    Surface k = 0 is the hot (innermost) surface and k = m the cold one; each gap is a
    two-surface enclosure in which the inner surface sees only the outer one (F = 1), so
    its resistance is (1 - e_a)/(e_a A_a) + 1/A_a + (1 - e_b)/(e_b A_b), Eq. (13.14).
    ``eps_outer[k]`` is the emissivity of the outward face of surface k (k = 0..m-1) and
    ``eps_inner[k]`` that of the inward face of surface k (k = 1..m); the unused entries
    (``eps_inner[0]``, ``eps_outer[m]``) are ignored.  For plane layers give equal areas.
    Returns the heat flow Q [W] and the temperatures T [K] of all surfaces.
    """
    A = np.asarray(areas, dtype=float)
    ei = np.asarray(eps_inner, dtype=float)
    eo = np.asarray(eps_outer, dtype=float)
    m = len(A) - 1
    R = np.array([(1.0 - eo[k]) / (eo[k] * A[k]) + 1.0 / A[k] + (1.0 - ei[k + 1]) / (ei[k + 1] * A[k + 1])
                  for k in range(m)])
    Q = SIGMA_SB * (T_hot ** 4 - T_cold ** 4) / np.sum(R)
    Eb = SIGMA_SB * T_hot ** 4 - Q * np.concatenate([[0.0], np.cumsum(R)])
    return {"Q": float(Q), "T": (Eb / SIGMA_SB) ** 0.25, "R": R}

"""Boiling and condensation: Nusselt film condensation (with the subcooling of the film),
Rohsenow nucleate boiling, the peak and minimum heat fluxes, and Bromley film boiling."""

import math

import numpy as np

G_STD = 9.80665


def h_condensation_vertical_plate(
    rho_l: float, rho_v: float, k_l: float, mu_l: float, h_fg: float, cp_l: float, T_sat: float, T_s: float, L: float, g: float = G_STD
) -> dict:
    """
    Nusselt (1916) laminar film condensation average heat transfer coefficient h_bar [W/(m^2*K)]
    on a vertical plate of height L, with Rohsenow modified latent heat h_fg' = h_fg + 0.68 * cp_l * (T_sat - T_s).
    """
    dT = T_sat - T_s
    if dT <= 0.0 or L <= 0.0:
        raise ValueError("Condensation requires T_sat > T_s and L > 0.")
    ja = cp_l * dT / h_fg
    h_fg_prime = h_fg * (1.0 + 0.68 * ja)
    h_bar = 0.9428 * ((g * rho_l * (rho_l - rho_v) * (k_l ** 3) * h_fg_prime / (mu_l * dT * L)) ** 0.25)
    q_total_per_width = h_bar * L * dT
    mdot_per_width = q_total_per_width / h_fg_prime
    Re_delta = 4.0 * mdot_per_width / mu_l
    return {"h_bar": float(h_bar), "Jakob": float(ja), "Re_delta": float(Re_delta), "q_per_width": float(q_total_per_width)}


def nusselt_tube_constant() -> float:
    """Nusselt's constant for film condensation on a horizontal tube, from its closed form.

    h_bar = C [g rho_l (rho_l - rho_v) h_fg k^3 / (mu dT D)]^(1/4) with
    C = 4 S^(3/4) / (3 pi 2^(1/4)) and S = int_0^pi sin^(1/3) = sqrt(pi) Gamma(2/3) / Gamma(7/6).
    """
    S = math.sqrt(math.pi) * math.gamma(2.0 / 3.0) / math.gamma(7.0 / 6.0)
    return 4.0 * S ** 0.75 / (3.0 * math.pi * 2.0 ** 0.25)


def h_film_condensation(rho_l: float, rho_v: float, k_l: float, mu_l: float, h_fg: float, dT: float, L: float,
                        geometry: str = "vertical", g: float = G_STD) -> float:
    """Mean coefficient of laminar film condensation (Nusselt 1916) [W/(m^2 K)].

    h_bar = C [g rho_l (rho_l - rho_v) h_fg k^3 / (mu dT L)]^(1/4), with C = 0.943 and L the height
    for a vertical plate, or C = 0.728 and L the diameter for a horizontal tube.  Pass a
    corrected latent heat h_fg' (see ``film_condensation_subcooling``) to include the
    sensible heat given up by the film.
    """
    if dT <= 0.0 or L <= 0.0:
        raise ValueError("Condensation requires dT = T_sat - T_wall > 0 and L > 0.")
    C = {"vertical": (4.0 / 3.0) / 2.0 ** 0.5, "tube": nusselt_tube_constant()}[geometry]
    return float(C * (g * rho_l * (rho_l - rho_v) * h_fg * k_l**3 / (mu_l * dT * L)) ** 0.25)


def film_condensation_subcooling(Ja: float, n_nodes: int = 40) -> dict:
    """Exact effect of the film's subcooling on laminar film condensation on a vertical plate.

    Keeping Nusselt's velocity profile but solving the energy equation of the film with its
    convection terms, the temperature theta(eta) = (T - T_w)/(T_sat - T_w), eta = y/delta,
    obeys theta'' + lam f theta' = 0 with f = eta^2/2 - eta^3/6, theta(0) = 0, theta(1) = 1,
    and the interface balance theta'(1) = lam / (3 Ja), Ja = c_p (T_sat - T_w)/h_fg.  So
    theta' = c exp(-lam F), F = eta^3/6 - eta^4/24, with 1/c = int_0^1 exp(-lam F), and lam
    solves exp(-lam/8) c = lam/(3 Ja).  The heat transfer equals Nusselt's with the latent
    heat replaced by h_fg' = h_fg c^4 (3 Ja / lam); for small Ja, h_fg'/h_fg = 1 + (27/40) Ja.
    Gauss-Legendre quadrature and bisection; returns lam, c and the ratio h_fg'/h_fg.
    """
    if Ja <= 0.0:
        raise ValueError("Ja must be positive.")
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    eta, w = 0.5 * (x + 1.0), 0.5 * w
    F = eta**3 / 6.0 - eta**4 / 24.0

    def c_of(lam):
        return 1.0 / np.sum(w * np.exp(-lam * F))

    def residual(lam):
        return np.exp(-lam / 8.0) * c_of(lam) - lam / (3.0 * Ja)

    lo, hi = 0.0, 3.0 * Ja
    while residual(hi) > 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15 * hi:
            break
    lam = 0.5 * (lo + hi)
    c = c_of(lam)
    return {"lam": float(lam), "c": float(c), "ratio": float(c**4 * 3.0 * Ja / lam)}


def rohsenow_heat_flux(dT_e: float, mu_l: float, h_fg: float, rho_l: float, rho_v: float, sigma: float,
                       cp_l: float, Pr_l: float, C_sf: float = 0.013, n: float = 1.0, g: float = G_STD) -> float:
    """Rohsenow (1952) nucleate pool-boiling heat flux [W/m^2] for an excess temperature dT_e = T_w - T_sat.

    q = mu_l h_fg [g (rho_l - rho_v)/sigma]^(1/2) [c_p dT_e / (C_sf h_fg Pr_l^n)]^3, with C_sf = 0.013
    and n = 1 for water on copper (Rohsenow's value; about +-20 % in q for a given dT_e, much
    more between surfaces).  ``dT_e`` may be an array.
    """
    dT_e = np.asarray(dT_e, dtype=float)
    if np.any(dT_e < 0.0):
        raise ValueError("dT_e must be non-negative.")
    q = mu_l * h_fg * math.sqrt(g * (rho_l - rho_v) / sigma) * (cp_l * dT_e / (C_sf * h_fg * Pr_l**n)) ** 3
    return float(q) if q.ndim == 0 else q


def zuber_critical_heat_flux(rho_l: float, rho_v: float, h_fg: float, sigma: float, g: float = G_STD, C: float = 0.149) -> float:
    """Peak (critical) pool-boiling heat flux on a large flat heater [W/m^2].

    q_max = C rho_v h_fg [sigma g (rho_l - rho_v) / rho_v^2]^(1/4); Zuber's hydrodynamic theory
    gives C = pi/24 = 0.131, and C = 0.149 (default) is Lienhard and Dhir's value for large plates.
    """
    if rho_l <= rho_v or h_fg <= 0.0 or sigma <= 0.0:
        raise ValueError("Invalid fluid properties for Zuber CHF.")
    return float(C * h_fg * rho_v * ((sigma * g * (rho_l - rho_v) / (rho_v ** 2)) ** 0.25))


def minimum_heat_flux(rho_l: float, rho_v: float, h_fg: float, sigma: float, g: float = G_STD, C: float = 0.09) -> float:
    """Minimum film-boiling heat flux on a large flat heater [W/m^2] (Zuber; C = 0.09 from Berenson 1961).

    q_min = C rho_v h_fg [sigma g (rho_l - rho_v) / (rho_l + rho_v)^2]^(1/4).
    """
    return float(C * rho_v * h_fg * (sigma * g * (rho_l - rho_v) / (rho_l + rho_v) ** 2) ** 0.25)


def bromley_film_boiling(D: float, dT_e: float, rho_l: float, rho_v: float, k_v: float, mu_v: float, cp_v: float,
                         h_fg: float, g: float = G_STD) -> float:
    """Bromley (1950) convective coefficient of film boiling on a horizontal cylinder [W/(m^2 K)].

    h = 0.62 [g rho_v (rho_l - rho_v) h_fg' k_v^3 / (mu_v D dT_e)]^(1/4), h_fg' = h_fg + 0.4 c_p,v dT_e,
    with vapour properties at the film temperature.  Radiation across the vapour film adds
    roughly 3/4 of its own coefficient (Bromley's approximation), left to the caller.
    """
    h_fgp = h_fg + 0.4 * cp_v * dT_e
    return float(0.62 * (g * rho_v * (rho_l - rho_v) * h_fgp * k_v**3 / (mu_v * D * dT_e)) ** 0.25)

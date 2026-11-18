"""Natural convection: Ostrach's similarity solution for a vertical plate by shooting, and
correlations for vertical plates, horizontal plates and horizontal cylinders."""

import numpy as np


def rayleigh_number(g: float, beta: float, dT: float, L_c: float, nu: float, alpha: float) -> float:
    """Dimensionless Rayleigh number Ra_L = g * beta * |dT| * L_c^3 / (nu * alpha)."""
    if L_c <= 0.0 or nu <= 0.0 or alpha <= 0.0 or beta <= 0.0:
        raise ValueError("Invalid parameters for Rayleigh number.")
    return float(g * beta * abs(dT) * (L_c ** 3) / (nu * alpha))


def nu_vertical_plate_churchill_chu(Ra_L: float, Pr: float) -> float:
    """Churchill and Chu (1975) correlation for natural convection on an isothermal vertical plate across all Ra_L."""
    if Ra_L < 0.0 or Pr <= 0.0:
        raise ValueError("Require Ra_L >= 0 and Pr > 0.")
    denom = (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    return float((0.825 + 0.387 * (Ra_L ** (1.0 / 6.0)) / denom) ** 2)


def nu_horizontal_cylinder_churchill_chu(Ra_D: float, Pr: float) -> float:
    """Churchill and Chu (1975) correlation for natural convection from an isothermal horizontal cylinder."""
    if Ra_D < 0.0 or Pr <= 0.0:
        raise ValueError("Require Ra_D >= 0 and Pr > 0.")
    denom = (1.0 + (0.559 / Pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)
    return float((0.60 + 0.387 * (Ra_D ** (1.0 / 6.0)) / denom) ** 2)


def nu_vertical_plate_churchill_chu_laminar(Ra_L: float, Pr: float) -> float:
    """Churchill and Chu (1975) mean Nusselt number for a laminar isothermal vertical plate (Ra_L < 1e9).

    Nu_L = 0.68 + 0.670 Ra_L^(1/4) / [1 + (0.492/Pr)^(9/16)]^(4/9).
    """
    if Ra_L < 0.0 or Pr <= 0.0:
        raise ValueError("Require Ra_L >= 0 and Pr > 0.")
    return float(0.68 + 0.670 * Ra_L ** 0.25 / (1.0 + (0.492 / Pr) ** (9.0 / 16.0)) ** (4.0 / 9.0))


def nu_horizontal_plate_up_lloyd_moran(Ra_L: float) -> float:
    """Lloyd and Moran (1974) mean Nusselt number for the upper face of a hot horizontal plate.

    The length is L = A/P (area over perimeter); Nu_L = 0.54 Ra_L^(1/4) for Ra_L below 8e6 and
    0.15 Ra_L^(1/3) above (measured over 2.2e4 < Ra_L < 1.6e9).
    """
    if Ra_L <= 0.0:
        raise ValueError("Require Ra_L > 0.")
    return float(0.54 * Ra_L ** 0.25 if Ra_L < 8e6 else 0.15 * Ra_L ** (1.0 / 3.0))


def _free_convection_rk4(s, Pr: float, eta_max: float, n_steps: int, keep: bool = False):
    """Integrate the free-convection similarity equations from the wall for several starts at once.

    ``s`` is an (m, 2) array of trial values [F''(0), -theta'(0)].  Returns the state at
    ``eta_max`` (5 x m), or the whole path for the first start if ``keep``.
    """
    s = np.atleast_2d(s)
    y = np.zeros((5, s.shape[0]))
    y[2], y[3], y[4] = s[:, 0], 1.0, -s[:, 1]
    h = eta_max / n_steps
    path = [y[:, 0].copy()] if keep else None

    def rhs(y):
        F, Fp, Fpp, th, thp = y
        return np.array([Fp, Fpp, -3.0 * F * Fpp + 2.0 * Fp**2 - th, thp, -3.0 * Pr * F * thp])

    for _ in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if keep:
            path.append(y[:, 0].copy())
    return np.array(path).T if keep else y


def free_convection_similarity(Pr: float, eta_max: float | None = None, n_per_unit: int | None = None) -> dict:
    """Laminar free convection on an isothermal vertical plate: Ostrach's similarity solution.

    With eta = (y/x)(Gr_x/4)^(1/4) and theta = (T - T_inf)/(T_w - T_inf), the stream function
    psi = 4 nu (Gr_x/4)^(1/4) F(eta) satisfies

        F''' + 3 F F'' - 2 F'^2 + theta = 0,   theta'' + 3 Pr F theta' = 0,

    with F = F' = 0 and theta = 1 at the wall, F' = theta = 0 far away.  The two unknown wall
    values F''(0) and -theta'(0) are found by shooting (RK4 and Newton's method), lengthening
    the domain in stages so that the growing solutions of the equations stay small.  The
    default domain and step suit 0.1 <= Pr <= 100.  Returns F''(0), G = -theta'(0) (so that
    Nu_x = G (Gr_x/4)^(1/4)) and the profiles.
    """
    if Pr <= 0.0:
        raise ValueError("Pr must be positive.")
    if eta_max is None:
        eta_max = 12.0 * Pr ** -0.5 if Pr < 1.0 else 12.0 * Pr ** 0.1
    if n_per_unit is None:
        n_per_unit = int(100 * max(1.0, Pr ** 0.25))
    s = np.array([0.68 * (Pr / 0.72) ** -0.2,
                  0.75 * np.sqrt(2.0 * Pr) / (0.609 + 1.221 * np.sqrt(Pr) + 1.238 * Pr) ** 0.25])
    for E in np.append(np.arange(min(4.0, eta_max), eta_max, 1.0), eta_max):
        n = int(n_per_unit * E)
        for _ in range(40):
            d = 1e-7 * np.maximum(np.abs(s), 1e-3)
            y = _free_convection_rk4(np.array([s, s + [d[0], 0.0], s + [0.0, d[1]]]), Pr, E, n)
            r = np.array([y[1], y[3]])                      # F'(E) and theta(E), for the three starts
            J = np.column_stack(((r[:, 1] - r[:, 0]) / d[0], (r[:, 2] - r[:, 0]) / d[1]))
            ds = np.linalg.solve(J, -r[:, 0])
            s = s + ds
            if np.all(np.abs(ds) < 1e-13 * np.maximum(1.0, np.abs(s))):
                break
    if not np.all(np.isfinite(s)):
        raise RuntimeError("shooting failed; try a shorter eta_max")
    n = int(n_per_unit * eta_max)
    path = _free_convection_rk4(s, Pr, eta_max, n, keep=True)
    return {"fpp0": float(s[0]), "G": float(s[1]), "eta": np.linspace(0.0, eta_max, n + 1),
            "F": path[0], "Fp": path[1], "theta": path[3]}

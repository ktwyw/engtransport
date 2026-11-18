"""External forced convection: the Blasius similarity solution by shooting, exact laminar heat
transfer for any Prandtl number, and correlations for flat plates, cylinders and spheres."""

import numpy as np


def solve_blasius(eta_max: float = 10.0, n_steps: int = 2000) -> dict:
    """
    Solve the Blasius boundary-layer ODE 2 * d3f/deta3 + f * d2f/deta2 = 0 with f(0)=0, f'(0)=0, f'(inf)=1
    from scratch using RK4 integration coupled with secant shooting on s = f''(0).
    """
    eta = np.linspace(0.0, eta_max, n_steps + 1)
    h = eta[1] - eta[0]

    def integrate_ivp(fpp0: float) -> np.ndarray:
        y = np.zeros((n_steps + 1, 3), dtype=float)
        y[0] = [0.0, 0.0, fpp0]

        def rhs(state):
            f_val, fp_val, fpp_val = state
            return np.array([fp_val, fpp_val, -0.5 * f_val * fpp_val])

        for i in range(n_steps):
            k1 = rhs(y[i])
            k2 = rhs(y[i] + 0.5 * h * k1)
            k3 = rhs(y[i] + 0.5 * h * k2)
            k4 = rhs(y[i] + h * k3)
            y[i + 1] = y[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return y

    s0, s1 = 0.30, 0.35
    r0 = integrate_ivp(s0)[-1, 1] - 1.0
    r1 = integrate_ivp(s1)[-1, 1] - 1.0
    for _ in range(25):
        if abs(r1) < 1e-13:
            break
        s_new = s1 - r1 * (s1 - s0) / (r1 - r0)
        s0, r0 = s1, r1
        s1 = s_new
        r1 = integrate_ivp(s1)[-1, 1] - 1.0

    sol = integrate_ivp(s1)
    return {"eta": eta, "f": sol[:, 0], "fp": sol[:, 1], "fpp": sol[:, 2], "fpp0": float(s1)}


def nu_flat_plate_laminar(Re_x: float, Pr: float, average: bool = True) -> float:
    """Pohlhausen laminar flat-plate Nusselt number (Pr >= 0.6): Nu_x = 0.332 Re_x^(1/2) Pr^(1/3), Nu_bar = 2 * Nu_L."""
    if Re_x <= 0.0 or Pr <= 0.0:
        raise ValueError("Re_x and Pr must be positive.")
    nu_loc = 0.332057 * (Re_x ** 0.5) * (Pr ** (1.0 / 3.0))
    return float(2.0 * nu_loc if average else nu_loc)


def nu_cylinder_churchill_bernstein(Re_D: float, Pr: float) -> float:
    """Churchill-Bernstein (1977) average Nusselt number for a circular cylinder in crossflow (Re_D * Pr >= 0.2)."""
    if Re_D <= 0.0 or Pr <= 0.0:
        raise ValueError("Re_D and Pr must be positive.")
    term1 = 0.62 * (Re_D ** 0.5) * (Pr ** (1.0 / 3.0)) / ((1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25)
    term2 = (1.0 + (Re_D / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    return float(0.3 + term1 * term2)


def nu_sphere_whitaker(Re_D: float, Pr: float, mu_ratio: float = 1.0) -> float:
    """Whitaker (1972) correlation for flow past a sphere: Nu_D = 2 + (0.4 Re^(1/2) + 0.06 Re^(2/3)) Pr^0.4 (mu/mu_s)^(1/4)."""
    if Re_D < 0.0 or Pr <= 0.0 or mu_ratio <= 0.0:
        raise ValueError("Invalid sphere convection parameters.")
    return float(2.0 + (0.4 * (Re_D ** 0.5) + 0.06 * (Re_D ** (2.0 / 3.0))) * (Pr ** 0.4) * (mu_ratio ** 0.25))


def pohlhausen_heat_transfer(Pr: float, eta_max: float = 10.0, n_steps: int = 2000, eta_far: float = 400.0) -> float:
    """Exact laminar flat-plate heat transfer, Nu_x / Re_x^(1/2), for any Prandtl number.

    The energy equation of the similarity solution, theta'' + (Pr/2) f theta' = 0 with
    theta(0) = 0 and theta(inf) = 1, is linear in theta once the Blasius f is known, and
    integrates to theta'(0) = 1 / int_0^inf exp(-(Pr/2) int_0^eta f) d eta.  Beyond
    ``eta_max`` the Blasius function is continued with its asymptote f = eta - 1.7208.
    """
    if Pr <= 0.0:
        raise ValueError("Pr must be positive.")
    bl = solve_blasius(eta_max=eta_max, n_steps=n_steps)
    eta, f = bl["eta"], bl["f"]
    beta = f[-1] - eta[-1]                                    # displacement constant, -1.7208
    tail = np.linspace(eta[-1], max(eta_far, eta[-1]), 40001)[1:]
    eta_all = np.concatenate((eta, tail))
    f_all = np.concatenate((f, tail + beta))
    F = np.concatenate(([0.0], np.cumsum(0.5 * (f_all[1:] + f_all[:-1]) * np.diff(eta_all))))
    g = np.exp(-0.5 * Pr * F)
    integral = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(eta_all)))
    return 1.0 / integral

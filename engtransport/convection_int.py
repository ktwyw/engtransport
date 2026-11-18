"""Internal forced convection in round tubes: the Graetz series, fully developed power-law
fluids, a laminar entry correlation and Gnielinski's turbulent correlation."""

import numpy as np

NU_LAMINAR_CONST_Q = 48.0 / 11.0  # 4.36363636...
NU_LAMINAR_CONST_T = 3.656793


def nu_laminar_power_law_const_q(n: float) -> float:
    """
    Exact analytical fully-developed laminar Nusselt number in a circular pipe with uniform wall heat flux q_s''
    for an Ostwald-de Waele Power-Law fluid (tau = K * gamma_dot^n):
      Nu(n) = 8 * (5n + 1) * (3n + 1) / (31*n^2 + 12*n + 1)
    Limits:
      n = 1 (Newtonian) -> 8 * 6 * 4 / 44 = 48 / 11 = 4.363636...
      n -> 0 (Plug flow) -> 8.0
    """
    if n <= 0.0:
        raise ValueError("Flow behavior index n must be positive.")
    return float(8.0 * (5.0 * n + 1.0) * (3.0 * n + 1.0) / (31.0 * (n ** 2) + 12.0 * n + 1.0))


def nu_pipe_hausen_entry(Re_D: float, Pr: float, D_over_L: float) -> float:
    """Hausen (1943) average Nusselt number for thermal entry region in laminar pipe flow at constant wall temperature."""
    gz = Re_D * Pr * D_over_L
    return float(3.66 + (0.0668 * gz) / (1.0 + 0.04 * (gz ** (2.0 / 3.0))))


def nu_pipe_gnielinski(Re_D: float, Pr: float) -> float:
    """
    Gnielinski (1976) correlation for turbulent flow in smooth circular tubes (3000 <= Re_D <= 5e6, 0.5 <= Pr <= 2000):
      f = (0.790 * ln(Re_D) - 1.64)^(-2)
      Nu_D = ((f/8) * (Re_D - 1000) * Pr) / (1 + 12.7 * (f/8)^(1/2) * (Pr^(2/3) - 1))
    """
    if Re_D < 2300.0 or Pr <= 0.0:
        raise ValueError("Gnielinski correlation requires Re_D >= 2300 and Pr > 0.")
    f = (0.790 * np.log(Re_D) - 1.64) ** -2
    f8 = f / 8.0
    return float((f8 * (Re_D - 1000.0) * Pr) / (1.0 + 12.7 * np.sqrt(f8) * ((Pr ** (2.0 / 3.0)) - 1.0)))


def _graetz_shoot(lams: np.ndarray, n_steps: int):
    """Integrate (1/r)(r R')' + lam^2 (1 - r^2) R = 0 from r = 0 (R = 1, R' = 0) to r = 1 for many lam at once."""
    lams = np.asarray(lams, dtype=float)
    h = 1.0 / n_steps
    R = np.ones_like(lams)
    dR = np.zeros_like(lams)
    I_uR = np.zeros_like(lams)        # int (1 - r^2) R r dr
    I_uRR = np.zeros_like(lams)       # int (1 - r^2) R^2 r dr

    def rhs(r, R, dR):
        if r == 0.0:
            return dR, -0.5 * lams**2 * R        # R'/r -> R'' at the axis
        return dR, -dR / r - lams**2 * (1.0 - r * r) * R

    w = lambda r, R: (1.0 - r * r) * r * R
    for i in range(n_steps):
        r = i * h
        k1R, k1d = rhs(r, R, dR)
        k2R, k2d = rhs(r + h / 2, R + h / 2 * k1R, dR + h / 2 * k1d)
        k3R, k3d = rhs(r + h / 2, R + h / 2 * k2R, dR + h / 2 * k2d)
        k4R, k4d = rhs(r + h, R + h * k3R, dR + h * k3d)
        R_mid = R + h / 2 * k1R + h * h / 8 * (k1d)          # Simpson weights need the midpoint value
        R_new = R + h / 6 * (k1R + 2 * k2R + 2 * k3R + k4R)
        I_uR += h / 6 * (w(r, R) + 4 * w(r + h / 2, R_mid) + w(r + h, R_new))
        I_uRR += h / 6 * (w(r, R) * R + 4 * w(r + h / 2, R_mid) * R_mid + w(r + h, R_new) * R_new)
        dR = dR + h / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)
        R = R_new
    return R, dR, I_uR, I_uRR


def graetz_series(n_modes: int = 12, n_steps: int = 2000) -> dict:
    """Graetz solution for laminar, fully developed (parabolic) flow in a tube whose wall is
    held at a constant temperature from x = 0, with a uniform inlet temperature.

    With zeta = (x/D)/(Re Pr) and theta = (T - T_w)/(T_in - T_w),
    theta(zeta, r) = sum C_n R_n(r/R) exp(-2 lam_n^2 zeta), where R_n solves
    (1/r)(r R')' + lam^2 (1 - r^2) R = 0, R'(0) = 0, R(1) = 0.  The eigenvalues are found by
    shooting (fourth-order Runge-Kutta, all trial values at once) and bisection.

    Returns a dict with ``lam``, ``C``, ``dR1`` (R_n'(1)), ``M`` (the mean-temperature weights)
    and functions ``theta_m(zeta)``, ``Nu_x(zeta)`` and ``Nu_mean(zeta)``.
    """
    scan = np.linspace(0.5, 4.0 * n_modes + 4.0, 40 * n_modes + 40)
    R1 = _graetz_shoot(scan, n_steps // 4)[0]
    brackets = [(scan[i], scan[i + 1]) for i in range(scan.size - 1) if R1[i] * R1[i + 1] < 0][:n_modes]
    lo = np.array([b[0] for b in brackets])
    hi = np.array([b[1] for b in brackets])
    f_lo = _graetz_shoot(lo, n_steps)[0]
    for _ in range(45):                                       # bisection on all roots at once
        mid = 0.5 * (lo + hi)
        f_mid = _graetz_shoot(mid, n_steps)[0]
        left = f_lo * f_mid <= 0
        hi = np.where(left, mid, hi)
        lo = np.where(left, lo, mid)
        f_lo = np.where(left, f_lo, f_mid)
    lam = 0.5 * (lo + hi)
    _, dR1, I_uR, I_uRR = _graetz_shoot(lam, n_steps)
    C = I_uR / I_uRR
    M = 4.0 * C * I_uR                                         # contribution of each mode to theta_m

    def theta_m(zeta):
        z = np.asarray(zeta, dtype=float)[..., None]
        return np.sum(M * np.exp(-2.0 * lam**2 * z), axis=-1)

    def Nu_x(zeta):
        z = np.asarray(zeta, dtype=float)[..., None]
        return -2.0 * np.sum(C * dR1 * np.exp(-2.0 * lam**2 * z), axis=-1) / theta_m(zeta)

    def Nu_mean(zeta):
        return -np.log(theta_m(zeta)) / (4.0 * np.asarray(zeta, dtype=float))

    return {"lam": lam, "C": C, "dR1": dR1, "M": M, "theta_m": theta_m, "Nu_x": Nu_x, "Nu_mean": Nu_mean}

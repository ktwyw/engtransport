"""Convective mass transfer (Chapters 16 and 17): the heat-mass analogy, blowing and the
Spalding number, evaporating and falling drops, moist air, wet surfaces and wetted pads."""

import numpy as np


def chilton_colburn_mass_transfer_coeff(
    h_heat: float, rho: float, cp: float, Pr: float, Sc: float
) -> float:
    """
    Compute convective mass transfer coefficient k_c [m/s] from heat transfer coefficient h_heat [W/(m^2*K)]
    via the Chilton-Colburn analogy j_H = j_D:
      k_c = (h_heat / (rho * cp)) * (Pr / Sc)^(2/3) = (h_heat / (rho * cp)) * Le^(-2/3)
    """
    if h_heat < 0.0 or rho <= 0.0 or cp <= 0.0 or Pr <= 0.0 or Sc <= 0.0:
        raise ValueError("All Chilton-Colburn inputs must be positive.")
    lewis = Sc / Pr
    return float((h_heat / (rho * cp)) * (lewis ** (-2.0 / 3.0)))


def stefan_blowing_factor(B_M: float) -> float:
    """High-flux mass transfer blowing correction factor ln(1 + B_M) / B_M where B_M = (x_As - x_Ainf)/(1 - x_As)."""
    if B_M <= -1.0:
        raise ValueError("Spalding mass transfer number B_M must exceed -1.")
    if abs(B_M) < 1e-8:
        return float(1.0 - 0.5 * B_M + (B_M ** 2) / 3.0)
    return float(np.log1p(B_M) / B_M)


def droplet_evaporation_time(
    d0: float, D_AB: float, rho_liquid: float, M_vapor: float, P_sat_s: float, P_inf_vapor: float, R_gas: float = 8.314462618, T_film: float = 300.0, Sh: float = 2.0
) -> dict:
    """
    Quasi-steady droplet evaporation d^2-law: d(t)^2 = d0^2 - K_evap * t
    For a quiescent or creeping-flow spherical droplet (Sh = 2 + 0.6 Re^(1/2) Sc^(1/3)):
      t_lifetime = (rho_liquid * R_gas * T_film * d0^2) / (2 * Sh * D_AB * M_vapor * (P_sat_s - P_inf_vapor))
    """
    dP = P_sat_s - P_inf_vapor
    if d0 <= 0.0 or D_AB <= 0.0 or rho_liquid <= 0.0 or dP <= 0.0 or Sh <= 0.0:
        raise ValueError("Droplet evaporation requires d0 > 0, D_AB > 0, and P_sat_s > P_inf_vapor.")
    K_evap = (2.0 * Sh * D_AB * M_vapor * dP) / (rho_liquid * R_gas * T_film)
    t_life = (d0 ** 2) / K_evap
    return {"K_evap": float(K_evap), "t_lifetime": float(t_life)}


# ---------------------------------------------------------------------------
# Chapter 16: coefficients, blowing, drops
# ---------------------------------------------------------------------------

M_AIR = 0.02897          # molar mass of dry air [kg/mol]


def vapour_mass_fraction(p_v, M_v: float, P: float = 101325.0, M_gas: float = M_AIR):
    """Mass fraction of a vapour of molar mass ``M_v`` at partial pressure ``p_v`` in a gas
    of molar mass ``M_gas`` at total pressure ``P`` (ideal-gas mixture)."""
    x = np.asarray(p_v, dtype=float) / P
    m = x * M_v / (x * M_v + (1.0 - x) * M_gas)
    return float(m) if np.ndim(m) == 0 else m


def spalding_number(m_s, m_e):
    """Spalding mass-transfer number B = (m_s - m_e) / (1 - m_s) for a single species
    crossing the surface; m_s at the surface, m_e in the free stream."""
    m_s, m_e = np.asarray(m_s, dtype=float), np.asarray(m_e, dtype=float)
    if np.any(m_s >= 1.0):
        raise ValueError("the surface mass fraction must be below 1.")
    B = (m_s - m_e) / (1.0 - m_s)
    return float(B) if np.ndim(B) == 0 else B


def film_blowing_factor(B):
    """Stagnant-film blowing factor ln(1 + B) / B (vectorised; 1 - B/2 + B^2/3 near B = 0)."""
    B = np.asarray(B, dtype=float)
    if np.any(B <= -1.0):
        raise ValueError("B must exceed -1.")
    small = np.abs(B) < 1e-6
    Bs = np.where(small, 1.0, B)
    out = np.where(small, 1.0 - 0.5 * B + B * B / 3.0, np.log1p(Bs) / Bs)
    return float(out) if np.ndim(out) == 0 else out


def sherwood_flat_plate(Re, Sc, average: bool = True):
    """Laminar flat plate with uniform surface concentration and low mass-transfer rate:
    Sh_x = 0.332 Re_x^(1/2) Sc^(1/3); the average over length L is twice the local value at L."""
    Re, Sc = np.asarray(Re, dtype=float), np.asarray(Sc, dtype=float)
    sh = 0.332057 * np.sqrt(Re) * np.cbrt(Sc) * (2.0 if average else 1.0)
    return float(sh) if np.ndim(sh) == 0 else sh


def sherwood_sphere_ranz_marshall(Re, Sc):
    """Ranz and Marshall (1952) for an evaporating drop: Sh = 2 + 0.6 Re^(1/2) Sc^(1/3)."""
    Re, Sc = np.asarray(Re, dtype=float), np.asarray(Sc, dtype=float)
    sh = 2.0 + 0.6 * np.sqrt(Re) * np.cbrt(Sc)
    return float(sh) if np.ndim(sh) == 0 else sh


def mass_conductance_from_heat(h, cp: float, Pr: float, Sc: float):
    """Chilton-Colburn analogy j_D = j_H: mass conductance g* = rho k_c [kg/(m^2 s)] from the
    heat transfer coefficient h [W/(m^2 K)]: g* = (h / cp) (Pr / Sc)^(2/3)."""
    return h / cp * (Pr / Sc) ** (2.0 / 3.0)


def drag_coefficient_sphere(Re):
    """Drag coefficient of a rigid sphere, Brown and Lawler (2003), Re < 2e5:
    C_D = 24/Re (1 + 0.150 Re^0.681) + 0.407 / (1 + 8710/Re)."""
    Re = np.asarray(Re, dtype=float)
    cd = 24.0 / Re * (1.0 + 0.150 * Re ** 0.681) + 0.407 / (1.0 + 8710.0 / Re)
    return float(cd) if np.ndim(cd) == 0 else cd


def terminal_velocity(d, rho_p: float, rho_g: float, mu: float, g: float = 9.81):
    """Terminal settling velocity [m/s] of a sphere of diameter ``d`` (vectorised), from
    C_D(Re) Re^2 = (4/3) Ar with Ar = d^3 rho_g (rho_p - rho_g) g / mu^2, solved for Re by
    bisection in log Re (Brown-Lawler drag).  Returns (u_t, Re)."""
    d = np.atleast_1d(np.asarray(d, dtype=float))
    target = 4.0 / 3.0 * d ** 3 * rho_g * (rho_p - rho_g) * g / mu ** 2
    lo, hi = np.full(d.shape, -12.0), np.full(d.shape, 6.0)       # log10 Re
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        Re = 10.0 ** mid
        too_big = drag_coefficient_sphere(Re) * Re ** 2 > target
        hi = np.where(too_big, mid, hi)
        lo = np.where(too_big, lo, mid)
    Re = 10.0 ** (0.5 * (lo + hi))
    u = Re * mu / (rho_g * d)
    if u.size == 1:
        return float(u[0]), float(Re[0])
    return u, Re


def d2_law_rate(d, rho_g: float, D: float, rho_l: float, B: float, Sh=2.0):
    """Rate K = -d(d^2)/dt [m^2/s] of a drop evaporating quasi-steadily:
    K = 4 Sh rho_g D ln(1 + B) / rho_l (Sh = 2 for a drop at rest)."""
    return 4.0 * np.asarray(Sh, dtype=float) * rho_g * D * np.log1p(B) / rho_l


def falling_drop(d0: float, rho_l: float, rho_g: float, mu: float, D: float, B: float, Sc: float,
                 d_end: float | None = None, n: int = 4000, g: float = 9.81):
    """A drop falling at its terminal velocity while it evaporates (quasi-steady).

    Along the way K(d) = 4 Sh(Re(d)) rho_g D ln(1+B) / rho_l with the Ranz-Marshall Sh, so
    dt = 2 d dd / K(d) and dz = u_t(d) dt.  Integrated from ``d_end`` (default d0/1000) to d0
    on a logarithmic grid with the trapezoidal rule.  Returns a dict with ``d`` (decreasing),
    the time ``t`` and fall ``z`` to reach each d, and ``t_life``, ``z_life`` (to d_end).
    """
    if d_end is None:
        d_end = 1e-3 * d0
    dd = np.geomspace(d0, d_end, n)
    u, Re = terminal_velocity(dd, rho_l, rho_g, mu, g)
    K = d2_law_rate(dd, rho_g, D, rho_l, B, sherwood_sphere_ranz_marshall(Re, Sc))
    dt_dd = 2.0 * dd / K                       # |dt/dd|
    w = -np.diff(dd)                           # positive steps
    t = np.concatenate(([0.0], np.cumsum(0.5 * (dt_dd[1:] + dt_dd[:-1]) * w)))
    dz = dt_dd * u
    z = np.concatenate(([0.0], np.cumsum(0.5 * (dz[1:] + dz[:-1]) * w)))
    return {"d": dd, "t": t, "z": z, "u": u, "Re": Re, "t_life": float(t[-1]), "z_life": float(z[-1])}


# ---------------------------------------------------------------------------
# Chapter 17: moist air, the wet surface and evaporative coolers
# ---------------------------------------------------------------------------

CP_DRY_AIR = 1006.0      # J/(kg K), dry air near room temperature
CP_VAPOUR = 1860.0       # J/(kg K), water vapour
CP_LIQUID = 4186.0       # J/(kg K), liquid water
H_FG0 = 2.501e6          # J/kg, enthalpy of vaporization at 0 C
RATIO_W = 0.621945       # molar mass of water over that of dry air


def _p_sat(T):
    from .properties import saturation_pressure_water
    return saturation_pressure_water(T)


def humidity_ratio(T, RH, P: float = 101325.0):
    """Humidity ratio W [kg water / kg dry air] of air at temperature T [K] and relative
    humidity RH (0-1): W = 0.622 p_v / (P - p_v) with p_v = RH p_sat(T)."""
    p_v = np.asarray(RH, dtype=float) * _p_sat(T)
    W = RATIO_W * p_v / (P - p_v)
    return float(W) if np.ndim(W) == 0 else W


def relative_humidity(T, W, P: float = 101325.0):
    """Relative humidity (0-1) of air at T [K] with humidity ratio W."""
    W = np.asarray(W, dtype=float)
    p_v = P * W / (RATIO_W + W)
    rh = p_v / _p_sat(T)
    return float(rh) if np.ndim(rh) == 0 else rh


def moist_air_enthalpy(T, W):
    """Enthalpy of moist air [J per kg of dry air], ideal-gas mixture with constant heat
    capacities; zero for dry air and for liquid water at 0 C: h = c_pa t + W (h_fg0 + c_pv t)."""
    t = np.asarray(T, dtype=float) - 273.15
    h = CP_DRY_AIR * t + np.asarray(W, dtype=float) * (H_FG0 + CP_VAPOUR * t)
    return float(h) if np.ndim(h) == 0 else h


def adiabatic_saturation_temperature(T: float, W: float, P: float = 101325.0) -> float:
    """Temperature T_as [K] reached by air (T, W) saturated adiabatically by water at T_as:
    h(T, W) + (W_s - W) c_w t_as = h(T_as, W_s(T_as)), solved by bisection."""
    from .numerics import bisect

    def f(Ts):
        Ws = humidity_ratio(Ts, 1.0, P)
        return moist_air_enthalpy(T, W) + (Ws - W) * CP_LIQUID * (Ts - 273.15) - moist_air_enthalpy(Ts, Ws)

    return bisect(f, 230.0, T + 1e-9, xtol=1e-10)[0]


def wet_surface_temperature(T: float, W: float, ratio: float = 1.0, P: float = 101325.0) -> float:
    """Temperature T_s [K] of a wet surface that exchanges heat and vapour with air (T, W)
    and nothing else:  c_s r (T - T_s) = h_fg(T_s) ln(1 + B),  B = (W_s - W)/(1 + W),
    with the humid heat c_s = c_pa + W c_pv and the psychrometric ratio r = h/(g* c_s)
    (Le^(2/3) for a boundary layer, Le for a sphere in still air, 1 for Lewis's relation)."""
    from .numerics import bisect
    from .properties import saturated_water
    c_s = CP_DRY_AIR + W * CP_VAPOUR

    def f(Ts):
        Ws = humidity_ratio(Ts, 1.0, P)
        return c_s * ratio * (T - Ts) - saturated_water(Ts)["h_fg"] * np.log1p((Ws - W) / (1.0 + W))

    return bisect(f, 273.16, T + 1e-9, xtol=1e-10)[0]


def evaporative_pad(T_in: float, W_in: float, NTU, P: float = 101325.0):
    """Direct evaporative cooler (wetted pad) with Lewis's relation and the water at the
    adiabatic saturation temperature T_as of the entering air: temperature and humidity
    approach saturation together,  T_out - T_as = (T_in - T_as) e^(-NTU),
    W_s - W_out = (W_s - W_in) e^(-NTU).  Returns (T_out, W_out, effectiveness)."""
    T_as = adiabatic_saturation_temperature(T_in, W_in, P)
    W_s = humidity_ratio(T_as, 1.0, P)
    decay = np.exp(-np.asarray(NTU, dtype=float))
    T_out = T_as + (T_in - T_as) * decay
    W_out = W_s - (W_s - W_in) * decay
    return T_out, W_out, 1.0 - decay

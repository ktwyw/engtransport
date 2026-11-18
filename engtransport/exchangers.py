"""Heat exchangers: LMTD and its correction factor, effectiveness-NTU relations and rating
(Chapter 11); fouling from operating data and the cleaning interval (Chapter 18)."""

import math

import numpy as np


def effectiveness_crossflow_exact(NTU: float, Cr: float, tol: float = 1e-15) -> float:
    """Exact effectiveness of a single-pass crossflow exchanger with both fluids unmixed.

    The series of Mason (1955), equivalent to Nusselt's solution (1911, 1930):
        eps = 1/(Cr NTU) sum_{n>=0} [1 - e^{-NTU} sum_{m<=n} NTU^m/m!] [1 - e^{-Cr NTU} sum_{m<=n} (Cr NTU)^m/m!].
    The terms fall off once n exceeds NTU (1 + Cr); the sum stops when they are below ``tol``.
    """
    if NTU < 0.0 or not (0.0 <= Cr <= 1.0 + 1e-12):
        raise ValueError("Require NTU >= 0 and 0 <= Cr <= 1.")
    if NTU == 0.0:
        return 0.0
    if Cr < 1e-12:
        return float(1.0 - math.exp(-NTU))
    a, b = NTU, Cr * NTU
    ta, tb = math.exp(-a), math.exp(-b)
    Pa, Pb = ta, tb
    total, n = 0.0, 0
    while True:
        term = (1.0 - Pa) * (1.0 - Pb)
        total += term
        if term < tol and n > a + b:
            break
        n += 1
        ta *= a / n
        tb *= b / n
        Pa += ta
        Pb += tb
    return float(total / (Cr * NTU))


def lmtd(dT1: float, dT2: float) -> float:
    """Log-mean temperature difference (dT1 - dT2) / ln(dT1 / dT2), handling the equal-dT limit smoothly."""
    if dT1 <= 0.0 or dT2 <= 0.0:
        raise ValueError("Terminal temperature differences dT1 and dT2 must be strictly positive (no temperature crossover).")
    if abs(dT1 - dT2) / max(dT1, dT2) < 1e-8:
        return float(0.5 * (dT1 + dT2) - ((dT1 - dT2) ** 2) / (12.0 * (dT1 + dT2)))
    return float((dT1 - dT2) / np.log(dT1 / dT2))


def effectiveness_from_ntu(NTU: float, Cr: float, flow_type: str = "counterflow") -> float:
    """
    Heat exchanger effectiveness epsilon(NTU, Cr) for:
      - 'counterflow'
      - 'parallel'
      - 'shell_and_tube_1_2n'
      - 'crossflow_both_unmixed' (a widely used empirical approximation, within about 0.01 in eps)
      - 'crossflow_exact' (both unmixed, the exact series of ``effectiveness_crossflow_exact``)
      - 'condenser_boiler' (or whenever Cr == 0)
    """
    if NTU < 0.0 or not (0.0 <= Cr <= 1.0 + 1e-12):
        raise ValueError("Require NTU >= 0 and 0 <= Cr <= 1.")
    if NTU == 0.0:
        return 0.0
    if Cr < 1e-12 or flow_type == "condenser_boiler":
        return float(1.0 - np.exp(-NTU))

    if flow_type == "counterflow":
        if abs(1.0 - Cr) < 1e-8:
            return float(NTU / (1.0 + NTU))
        exp_term = np.exp(-NTU * (1.0 - Cr))
        return float((1.0 - exp_term) / (1.0 - Cr * exp_term))

    if flow_type == "parallel":
        return float((1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr))

    if flow_type == "shell_and_tube_1_2n":
        s = np.sqrt(1.0 + Cr * Cr)
        exp_s = np.exp(-NTU * s)
        return float(2.0 / (1.0 + Cr + s * (1.0 + exp_s) / (1.0 - exp_s)))

    if flow_type == "crossflow_both_unmixed":
        return float(1.0 - np.exp((1.0 / Cr) * (NTU ** 0.22) * (np.exp(-Cr * (NTU ** 0.78)) - 1.0)))

    if flow_type == "crossflow_exact":
        return effectiveness_crossflow_exact(NTU, Cr)

    raise ValueError(f"Unsupported flow_type='{flow_type}'.")


def ntu_from_effectiveness(eps: float, Cr: float, flow_type: str = "counterflow") -> float:
    """Inverse NTU(epsilon, Cr): closed forms for counterflow, parallel and shell_and_tube_1_2n, bisection otherwise."""
    if not (0.0 <= eps < 1.0) or not (0.0 <= Cr <= 1.0 + 1e-12):
        raise ValueError("Require 0 <= eps < 1 and 0 <= Cr <= 1.")
    if eps == 0.0:
        return 0.0
    if Cr < 1e-12 or flow_type == "condenser_boiler":
        return float(-np.log(1.0 - eps))

    if flow_type == "counterflow":
        if abs(1.0 - Cr) < 1e-8:
            return float(eps / (1.0 - eps))
        return float(np.log((1.0 - eps * Cr) / (1.0 - eps)) / (1.0 - Cr))

    if flow_type == "parallel":
        if eps * (1.0 + Cr) >= 1.0:
            raise ValueError("Target effectiveness exceeds the parallel-flow pinch limit 1 / (1 + Cr).")
        return float(-np.log(1.0 - eps * (1.0 + Cr)) / (1.0 + Cr))

    if flow_type == "shell_and_tube_1_2n":
        s = np.sqrt(1.0 + Cr * Cr)
        E = (2.0 / eps - (1.0 + Cr)) / s
        if E <= 1.0:
            raise ValueError("Target effectiveness exceeds the 1-2n shell-and-tube thermodynamic limit.")
        return float(-np.log((E - 1.0) / (E + 1.0)) / s)

    # no closed-form inverse: bisection on the monotonic eps(NTU)
    hi = 1.0
    while effectiveness_from_ntu(hi, Cr, flow_type) < eps:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError("Target effectiveness is not reachable for this arrangement.")
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if effectiveness_from_ntu(mid, Cr, flow_type) < eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * hi:
            break
    return float(0.5 * (lo + hi))


def lmtd_correction_factor(NTU: float, Cr: float, flow_type: str) -> float:
    """LMTD correction factor F = NTU_counterflow(eps) / NTU for an arrangement with effectiveness eps(NTU, Cr).

    F is the fraction of the counterflow log-mean temperature difference that the arrangement
    achieves: Q = U A F LMTD_counterflow.
    """
    eps = effectiveness_from_ntu(NTU, Cr, flow_type)
    return float(ntu_from_effectiveness(eps, Cr, "counterflow") / NTU)


def rate_exchanger(
    UA: float, C_h: float, C_c: float, T_h_in: float, T_c_in: float, flow_type: str = "counterflow"
) -> dict:
    """
    Rate a heat exchanger given UA [W/K], heat capacity rates C_h, C_c [W/K], and inlet temperatures [K or C].
    Also computes the exact LMTD to verify LMTD == epsilon-NTU consistency.
    """
    if UA < 0.0 or C_h <= 0.0 or C_c <= 0.0:
        raise ValueError("Require UA >= 0 and C_h, C_c > 0.")
    C_min = min(C_h, C_c)
    C_max = max(C_h, C_c)
    Cr = C_min / C_max
    NTU = UA / C_min
    eps = effectiveness_from_ntu(NTU, Cr, flow_type=flow_type)

    Q_max = C_min * (T_h_in - T_c_in)
    Q = eps * Q_max
    T_h_out = T_h_in - Q / C_h
    T_c_out = T_c_in + Q / C_c

    if flow_type == "counterflow":
        dT1 = T_h_in - T_c_out
        dT2 = T_h_out - T_c_in
    else:
        dT1 = T_h_in - T_c_in
        dT2 = T_h_out - T_c_out

    if dT1 > 0.0 and dT2 > 0.0 and UA > 0.0:
        dT_lm = lmtd(dT1, dT2)
    else:                    # a pinch reached to rounding, or no area: the LMTD is not defined
        dT_lm = float("nan")
    return {
        "NTU": float(NTU),
        "Cr": float(Cr),
        "effectiveness": float(eps),
        "Q": float(Q),
        "T_h_out": float(T_h_out),
        "T_c_out": float(T_c_out),
        "LMTD": float(dT_lm),
        "Q_lmtd": float(UA * dT_lm) if flow_type in ("counterflow", "parallel") else float(Q),
    }


# ---------------------------------------------------------------------------
# Chapter 18: fouling from operating data, and when to clean
# ---------------------------------------------------------------------------

def ua_from_temperatures(m_h, cp_h, T_hi, T_ho, m_c, cp_c, T_ci, T_co, flow_type: str = "counterflow") -> dict:
    """Overall conductance UA from measured flows and terminal temperatures (arrays allowed).

    The duty is the mean of the hot-side and cold-side duties; the effectiveness is
    Q / (C_min (T_hi - T_ci)), and NTU follows from the inverse effectiveness relation of the
    arrangement.  Returns a dict with ``Q``, ``imbalance`` (hot minus cold duty over Q),
    ``eps``, ``Cr``, ``NTU`` and ``UA`` (W/K)."""
    m_h, T_hi, T_ho = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (m_h, T_hi, T_ho)))
    m_c, T_ci, T_co = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (m_c, T_ci, T_co)))
    C_h, C_c = m_h * cp_h, m_c * cp_c
    Q_h, Q_c = C_h * (T_hi - T_ho), C_c * (T_co - T_ci)
    Q = 0.5 * (Q_h + Q_c)
    C_min, C_max = np.minimum(C_h, C_c), np.maximum(C_h, C_c)
    eps = Q / (C_min * (T_hi - T_ci))
    Cr = C_min / C_max
    NTU = np.array([ntu_from_effectiveness(float(e), float(c), flow_type) for e, c in zip(np.ravel(eps), np.ravel(Cr))]).reshape(np.shape(eps))
    out = {"Q": Q, "imbalance": (Q_h - Q_c) / Q, "eps": eps, "Cr": Cr, "NTU": NTU, "UA": NTU * C_min}
    return {k: (float(v) if np.ndim(v) == 0 else v) for k, v in out.items()}


def asymptotic_fouling(t, R_inf: float, tau: float):
    """Kern and Seaton's asymptotic fouling resistance R_f = R_inf (1 - exp(-t / tau))."""
    return R_inf * (1.0 - np.exp(-np.asarray(t, dtype=float) / tau))


def fit_asymptotic_fouling(t, R, tau_bounds=(1.0, 1e4)) -> dict:
    """Least-squares fit of R_inf and tau in R_f = R_inf (1 - exp(-t / tau)).

    For a given tau the best R_inf is linear, so the sum of squares is minimised over
    log(tau) alone by golden-section search (variable projection).  The covariance of
    (R_inf, tau) is s^2 (J^T J)^-1 with J the Jacobian at the optimum and s^2 the residual
    variance.  Returns ``R_inf``, ``tau``, ``cov``, ``s`` and ``sse``."""
    t, R = np.asarray(t, dtype=float), np.asarray(R, dtype=float)

    def project(log_tau):
        g = 1.0 - np.exp(-t / np.exp(log_tau))
        R_inf = float(np.dot(g, R) / np.dot(g, g))
        return R_inf, float(np.sum((R - R_inf * g) ** 2))

    a, b = math.log(tau_bounds[0]), math.log(tau_bounds[1])
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = project(c)[1], project(d)[1]
    while b - a > 1e-10:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = project(c)[1]
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = project(d)[1]
    log_tau = 0.5 * (a + b)
    R_inf, sse = project(log_tau)
    tau = math.exp(log_tau)
    e = np.exp(-t / tau)
    J = np.column_stack((1.0 - e, -R_inf * t / tau**2 * e))
    s2 = sse / max(t.size - 2, 1)
    cov = s2 * np.linalg.inv(J.T @ J)
    return {"R_inf": R_inf, "tau": tau, "cov": cov, "s": math.sqrt(s2), "sse": sse}


def cleaning_cycle_cost(t_c, loss_rate, cost_clean: float, cost_down: float, t_down: float, n: int = 400):
    """Average cost per unit time of a cleaning cycle of length t_c (vectorised in t_c).

    ``loss_rate(t)`` is the cost per unit time of the lost heat a time t after a cleaning
    (vectorised); each cycle also costs ``cost_clean`` and ``cost_down`` per unit time of
    the ``t_down`` it is out of service:
        J(t_c) = [cost_clean + cost_down t_down + int_0^t_c loss_rate dt] / (t_c + t_down).
    """
    t_c = np.atleast_1d(np.asarray(t_c, dtype=float))
    out = []
    for tc in t_c:
        s = np.linspace(0.0, tc, n + 1)
        f = loss_rate(s)
        integral = float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(s)))
        out.append((cost_clean + cost_down * t_down + integral) / (tc + t_down))
    out = np.array(out)
    return float(out[0]) if out.size == 1 else out


def optimum_cleaning_interval(loss_rate, cost_clean: float, cost_down: float, t_down: float,
                              t_bounds=(1.0, 2000.0)) -> tuple:
    """Cleaning interval that minimises ``cleaning_cycle_cost``, by golden-section search in
    log t_c.  At the optimum the loss rate equals the average cost rate.  Returns (t_c, J)."""
    f = lambda lt: cleaning_cycle_cost(math.exp(lt), loss_rate, cost_clean, cost_down, t_down)
    a, b = math.log(t_bounds[0]), math.log(t_bounds[1])
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = f(c), f(d)
    while b - a > 1e-9:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    t_opt = math.exp(0.5 * (a + b))
    return t_opt, f(math.log(t_opt))

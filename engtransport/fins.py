"""Extended surfaces (fins): one-dimensional solutions, efficiency, effectiveness and plate-fin heat sinks."""

import numpy as np


def uniform_fin_solution(
    x: np.ndarray,
    L: float,
    P: float,
    Ac: float,
    k: float,
    h: float,
    T_b: float,
    T_inf: float,
    tip_bc: str = "adiabatic",
    T_L: float | None = None,
) -> dict:
    """
    Exact 1-D analytical solution for a uniform cross-section fin of length L, perimeter P, area Ac.
    tip_bc in {'adiabatic', 'convective', 'prescribed_temp', 'infinite'}.
    """
    if L <= 0.0 or P <= 0.0 or Ac <= 0.0 or k <= 0.0 or h <= 0.0:
        raise ValueError("Fin geometry and transport parameters must be positive.")

    m = float(np.sqrt(h * P / (k * Ac)))
    M = float(np.sqrt(h * P * k * Ac) * (T_b - T_inf))
    mL = m * L

    if tip_bc == "adiabatic":
        theta_ratio = np.cosh(m * (L - x)) / np.cosh(mL)
        q_fin = M * np.tanh(mL)
        eta_f = np.tanh(mL) / mL if mL > 1e-8 else 1.0 - (mL ** 2) / 3.0
    elif tip_bc == "convective":
        ratio_h = h / (m * k)
        num = np.cosh(m * (L - x)) + ratio_h * np.sinh(m * (L - x))
        den = np.cosh(mL) + ratio_h * np.sinh(mL)
        theta_ratio = num / den
        q_fin = M * (np.sinh(mL) + ratio_h * np.cosh(mL)) / den
        A_fin = P * L + Ac
        eta_f = q_fin / (h * A_fin * (T_b - T_inf)) if abs(T_b - T_inf) > 0 else float(np.tanh(mL) / mL)
    elif tip_bc == "infinite":
        theta_ratio = np.exp(-m * x)
        q_fin = M
        eta_f = 1.0 / mL
    elif tip_bc == "prescribed_temp":
        if T_L is None:
            raise ValueError("T_L must be provided for tip_bc='prescribed_temp'.")
        theta_L_ratio = (T_L - T_inf) / (T_b - T_inf)
        theta_ratio = (theta_L_ratio * np.sinh(m * x) + np.sinh(m * (L - x))) / np.sinh(mL)
        q_fin = M * (np.cosh(mL) - theta_L_ratio) / np.sinh(mL)
        eta_f = q_fin / (h * P * L * (T_b - T_inf))
    else:
        raise ValueError(f"Unsupported tip_bc='{tip_bc}'.")

    T_x = T_inf + (T_b - T_inf) * theta_ratio
    eps_f = q_fin / (h * Ac * (T_b - T_inf)) if abs(T_b - T_inf) > 0 else float(np.sqrt(k * P / (h * Ac)))
    return {
        "x": x,
        "T": T_x,
        "theta_ratio": theta_ratio,
        "m": m,
        "mL": mL,
        "q_fin": float(q_fin),
        "efficiency": float(eta_f),
        "effectiveness": float(eps_f),
    }


def overall_surface_efficiency(eta_fin: float, N_fins: int, A_single_fin: float, A_total: float) -> float:
    """Overall surface efficiency eta_o = 1 - (N * A_f / A_t) * (1 - eta_f)."""
    if not (0.0 <= eta_fin <= 1.0) or N_fins < 0 or A_single_fin < 0.0 or A_total <= 0.0:
        raise ValueError("Invalid fin array parameters.")
    Af_total = N_fins * A_single_fin
    if Af_total > A_total + 1e-12:
        raise ValueError("Total fin area cannot exceed total surface area.")
    return float(1.0 - (Af_total / A_total) * (1.0 - eta_fin))


def plate_fin_heat_sink(N: int, H: float, t: float, length: float, width: float, k: float, h: float) -> dict:
    """Thermal resistance of a plate-fin heat sink with a uniform heat transfer coefficient.

    ``N`` straight fins of height ``H`` and thickness ``t`` stand on a base ``width`` wide,
    and run its full ``length`` (along the air flow).  Each fin is treated as a fin with an
    adiabatic tip and the corrected height H + t/2; the exposed base between the fins is
    included.  Returns a dict with the fin parameter ``mHc``, the fin efficiency ``eta_f``,
    the overall surface efficiency ``eta_o``, the total area ``A_t`` and the resistance
    ``R`` = 1 / (eta_o h A_t) in K/W.
    """
    if N < 0 or min(H, t, length, width, k, h) <= 0.0 or N * t > width:
        raise ValueError("Invalid heat sink geometry or properties.")
    Hc = H + 0.5 * t
    m = float(np.sqrt(2.0 * h / (k * t)))
    mHc = m * Hc
    eta_f = float(np.tanh(mHc) / mHc)
    A_f = 2.0 * Hc * length
    A_t = N * A_f + (width - N * t) * length
    eta_o = 1.0 - N * A_f / A_t * (1.0 - eta_f)
    return {"mHc": mHc, "eta_f": eta_f, "eta_o": eta_o, "A_t": A_t, "R": 1.0 / (eta_o * h * A_t)}

"""Thermophysical properties: air, water, steam and four solvents from formulas fitted to the
reference equations; typical solids; vapour pressure of water; gas and liquid diffusivities."""

import numpy as np

from . import _property_fits as _fits

R_AIR = 287.05  # J/(kg K), specific gas constant of dry air (M = 28.965 g/mol)

# Fuller, Schettler & Giddings (1966) atomic/molecular diffusion volumes [cm^3/mol]
FULLER_VOLUMES = {
    "H2": 6.12, "He": 2.67, "N2": 18.5, "O2": 16.3, "Air": 19.7,
    "CO": 18.0, "CO2": 26.9, "H2O": 13.1, "NH3": 20.7, "SO2": 41.8,
    "CH4": 25.14, "C2H6": 45.66, "C3H8": 66.18, "n-C4H10": 86.70,
    "C6H6": 90.68, "CH3OH": 31.25, "C2H5OH": 51.77, "Ar": 16.2,
    "C3H6O": 67.67, "C4H10O": 92.81,      # acetone, diethyl ether: atomic increments C 15.9, H 2.31, O 6.11
}

MOLECULAR_WEIGHTS = {
    "H2": 2.016, "He": 4.0026, "N2": 28.0134, "O2": 31.9988, "Air": 28.97,
    "CO": 28.010, "CO2": 44.010, "H2O": 18.015, "NH3": 17.031, "SO2": 64.066,
    "CH4": 16.043, "C2H6": 30.070, "C3H8": 44.097, "n-C4H10": 58.123,
    "C6H6": 78.114, "CH3OH": 32.042, "C2H5OH": 46.069, "Ar": 39.948,
    "C3H6O": 58.080, "C4H10O": 74.123,
}

# Typical properties of solids at 300 K (values vary by a few per cent between sources and grades): k [W/(m*K)], rho [kg/m^3], cp [J/(kg*K)]
SOLIDS_300K = {
    "copper": {"k": 401.0, "rho": 8933.0, "cp": 385.0},
    "aluminum_2024": {"k": 177.0, "rho": 2770.0, "cp": 875.0},
    "carbon_steel": {"k": 60.5, "rho": 7854.0, "cp": 434.0},
    "stainless_304": {"k": 14.9, "rho": 7900.0, "cp": 477.0},
    "glass": {"k": 1.4, "rho": 2500.0, "cp": 750.0},
    "concrete": {"k": 1.4, "rho": 2300.0, "cp": 880.0},
    "fiberglass": {"k": 0.038, "rho": 32.0, "cp": 835.0},
    "polyurethane_foam": {"k": 0.026, "rho": 70.0, "cp": 1045.0},
}


def solid_properties(name: str) -> dict:
    """Return thermal conductivity k [W/(m*K)], density rho [kg/m^3], cp [J/(kg*K)], and diffusivity alpha [m^2/s]."""
    key = name.lower().strip()
    if key not in SOLIDS_300K:
        raise ValueError(f"Unknown solid '{name}'. Available: {sorted(SOLIDS_300K)}")
    d = dict(SOLIDS_300K[key])
    d["alpha"] = d["k"] / (d["rho"] * d["cp"])
    return d


def _poly(coeffs, x):
    """Evaluate c0 + c1 x + c2 x^2 + ... (Horner's rule)."""
    result = 0.0 * x
    for c in reversed(coeffs):
        result = result * x + c
    return result


def dry_air_properties(T, P: float = 101325.0) -> dict:
    """Properties of dry air at temperature ``T`` [K] (200-1000 K) and pressure ``P`` [Pa].

    Density from the ideal-gas law; viscosity and conductivity from Sutherland-type laws
    a T^n / (T + S), and c_p from a cubic in T, all fitted to the reference equations of
    Lemmon et al. (2000) and Lemmon and Jacobsen (2004) by ``scripts/fit_properties.py``
    (largest errors 0.05 %, 0.11 % and 0.09 %).  Viscosity, conductivity and c_p are taken
    at 1 atm; at moderate pressures they barely depend on pressure.

    Returns a dict with rho [kg/m^3], mu [Pa s], nu [m^2/s], k [W/(m K)], cp [J/(kg K)],
    alpha [m^2/s], Pr and beta = 1/T [1/K].  ``T`` may be an array.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T < 200.0) or np.any(T > 1000.0) or P <= 0.0:
        raise ValueError("dry_air_properties needs 200 <= T <= 1000 K and P > 0.")
    rho = P / (R_AIR * T)
    mu = _fits.AIR_MU[0] * T ** _fits.AIR_MU[1] / (T + _fits.AIR_MU[2])
    k = _fits.AIR_K[0] * T ** _fits.AIR_K[1] / (T + _fits.AIR_K[2])
    cp = _poly(_fits.AIR_CP, T / 1000.0)
    nu = mu / rho
    alpha = k / (rho * cp)
    out = {"rho": rho, "mu": mu, "nu": nu, "k": k, "cp": cp, "alpha": alpha, "Pr": nu / alpha, "beta": 1.0 / T}
    return {key: (float(v) if np.ndim(v) == 0 else v) for key, v in out.items()}


def water_properties(T) -> dict:
    """Properties of saturated liquid water at ``T`` [K] (273.16-473.15 K, i.e. 0-200 C).

    Polynomials in (T - 273.15)/100 for c_p, rho and k, and a polynomial in 1000/T for ln mu,
    fitted to IAPWS-95 and the IAPWS viscosity and conductivity formulations by
    ``scripts/fit_properties.py`` (largest errors 0.19 %, 0.03 %, 0.22 % and 0.03 %).
    Liquid properties barely depend on pressure, so the values also hold for compressed
    water, such as that in a district-heating pipe.

    Returns a dict with rho, mu, nu, k, cp, alpha and Pr in SI units.  ``T`` may be an array.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T < 273.16 - 1e-9) or np.any(T > 473.15 + 1e-9):
        raise ValueError("water_properties needs 273.16 <= T <= 473.15 K.")
    theta = (T - 273.15) / 100.0
    cp = _poly(_fits.WATER_CP, theta)
    rho = _poly(_fits.WATER_RHO, theta)
    k = _poly(_fits.WATER_K, theta)
    mu = np.exp(_poly(_fits.WATER_MU, 1000.0 / T))
    nu = mu / rho
    alpha = k / (rho * cp)
    out = {"rho": rho, "mu": mu, "nu": nu, "k": k, "cp": cp, "alpha": alpha, "Pr": nu / alpha}
    return {key: (float(v) if np.ndim(v) == 0 else v) for key, v in out.items()}


def saturated_water(T) -> dict:
    """Saturation properties of water at ``T`` [K] (273.16-473.15 K, i.e. 0-200 C).

    Saturation pressure, latent heat and vapour density are fitted to IAPWS-95 by
    ``scripts/fit_properties.py`` (largest errors 0.005 %, 0.003 % and 0.03 %); the surface
    tension is the IAPWS formula (release R1-76, 2014), sigma = B tau^1.256 (1 - 0.625 tau)
    with tau = 1 - T/647.096 and B = 0.2358 N/m.  The liquid properties of
    ``water_properties`` are included.

    Returns a dict with p_sat [Pa], h_fg [J/kg], rho_v [kg/m^3], sigma [N/m] and the liquid
    rho, mu, nu, k, cp, alpha, Pr.  ``T`` may be an array.
    """
    out = water_properties(T)
    T = np.asarray(T, dtype=float)
    inv, theta = 1000.0 / T, (T - 273.15) / 100.0
    tau = 1.0 - T / 647.096
    sat = {"p_sat": np.exp(_poly(_fits.SAT_PSAT, inv)), "h_fg": _poly(_fits.SAT_HFG, theta),
           "rho_v": np.exp(_poly(_fits.SAT_RHOV, inv)), "sigma": 0.2358 * tau**1.256 * (1.0 - 0.625 * tau)}
    out.update({key: (float(v) if np.ndim(v) == 0 else v) for key, v in sat.items()})
    return out


def saturation_temperature_water(p: float) -> float:
    """Saturation temperature of water [K] at pressure ``p`` [Pa] (611 Pa to 1.55 MPa), by Newton's method."""
    if not 611.0 <= p <= 1.56e6:
        raise ValueError("saturation_temperature_water needs 611 Pa <= p <= 1.56 MPa.")
    T = 373.15
    for _ in range(50):
        f = _poly(_fits.SAT_PSAT, 1000.0 / T) - np.log(p)
        dc = np.polynomial.polynomial.polyder(_fits.SAT_PSAT)
        df = _poly(dc, 1000.0 / T) * (-1000.0 / T**2)
        step = f / df
        T -= step
        if abs(step) < 1e-10:
            break
    return float(T)


def steam_properties(T) -> dict:
    """Properties of superheated steam at 1 atm and ``T`` [K] (380-1300 K).

    Density from the ideal-gas law (1.1 % below IAPWS-95 at 400 K, 0.1 % at 800 K);
    c_p, viscosity and conductivity fitted to IAPWS-95 and the IAPWS transport formulations
    by ``scripts/fit_properties.py`` (largest errors 0.09 %, 0.03 % and 0.06 %).

    Returns a dict with rho, mu, nu, k, cp, alpha and Pr in SI units.  ``T`` may be an array.
    """
    T = np.asarray(T, dtype=float)
    if np.any(T < 380.0) or np.any(T > 1300.0):
        raise ValueError("steam_properties needs 380 <= T <= 1300 K.")
    rho = 101325.0 / (8.314462618 / 0.018015268 * T)
    cp = _poly(_fits.STEAM_CP, 1000.0 / T)
    k = _poly(_fits.STEAM_K, T / 1000.0)
    mu = _poly(_fits.STEAM_MU, T / 1000.0)
    nu, alpha = mu / rho, k / (rho * cp)
    out = {"rho": rho, "mu": mu, "nu": nu, "k": k, "cp": cp, "alpha": alpha, "Pr": nu / alpha}
    return {key: (float(v) if np.ndim(v) == 0 else v) for key, v in out.items()}


def fuller_diffusivity(species_a: str, species_b: str, T: float, P: float = 101325.0) -> float:
    """
    Binary gas diffusivity D_AB [m^2/s] by the Fuller-Schettler-Giddings (1966) correlation:
      D_AB = 1.0e-7 * T^1.75 * sqrt(1/M_A + 1/M_B) / (P_atm * (v_A^(1/3) + v_B^(1/3))^2)
    """
    if T <= 0.0 or P <= 0.0:
        raise ValueError("Temperature T [K] and pressure P [Pa] must be positive.")
    if species_a not in FULLER_VOLUMES or species_b not in FULLER_VOLUMES:
        raise ValueError(f"Species must be in {sorted(FULLER_VOLUMES)}")
    va, vb = FULLER_VOLUMES[species_a], FULLER_VOLUMES[species_b]
    ma, mb = MOLECULAR_WEIGHTS[species_a], MOLECULAR_WEIGHTS[species_b]
    P_atm = P / 101325.0
    m_ab = np.sqrt(1.0 / ma + 1.0 / mb)
    denom = P_atm * ((va ** (1.0 / 3.0) + vb ** (1.0 / 3.0)) ** 2)
    return float(1.0e-7 * (T ** 1.75) * m_ab / denom)


def water_vapour_diffusivity_air(T, P: float = 101325.0):
    """Diffusivity of water vapour in air [m^2/s] recommended by Massman (1998):
    D = 2.178e-5 (T/273.15)^1.81 (101325/P), T in K, P in Pa."""
    T = np.asarray(T, dtype=float)
    D = 2.178e-5 * (T / 273.15) ** 1.81 * (101325.0 / P)
    return float(D) if np.ndim(D) == 0 else D


def solvent_saturation(name: str, T):
    """Saturation pressure [Pa], saturated-liquid density [kg/m^3] and molar mass [kg/mol] of
    ethanol, methanol, acetone or diethyl ether, 250-330 K, from formulas fitted to the
    reference equations of state in CoolProp by ``scripts/fit_properties.py`` (within 0.03 %).
    """
    if name not in _fits.SOLVENTS:
        raise ValueError(f"solvent must be one of {sorted(_fits.SOLVENTS)}")
    T = np.asarray(T, dtype=float)
    if np.any(T < 249.99) or np.any(T > 330.01):
        raise ValueError("solvent_saturation needs 250 <= T <= 330 K.")
    cp_, cr, M = _fits.SOLVENTS[name]
    p = np.exp(_poly(cp_, 1000.0 / T))
    rho = _poly(cr, T - 273.15)
    out = {"p_sat": p, "rho_l": rho}
    out = {key: (float(v) if np.ndim(v) == 0 else v) for key, v in out.items()}
    out["M"] = float(M)
    return out


def wilke_chang_diffusivity(
    T: float, mu_solvent_pas: float, Vb_solute_cm3_mol: float, M_solvent_g_mol: float, phi_assoc: float = 1.0
) -> float:
    """
    Liquid-phase binary diffusivity D_AB [m^2/s] by the Wilke-Chang (1955) correlation.
      mu_solvent_pas: dynamic viscosity of solvent [Pa*s]
      Vb_solute_cm3_mol: molar volume of solute at normal boiling point [cm^3/mol]
      M_solvent_g_mol: molecular weight of solvent [g/mol]
      phi_assoc: association factor of solvent (2.6 for water, 1.9 for methanol, 1.0 for unassociated)
    """
    if T <= 0.0 or mu_solvent_pas <= 0.0 or Vb_solute_cm3_mol <= 0.0 or M_solvent_g_mol <= 0.0:
        raise ValueError("All Wilke-Chang inputs must be strictly positive.")
    mu_cp = mu_solvent_pas * 1000.0
    D_cm2_s = 7.4e-8 * np.sqrt(phi_assoc * M_solvent_g_mol) * T / (mu_cp * (Vb_solute_cm3_mol ** 0.6))
    return float(D_cm2_s * 1e-4)


def uo2_conductivity(T):
    """Thermal conductivity of uranium dioxide of 95 % theoretical density [W/(m K)], T in K.

    The recommendation of Fink, J. Nucl. Mater. 279, 1 (2000), for unirradiated fuel,
    298-3120 K: a phonon term 100/(7.5408 + 17.692 t + 3.6142 t^2) and an ambipolar term
    (6400/t^(5/2)) exp(-16.35/t), with t = T/1000.
    """
    t = np.asarray(T, dtype=float) / 1000.0
    k = 100.0 / (7.5408 + 17.692 * t + 3.6142 * t ** 2) + 6400.0 / t ** 2.5 * np.exp(-16.35 / t)
    return float(k) if np.ndim(k) == 0 else k


def saturation_pressure_water(T):
    """Saturation pressure of water vapour over liquid water [Pa], T in K (about -40 to 50 C).

    The Magnus form with the coefficients of Alduchov and Eskridge, J. Appl. Meteor. 35, 601
    (1996): p_s = 610.94 exp(17.625 t / (t + 243.04)), t in C.
    """
    t = np.asarray(T, dtype=float) - 273.15
    p = 610.94 * np.exp(17.625 * t / (t + 243.04))
    return float(p) if np.ndim(p) == 0 else p


def saturation_pressure_ice(T):
    """Saturation pressure of water vapour over ice [Pa], T in K (about -80 to 0 C).

    The Magnus form over ice of Alduchov and Eskridge (1996):
    p_s = 611.21 exp(22.587 t / (t + 273.86)), t in C.
    """
    t = np.asarray(T, dtype=float) - 273.15
    p = 611.21 * np.exp(22.587 * t / (t + 273.86))
    return float(p) if np.ndim(p) == 0 else p


def dew_point(T, relative_humidity):
    """Dew-point temperature [K] of air at temperature T [K] and relative humidity (0-1),
    from the inverse of `saturation_pressure_water`."""
    g = np.log(relative_humidity * saturation_pressure_water(T) / 610.94)
    t_d = 243.04 * g / (17.625 - g)
    return t_d + 273.15

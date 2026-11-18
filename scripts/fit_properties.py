"""Fit the property formulas of `engtransport.properties` to reference equations.

The reference values come from CoolProp, which implements
  - dry air: the equation of state of Lemmon, Jacobsen, Penoncello and Friend (2000)
    and the viscosity and thermal conductivity of Lemmon and Jacobsen (2004);
  - water and steam: IAPWS-95 (Wagner and Pruss 2002), with the IAPWS 2008 viscosity
    and IAPWS 2011 thermal conductivity formulations;
  - ethanol: the equation of state of Schroeder, Penoncello and Schroeder (2014); methanol,
    acetone and diethyl ether: the equations of state that CoolProp implements for them;
  - moist air: CoolProp's real-gas humid-air model (after Herrmann, Kretzschmar and Gatley
    2009, ASHRAE RP-1485), tabulated (not fitted) as a reference for the checks of Chapter 17.

Run once (`python scripts/fit_properties.py`); the script writes
`engtransport/_property_fits.py`, so that no coefficient is typed by hand,
and prints the largest relative error of every fit.  CoolProp and SciPy are
needed only here, not at run time.
"""

from __future__ import annotations

from pathlib import Path

import CoolProp
import CoolProp.CoolProp as CP
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
P_ATM = 101325.0


def fit_air():
    T = np.linspace(200.0, 1000.0, 161)
    prop = {key: np.array([CP.PropsSI(key, "T", t, "P", P_ATM, "Air") for t in T]) for key in ("C", "V", "L")}
    cp = np.polyfit(T / 1000.0, prop["C"], 3)[::-1]
    err_cp = np.max(np.abs(np.polynomial.polynomial.polyval(T / 1000.0, cp) / prop["C"] - 1.0))

    def form(p, t):                       # a T^n / (T + S): Sutherland's law with a free exponent
        return p[0] * t ** p[1] / (t + p[2])

    out = {"cp": (cp, err_cp)}
    for key, name, start in (("V", "mu", [8.6e-7, 1.58, 76.0]), ("L", "k", [6.3e-4, 1.68, 57.0])):
        res = least_squares(lambda p: form(p, T) / prop[key] - 1.0, start, x_scale="jac")
        out[name] = (res.x, np.max(np.abs(res.fun)))
    return out


def fit_water():
    T = np.linspace(273.16, 473.15, 201)
    prop = {key: np.array([CP.PropsSI(key, "T", t, "Q", 0, "Water") for t in T]) for key in ("C", "D", "L", "V")}
    theta = (T - 273.15) / 100.0
    out = {}
    for key, name in (("C", "cp"), ("D", "rho"), ("L", "k")):
        c = np.polyfit(theta, prop[key], 4)[::-1]
        out[name] = (c, np.max(np.abs(np.polynomial.polynomial.polyval(theta, c) / prop[key] - 1.0)))
    c = np.polyfit(1000.0 / T, np.log(prop["V"]), 5)[::-1]
    out["mu"] = (c, np.max(np.abs(np.exp(np.polynomial.polynomial.polyval(1000.0 / T, c)) / prop["V"] - 1.0)))
    return out


def fit_saturation():
    """Saturation pressure, latent heat and vapour density of water, 0-200 C."""
    T = np.linspace(273.16, 473.15, 201)
    p = np.array([CP.PropsSI("P", "T", t, "Q", 0, "Water") for t in T])
    h_l = np.array([CP.PropsSI("H", "T", t, "Q", 0, "Water") for t in T])
    h_v = np.array([CP.PropsSI("H", "T", t, "Q", 1, "Water") for t in T])
    rho_v = np.array([CP.PropsSI("D", "T", t, "Q", 1, "Water") for t in T])
    theta, inv = (T - 273.15) / 100.0, 1000.0 / T
    out = {}
    c = np.polyfit(inv, np.log(p), 5)[::-1]
    out["psat"] = (c, np.max(np.abs(np.exp(np.polynomial.polynomial.polyval(inv, c)) / p - 1.0)))
    c = np.polyfit(theta, h_v - h_l, 4)[::-1]
    out["hfg"] = (c, np.max(np.abs(np.polynomial.polynomial.polyval(theta, c) / (h_v - h_l) - 1.0)))
    c = np.polyfit(inv, np.log(rho_v), 5)[::-1]
    out["rhov"] = (c, np.max(np.abs(np.exp(np.polynomial.polynomial.polyval(inv, c)) / rho_v - 1.0)))
    return out


def fit_steam():
    """Superheated steam at 1 atm, 380-1300 K."""
    T = np.linspace(380.0, 1300.0, 185)
    prop = {key: np.array([CP.PropsSI(key, "T", t, "P", P_ATM, "Water") for t in T]) for key in ("C", "V", "L")}
    out = {}
    c = np.polyfit(1000.0 / T, prop["C"], 4)[::-1]          # c_p rises steeply near saturation: fit in 1000/T
    out["cp"] = (c, np.max(np.abs(np.polynomial.polynomial.polyval(1000.0 / T, c) / prop["C"] - 1.0)))
    for key, name in (("L", "k"), ("V", "mu")):
        c = np.polyfit(T / 1000.0, prop[key], 5)[::-1]
        out[name] = (c, np.max(np.abs(np.polynomial.polynomial.polyval(T / 1000.0, c) / prop[key] - 1.0)))
    return out


SOLVENTS = {"ethanol": "Ethanol", "methanol": "Methanol", "acetone": "Acetone", "diethyl ether": "DiethylEther"}


def fit_solvents():
    """Saturation pressure (ln p in 1000/T, cubic) and saturated-liquid density (quadratic in
    T - 273.15) of four solvents, 250-330 K."""
    T = np.linspace(250.0, 330.0, 161)
    out = {}
    for key, fluid in SOLVENTS.items():
        p = np.array([CP.PropsSI("P", "T", t, "Q", 0, fluid) for t in T])
        rho = np.array([CP.PropsSI("D", "T", t, "Q", 0, fluid) for t in T])
        cp_ = np.polyfit(1000.0 / T, np.log(p), 3)[::-1]
        cr = np.polyfit(T - 273.15, rho, 2)[::-1]
        err_p = np.max(np.abs(np.exp(np.polynomial.polynomial.polyval(1000.0 / T, cp_)) / p - 1.0))
        err_r = np.max(np.abs(np.polynomial.polynomial.polyval(T - 273.15, cr) / rho - 1.0))
        out[key] = (cp_, cr, CP.PropsSI("M", fluid), max(err_p, err_r))
    return out


def humid_air_table():
    """Humidity ratio, enthalpy per kg of dry air and wet-bulb (adiabatic saturation)
    temperature of moist air at 1 atm on a grid of temperature and relative humidity."""
    from CoolProp.HumidAirProp import HAPropsSI
    rows = []
    for t in (5.0, 15.0, 25.0, 35.0, 45.0):
        for rh in (0.1, 0.3, 0.6, 0.9):
            T = t + 273.15
            rows.append((t, rh, HAPropsSI("W", "T", T, "P", P_ATM, "R", rh), HAPropsSI("H", "T", T, "P", P_ATM, "R", rh),
                         HAPropsSI("Twb", "T", T, "P", P_ATM, "R", rh) - 273.15))
    return rows


def main():
    air, water = fit_air(), fit_water()
    sat, steam = fit_saturation(), fit_steam()
    solv = fit_solvents()
    lines = [
        '"""Coefficients of the property formulas, written by scripts/fit_properties.py. Do not edit."""',
        "",
        f"COOLPROP_VERSION = {CoolProp.__version__!r}",
        "",
        "# dry air at 1 atm, 200-1000 K",
    ]
    for name, (c, err) in air.items():
        lines.append(f"AIR_{name.upper()} = {tuple(float(v) for v in c)!r}  # max relative error {err:.1e}")
    lines += ["", "# saturated liquid water, 0-200 C"]
    for name, (c, err) in water.items():
        lines.append(f"WATER_{name.upper()} = {tuple(float(v) for v in c)!r}  # max relative error {err:.1e}")
    lines += ["", "# saturated water: ln p_sat and ln rho_v in 1000/T, h_fg in (T - 273.15)/100; 0-200 C"]
    for name, (c, err) in sat.items():
        lines.append(f"SAT_{name.upper()} = {tuple(float(v) for v in c)!r}  # max relative error {err:.1e}")
    lines += ["", "# superheated steam at 1 atm, 380-1300 K: c_p in 1000/T, k and mu in T/1000"]
    for name, (c, err) in steam.items():
        lines.append(f"STEAM_{name.upper()} = {tuple(float(v) for v in c)!r}  # max relative error {err:.1e}")
    lines += ["", "# solvents, 250-330 K: (ln p_sat in 1000/T, rho_liquid in T - 273.15 C, molar mass kg/mol)",
              "SOLVENTS = {"]
    for name, (cp_, cr, M, err) in solv.items():
        lines.append(f"    {name!r}: ({tuple(float(v) for v in cp_)!r}, {tuple(float(v) for v in cr)!r}, {M!r}),"
                     f"  # max relative error {err:.1e}")
    lines.append("}")
    lines += ["", "# moist air at 1 atm (CoolProp HAPropsSI): (t [C], RH, W [kg/kg], h [J/kg dry air], t_wb [C])",
              "HUMID_AIR_REF = ("]
    for row in humid_air_table():
        lines.append("    (" + ", ".join(repr(float(v)) for v in row) + "),")
    lines.append(")")
    errors = ({f"air {k}": v[1] for k, v in air.items()} | {f"water {k}": v[1] for k, v in water.items()}
              | {f"saturation {k}": v[1] for k, v in sat.items()} | {f"steam {k}": v[1] for k, v in steam.items()}
              | {f"{k}": v[3] for k, v in solv.items()})
    lines += ["", f"MAX_RELATIVE_ERROR = {{{', '.join(f'{k!r}: {v:.1e}' for k, v in errors.items())}}}", ""]
    (ROOT / "engtransport" / "_property_fits.py").write_text("\n".join(lines))
    for k, v in errors.items():
        print(f"{k:10s} max relative error {v:.1e}")


if __name__ == "__main__":
    main()

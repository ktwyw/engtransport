"""Checks of engtransport against exact solutions, limits and reference data.

Each call of `check` records one comparison; `tests/test_reference_checks.py` turns every
record into a test.  The checks are grouped by subject and follow the chapters of the book.
"""

from __future__ import annotations

import math
import numpy as np


from engtransport import (
    conduction,
    convection_ext,
    convection_int,
    convective_mass,
    diffusion,
    exchangers,
    fins,
    natural_conv,
    phase_change,
    properties,
    reference,
    radiation,
    transient,
)

RESULTS: list[tuple] = []


def check(section: str, name: str, reference: str, value: float, expected: float, rtol: float = 0.0, atol: float = 0.0) -> None:
    ok = bool(np.isclose(value, expected, rtol=rtol, atol=atol))
    RESULTS.append((section, name, reference, float(value), float(expected), ok))


# ------------------------------------------------------------------ 1. Thermophysical properties & diffusivities
S = "Thermophysical properties and binary diffusivities"
# Reference values from CoolProp 8.0: Lemmon et al. (2000) and Lemmon and Jacobsen (2004) for air,
# IAPWS-95 with the IAPWS viscosity and conductivity formulations for saturated liquid water.
AIR_REF = [  # T [K], rho, mu, k, cp, Pr at 1 atm
    (250.0, 1.4133100353, 1.60381e-05, 0.0225644027, 1005.5419211301, 0.7147111962),
    (300.0, 1.1769955884, 1.85373e-05, 0.0263844657, 1006.3739076641, 0.7070636188),
    (400.0, 0.8823072513, 2.30554e-05, 0.0334532006, 1014.144050291, 0.6989322193),
    (600.0, 0.5880969601, 3.07687e-05, 0.0460112538, 1051.2032568664, 0.7029621396),
    (800.0, 0.4410791844, 3.737e-05, 0.0572488201, 1098.6899450174, 0.717184923),
]
WATER_REF = [  # T [K], rho, mu, k, cp of saturated liquid
    (275.15, 999.8925397203, 0.0016737043, 0.5605883573, 4213.4956638646),
    (293.15, 998.1618013432, 0.0010016267, 0.597954029, 4184.360514412),
    (323.15, 987.9962106112, 0.0005464984, 0.6405745402, 4181.5477428417),
    (373.15, 958.3490516049, 0.000281582, 0.6772105145, 4215.6736168158),
    (423.15, 917.0077392689, 0.0001826109, 0.6810160993, 4307.0798481299),
]
for T, rho_ref, mu_ref, k_ref, cp_ref, pr_ref in AIR_REF:
    air = properties.dry_air_properties(T, 101325.0)
    ref = "Lemmon et al. (2000), Lemmon & Jacobsen (2004)"
    check(S, f"dry air density at {T:.0f} K (kg/m3), ideal gas", ref, air["rho"], rho_ref, rtol=1e-3)
    check(S, f"dry air viscosity at {T:.0f} K (Pa s)", ref, air["mu"], mu_ref, rtol=1e-3)
    check(S, f"dry air conductivity at {T:.0f} K (W/(m K))", ref, air["k"], k_ref, rtol=2e-3)
    check(S, f"dry air heat capacity at {T:.0f} K (J/(kg K))", ref, air["cp"], cp_ref, rtol=1e-3)
    check(S, f"dry air Prandtl number at {T:.0f} K", ref, air["Pr"], pr_ref, rtol=3e-3)
for T, rho_ref, mu_ref, k_ref, cp_ref in WATER_REF:
    wat = properties.water_properties(T)
    ref = "IAPWS-95 and IAPWS transport formulations"
    check(S, f"liquid water density at {T - 273.15:.0f} C (kg/m3)", ref, wat["rho"], rho_ref, rtol=1e-3)
    check(S, f"liquid water viscosity at {T - 273.15:.0f} C (Pa s)", ref, wat["mu"], mu_ref, rtol=1e-3)
    check(S, f"liquid water conductivity at {T - 273.15:.0f} C (W/(m K))", ref, wat["k"], k_ref, rtol=3e-3)
    check(S, f"liquid water heat capacity at {T - 273.15:.0f} C (J/(kg K))", ref, wat["cp"], cp_ref, rtol=3e-3)

for gas, D0 in (("H2O", 2.178e-5), ("CO2", 1.381e-5)):
    for T in (253.15, 298.15, 333.15):
        check(S, f"Fuller diffusivity of {gas} in air at {T - 273.15:.0f} C vs Massman (1998)", "Massman (1998) review",
              properties.fuller_diffusivity(gas, "Air", T), D0 * (T / 273.15) ** 1.81, rtol=0.035)
check(S, "Massman water-vapour diffusivity function at 0 C", "Massman (1998)", properties.water_vapour_diffusivity_air(273.15), 2.178e-5, rtol=1e-12)
check(S, "Ethanol vapour pressure from the fit, 25 C", "Schroeder et al. (2014) via CoolProp: 7885.9 Pa", properties.solvent_saturation("ethanol", 298.15)["p_sat"], 7885.87, rtol=5e-4)

check(
    S,
    "Fuller pressure scaling: D_AB(2 atm) = 0.5 * D_AB(1 atm)",
    "kinetic theory",
    properties.fuller_diffusivity("CO2", "N2", 350.0, 202650.0) / properties.fuller_diffusivity("CO2", "N2", 350.0, 101325.0),
    0.5,
    rtol=1e-14,
)
check(
    S,
    "Wilke-Chang Stokes-Einstein T/mu scaling ratio",
    "identity",
    properties.wilke_chang_diffusivity(320.0, 0.8e-3, 74.0, 18.015, 2.6)
    / properties.wilke_chang_diffusivity(290.0, 1.0e-3, 74.0, 18.015, 2.6),
    (320.0 / 290.0) * (1.0 / 0.8),
    rtol=1e-14,
)


# ------------------------------------------------------------------ 2. Conduction and extended surfaces (fins)
S = "Steady conduction and extended surfaces (fins)"
R_big, dr_thin, k_wall = 80.0, 1e-4, 40.0
check(
    S,
    "cylindrical shell resistance approaches plane wall as dr/R -> 0",
    "exact limit",
    conduction.r_cylindrical_shell(R_big, R_big + dr_thin, 1.0, k_wall)
    / conduction.r_plane_wall(dr_thin, k_wall, 2.0 * math.pi * R_big),
    1.0,
    rtol=1e-6,
)
check(
    S,
    "spherical shell resistance approaches plane wall as dr/R -> 0",
    "exact limit",
    conduction.r_spherical_shell(R_big, R_big + dr_thin, k_wall)
    / conduction.r_plane_wall(dr_thin, k_wall, 4.0 * math.pi * (R_big ** 2)),
    1.0,
    rtol=2e-6,
)
check(S, "critical insulation radius of cylinder r_crit = k/h", "exact", conduction.critical_insulation_radius(0.15, 10.0, "cylinder"), 0.015, rtol=1e-14)
check(S, "critical insulation radius of sphere r_crit = 2k/h", "exact", conduction.critical_insulation_radius(0.15, 10.0, "sphere"), 0.030, rtol=1e-14)

x_k = np.linspace(0.0, 0.20, 2001)
q_k, T_k = conduction.kirchhoff_linear_k_wall(x_k, 0.20, 700.0, 300.0, 20.0, 0.002)
k_local = 20.0 * (1.0 + 0.002 * (T_k - 700.0))
q_local = -k_local * np.gradient(T_k, x_k, edge_order=2)
check(S, "Kirchhoff transform: steady heat flux equals k_mean * dT / L", "exact", q_k, 20.0 * (1.0 + 0.002 * (-200.0)) * 400.0 / 0.20, rtol=1e-14)
check(S, "Kirchhoff transform: max pointwise flux residual across wall", "conservation", float(np.max(np.abs(q_local[2:-2] - q_k)) / q_k), 0.0, atol=5e-5)

x2, y2, T2_fd = conduction.solve_2d_steady_rect(41, 41, 1.0, 1.0, 300.0, 300.0, 300.0, 500.0)
T2_ex = conduction.analytical_2d_rectangular_plate(x2, y2, 1.0, 1.0, 300.0, 500.0, n_terms=100)
check(S, "2-D Laplace on unit square: exact midpoint T(0.5, 0.5) by symmetry", "superposition", T2_fd[20, 20], 350.0, rtol=1e-12)
check(S, "2-D Laplace on unit square: Fourier series midpoint T(0.5, 0.5)", "Fourier series", T2_ex[20, 20], 350.0, rtol=1e-6)
check(S, "2-D Laplace on unit square: max interior FD vs Fourier difference", "cross-check", float(np.max(np.abs(T2_fd[4:-4, 4:-4] - T2_ex[4:-4, 4:-4])) / 200.0), 0.0, atol=2e-3)

xf = np.linspace(0.0, 0.05, 51)
check(
    S,
    "short fin limit (mL -> 0): adiabatic fin efficiency -> 1",
    "exact limit",
    fins.uniform_fin_solution(xf, 1e-5, 0.04, 1e-4, 200.0, 10.0, 400.0, 300.0, "adiabatic")["efficiency"],
    1.0,
    rtol=1e-8,
)
f_long = fins.uniform_fin_solution(xf, 2.5, 0.04, 1e-5, 20.0, 100.0, 400.0, 300.0, "adiabatic")
f_inf = fins.uniform_fin_solution(xf, 2.5, 0.04, 1e-5, 20.0, 100.0, 400.0, 300.0, "infinite")
check(S, "long fin limit (mL >> 1): adiabatic fin heat rate -> infinite fin M", "exact limit", f_long["q_fin"], f_inf["q_fin"], rtol=1e-8)
check(S, "long fin limit (mL >> 1): efficiency -> 1 / (mL)", "exact limit", f_long["efficiency"], 1.0 / f_long["mL"], rtol=1e-8)
_fin1d = fins.uniform_fin_solution(xf, 0.05, 2.0, 2e-3, 200.0, 40.0, 1.0, 0.0, "adiabatic")["q_fin"] / 2.0
_fin2d = reference.plate_fin_2d_series(0.05, 1e-3, 200.0, 40.0)["Q_per_K"]
check(S, "thin plate fin (Bi = 2e-4): 1-D fin theory vs exact 2-D series", "2-D series", _fin1d, _fin2d, rtol=1e-4)
_fv = reference.plate_fin_2d_fv(1.5, 0.09, 2.3, 23.0, 1.0, 0.0, nx=667, ny=40)["Q_base"]
check(S, "thick plate fin (Bi = 0.9): 2-D finite volumes vs exact 2-D series", "2-D series", _fv, reference.plate_fin_2d_series(1.5, 0.09, 2.3, 23.0)["Q_per_K"], rtol=2e-3)


# ------------------------------------------------------------------ 3. Transient conduction & Stefan phase change
S = "Transient conduction, Heisler eigenvalues, and Stefan moving boundaries"
for bi_val in [0.1, 1.0, 5.0, 20.0]:
    xi_slab = transient.slab_eigenvalues(bi_val, 5)
    xi_cyl = transient.cylinder_eigenvalues(bi_val, 5)
    xi_sph = transient.sphere_eigenvalues(bi_val, 5)
    check(S, f"slab transcendental residual max |xi tan(xi) - Bi| at Bi={bi_val}", "eigenvalue eq.", float(np.max(np.abs(xi_slab * np.tan(xi_slab) - bi_val))), 0.0, atol=1e-10)
    check(S, f"cylinder transcendental residual max |xi J1(xi)/J0(xi) - Bi| at Bi={bi_val}", "eigenvalue eq.", float(np.max(np.abs(xi_cyl * transient.bessel_j1(xi_cyl) / transient.bessel_j0(xi_cyl) - bi_val))), 0.0, atol=2e-10)
    check(S, f"sphere transcendental residual max |1 - xi cot(xi) - Bi| at Bi={bi_val}", "eigenvalue eq.", float(np.max(np.abs(1.0 - xi_sph / np.tan(xi_sph) - bi_val))), 0.0, atol=1e-10)

for geom, factor in [("slab", 1.0), ("cylinder", 2.0), ("sphere", 3.0)]:
    res_t = transient.transient_1d_series(0.0, Fo=3000.0, Bi=1e-5, geometry=geom, n_modes=12)
    check(S, f"{geom} eigenfunction series converges to lumped capacitance at Bi=1e-5", "asymptotic limit", float(res_t["theta_star"]), math.exp(-factor * 1e-5 * 3000.0), rtol=1e-5)
    check(S, f"{geom} fractional energy loss Q/Q_max at Bi=1e-5 matches 1 - exp(-Bi_L Fo)", "energy conservation", res_t["Q_over_Qmax"], 1.0 - math.exp(-factor * 1e-5 * 3000.0), rtol=1e-4)

# Reference Heisler / Carslaw & Jaeger fundamental eigenvalues xi_1 at Bi = 1.0:
check(S, "slab fundamental eigenvalue xi_1 at Bi = 1.0", "Carslaw & Jaeger", float(transient.slab_eigenvalues(1.0, 1)[0]), 0.86033359, rtol=1e-7)
check(S, "cylinder fundamental eigenvalue xi_1 at Bi = 1.0", "Carslaw & Jaeger", float(transient.cylinder_eigenvalues(1.0, 1)[0]), 1.2557837, rtol=1e-6)
check(S, "sphere fundamental eigenvalue xi_1 at Bi = 1.0", "Carslaw & Jaeger", float(transient.sphere_eigenvalues(1.0, 1)[0]), 1.5707963, rtol=1e-6)

check(S, "semi-infinite solid: surface temperature at x = 0 equals T_s", "boundary condition", float(transient.semi_infinite_const_surface_T(0.0, 10.0, 1e-5, 400.0, 300.0)["T"]), 300.0, rtol=1e-14)
check(S, "semi-infinite solid: far-field temperature at x -> inf equals T_i", "boundary condition", float(transient.semi_infinite_const_surface_T(10.0, 1.0, 1e-6, 400.0, 300.0)["T"]), 400.0, rtol=1e-14)
for ste_val in [1e-5, 1e-4]:
    lam_s = transient.stefan_melting_root(ste_val)
    check(S, f"Neumann-Stefan root at Ste={ste_val} matches quasi-steady sqrt(Ste/2)", "asymptotic limit", lam_s, math.sqrt(0.5 * ste_val), rtol=1e-4)


# ------------------------------------------------------------------ 4. External, internal & natural convection
S = "External, internal, and natural convection"
bl = convection_ext.solve_blasius(eta_max=10.0, n_steps=1600)
check(S, "Blasius laminar boundary-layer wall shear f''(0)", "Howarth (1938): 0.33205734", bl["fpp0"], 0.332057336, rtol=1e-6)
check(S, "Blasius far-field velocity f'(10) = 1", "boundary condition", float(bl["fp"][-1]), 1.0, rtol=1e-9)
check(S, "Pohlhausen average flat-plate Nusselt number is 2 * local Nu_L", "exact integral", convection_ext.nu_flat_plate_laminar(1e5, 0.71, True) / convection_ext.nu_flat_plate_laminar(1e5, 0.71, False), 2.0, rtol=1e-14)
check(S, "Whitaker sphere correlation recovers conduction limit Nu_D = 2 as Re_D -> 0", "exact limit", convection_ext.nu_sphere_whitaker(0.0, 0.71), 2.0, rtol=1e-14)
check(S, "Churchill-Bernstein cylinder correlation approaches 0.3 as Re_D -> 0", "limit", convection_ext.nu_cylinder_churchill_bernstein(1e-12, 0.71), 0.3, atol=1e-5)

check(S, "Power-Law laminar pipe Nu(n=1) recovers Newtonian 48/11 = 4.363636", "exact", convection_int.nu_laminar_power_law_const_q(1.0), 48.0 / 11.0, rtol=1e-14)
check(S, "Power-Law laminar pipe Nu(n->0) recovers plug-flow limit 8.0", "exact limit", convection_int.nu_laminar_power_law_const_q(1e-9), 8.0, rtol=1e-7)
check(S, "Power-Law laminar pipe Nu(n=0.5): exact formula vs marching finite volumes", "graetz_march", convection_int.nu_laminar_power_law_const_q(0.5), float(reference.graetz_march(2.0, n_r=100, n_x=300, n=0.5, wall="flux")["Nu_x"][-1]), rtol=2e-4)
check(S, "Hausen thermal entry correlation approaches 3.66 as L -> inf (D/L -> 0)", "asymptotic limit", convection_int.nu_pipe_hausen_entry(1500.0, 5.0, 0.0), 3.66, rtol=1e-14)
check(S, "Gnielinski turbulent pipe Nu at Re=10000, Pr=0.70", "Gnielinski (1976)", convection_int.nu_pipe_gnielinski(10000.0, 0.70), 29.8174, rtol=1e-4)
check(S, "Gnielinski turbulent pipe Nu at Re=100000, Pr=5.0", "Gnielinski (1976)", convection_int.nu_pipe_gnielinski(100000.0, 5.0), 515.6835, rtol=1e-4)

check(S, "Churchill-Chu vertical plate Nu at Ra = 0 equals 0.825^2", "limit", natural_conv.nu_vertical_plate_churchill_chu(0.0, 0.71), 0.825 ** 2, rtol=1e-14)
check(S, "Churchill-Chu horizontal cylinder Nu at Ra = 0 equals 0.60^2", "limit", natural_conv.nu_horizontal_cylinder_churchill_chu(0.0, 0.71), 0.36, rtol=1e-14)
check(S, "Churchill-Chu laminar coefficient (Pr = 0.72) vs exact similarity coefficient (4/3) G (4 Pr)^(-1/4)", "free_convection_bvp", 0.670 / (1 + (0.492 / 0.72) ** (9 / 16)) ** (4 / 9), (4 / 3) * reference.free_convection_bvp(0.72)["G"] * (4 * 0.72) ** -0.25, rtol=5e-3)
check(S, "Film condensation: exact subcooling correction, quadrature vs collocation (Ja = 0.5)", "film_condensation_bvp", phase_change.film_condensation_subcooling(0.5)["ratio"], reference.film_condensation_bvp(0.5)["ratio"], rtol=1e-10)
check(S, "Film condensation: small-Ja coefficient of h_fg'/h_fg equals 27/40", "perturbation theory", (phase_change.film_condensation_subcooling(1e-6)["ratio"] - 1) / 1e-6, 27 / 40, rtol=1e-4)
check(S, "Nusselt horizontal-tube constant vs 0.728", "Nusselt (1916)", phase_change.nusselt_tube_constant(), 0.728, rtol=1e-4)
check(S, "Zuber pool-boiling CHF for saturated water at 1 atm (MW/m2)", "Zuber (1958)", phase_change.zuber_critical_heat_flux(958.4, 0.598, 2.257e6, 0.0589) / 1e6, 1.261, rtol=2e-3)


# ------------------------------------------------------------------ 5. Heat exchangers (LMTD and epsilon-NTU)
S = "Heat exchangers (LMTD and epsilon-NTU)"
check(S, "LMTD equal-difference limit dT1 = dT2 = 40 K", "L'Hopital limit", exchangers.lmtd(40.0, 40.0), 40.0, rtol=1e-14)
check(S, "LMTD near-equal difference dT1 = 40.0001 K, dT2 = 39.9999 K", "Taylor expansion", exchangers.lmtd(40.0001, 39.9999), 40.0, rtol=1e-10)
check(S, "Crossflow both unmixed: exact series vs grid of 160 x 160 small exchangers (NTU 2, Cr 0.5)", "crossflow_cells", exchangers.effectiveness_crossflow_exact(2.0, 0.5), reference.crossflow_cells(2.0, 0.5, 160)["eps"], rtol=1e-5)
check(S, "1-2 shell and tube: formula vs collocation of the three-stream equations (NTU 2, Cr 0.5)", "shell_and_tube_1_2_bvp", exchangers.effectiveness_from_ntu(2.0, 0.5, "shell_and_tube_1_2n"), reference.shell_and_tube_1_2_bvp(2.0, 0.5)["eps"], rtol=1e-10)
check(S, "Counterflow balanced limit (Cr = 1): eps = NTU / (1 + NTU)", "exact limit", exchangers.effectiveness_from_ntu(2.5, 1.0, "counterflow"), 2.5 / 3.5, rtol=1e-12)
check(S, "Parallel-flow asymptotic pinch limit (NTU -> inf): eps -> 1 / (1 + Cr)", "second law", exchangers.effectiveness_from_ntu(50.0, 0.60, "parallel"), 1.0 / 1.60, rtol=1e-12)
check(S, "Condenser/boiler limit (Cr = 0) is identical across all flow arrangements", "identity", exchangers.effectiveness_from_ntu(1.8, 0.0, "counterflow"), exchangers.effectiveness_from_ntu(1.8, 0.0, "parallel"), rtol=1e-14)

for flow_cfg in ("counterflow", "parallel", "shell_and_tube_1_2n"):
    for cr_v in [0.0, 0.35, 0.75, 1.0]:
        if flow_cfg == "shell_and_tube_1_2n" and cr_v == 0.0:
            continue
        eps_v = exchangers.effectiveness_from_ntu(1.4, cr_v, flow_cfg)
        ntu_inv = exchangers.ntu_from_effectiveness(eps_v, cr_v, flow_cfg)
        check(S, f"NTU <-> effectiveness round-trip ({flow_cfg}, Cr={cr_v})", "inverse identity", ntu_inv, 1.4, rtol=1e-9)

for flow_cfg in ("counterflow", "parallel"):
    for ua_v in [600.0, 2400.0, 7500.0]:
        rx = exchangers.rate_exchanger(ua_v, 3200.0, 2100.0, 420.0, 300.0, flow_cfg)
        check(S, f"LMTD duty equals epsilon-NTU duty ({flow_cfg}, UA={ua_v:.0f} W/K)", "exact equivalence", rx["Q"], rx["Q_lmtd"], rtol=1e-11)


# ------------------------------------------------------------------ 6. Thermal radiation & radiosity networks
S = "Thermal radiation, view factors, and Oppenheim radiosity enclosures"
lam_int = np.geomspace(0.08, 250.0, 5000)
E_int = float(np.trapezoid(radiation.planck_spectral_emissive_power(lam_int, 1500.0), lam_int))
check(S, "Planck spectral integral int_0^inf E_{lambda,b} d_lambda equals sigma T^4", "Stefan-Boltzmann", E_int, radiation.SIGMA_SB * (1500.0 ** 4), rtol=2e-4)
for _lt in (2897.771955, 7300.0, 20000.0):
    _x = radiation.C2_PLANCK / _lt
    _Fq = 15.0 / np.pi**4 * __import__("scipy.integrate", fromlist=["quad"]).quad(lambda z: z**3 * np.exp(-z) / -np.expm1(-z), _x, np.inf, epsabs=1e-16, epsrel=1e-13)[0]
    check(S, f"Blackbody band fraction F({_lt:.0f} um K): series vs numerical quadrature", "quadrature", radiation.blackbody_band_fraction(_lt), _Fq, rtol=1e-12)
check(S, "Wien constant from x = 5(1 - exp(-x)) vs CODATA 2897.771955 um K", "CODATA 2018", radiation.wien_constant(), 2897.771955, rtol=1e-9)
check(S, "Stefan-Boltzmann constant from h, c, k vs CODATA 5.670374419e-8", "CODATA 2018", radiation.stefan_boltzmann_from_constants(), 5.670374419e-8, rtol=1e-9)
check(S, "Blackbody band fraction F_(0 -> inf) = 1", "normalization", radiation.blackbody_band_fraction(1e7), 1.0, rtol=1e-9)
check(S, "Coaxial parallel disks view factor F_12 -> 1 as separation L -> 0 (r1=r2)", "geometric limit", radiation.view_factor_coaxial_disks(1.0, 1.0, 1e-7), 1.0, rtol=1e-6)
check(S, "Coaxial parallel disks reciprocity A1 F12 = A2 F21", "reciprocity", (1.0 ** 2) * radiation.view_factor_coaxial_disks(1.0, 2.0, 1.5), (2.0 ** 2) * radiation.view_factor_coaxial_disks(2.0, 1.0, 1.5), rtol=1e-12)
check(S, "Hottel crossed-string view factor for unit square parallel plates (w=1, L=1)", "exact: sqrt(2) - 1", radiation.view_factor_hottel_crossed_strings(1.0, math.sqrt(2.0), math.sqrt(2.0), 1.0, 1.0), math.sqrt(2.0) - 1.0, rtol=1e-14)

A_enc = np.array([2.0, 2.0, 4.0])
eps_enc = np.array([0.85, 0.65, 0.50])
F_enc = np.array([[0.0, 0.4, 0.6], [0.4, 0.0, 0.6], [0.3, 0.3, 0.4]])
T_enc = np.array([1200.0, 500.0, 300.0])
rad_sol = radiation.solve_radiosity_enclosure(A_enc, eps_enc, F_enc, T_enc, np.array([False, False, True]))
check(S, "Oppenheim 3-surface enclosure with reradiating wall: total energy balance sum(Q_i) = 0", "first law", rad_sol["energy_balance_residual"], 0.0, atol=1e-8)

# 2-surface diffuse-gray infinite parallel plates check:
A_2p = np.array([1.0, 1.0])
eps_2p = np.array([0.8, 0.6])
F_2p = np.array([[0.0, 1.0], [1.0, 0.0]])
T_2p = np.array([1000.0, 400.0])
q_2p_exact = radiation.SIGMA_SB * (1000.0 ** 4 - 400.0 ** 4) / (1.0 / 0.8 + 1.0 / 0.6 - 1.0)
rad_2p = radiation.solve_radiosity_enclosure(A_2p, eps_2p, F_2p, T_2p)
check(S, "Oppenheim 2-surface solver reproduces exact parallel-plate formula", "exact formula", rad_2p["Q"][0], q_2p_exact, rtol=1e-12)


# Chapter 13: view factors and enclosures
_A6, _F6 = radiation.box_view_factors(0.8, 1.2, 0.6)
check(S, "Box view factors from Eqs. (13.5)-(13.6): every row sums to 1", "summation rule", float(np.abs(_F6.sum(axis=1) - 1).max()), 0.0, atol=1e-14)
_roof = ([0, 0, 0.6], [0.8, 0, 0], [0, 1.2, 0])
_hearth = ([0, 0, 0], [0.8, 0, 0], [0, 1.2, 0])
check(S, "Parallel rectangles, Eq. (13.5) vs Gauss-Legendre quadrature of Eq. (13.1)", "quadrature", reference.view_factor_gauss_rectangles(_hearth, _roof, 24), _F6[0, 1], rtol=1e-12)
_fr, _se = reference.mc_box_exchange(0.8, 1.2, 0.6, np.ones(6), 1, 1_000_000, seed=7)
check(S, "Perpendicular rectangles, Eq. (13.6) vs Monte Carlo ray tracing (1e6 bundles)", "Monte Carlo", _fr[2], _F6[1, 2], atol=4 * _se[2])
_a = math.radians(70.0)
check(S, "Crossed strings for hinged strips vs exact 1 - sin(alpha/2)", "exact", radiation.view_factor_crossed_strings([0, 0], [1, 0], [0, 0], [math.cos(_a), math.sin(_a)]), 1 - math.sin(_a / 2), rtol=1e-14)
check(S, "Crossed strings for parallel strips vs 2-D quadrature", "quadrature", reference.view_factor_gauss_strips([0, 0], [1, 0], [0.3, 0.8], [1.5, 0.8], 40), radiation.view_factor_crossed_strings([0, 0], [1, 0], [0.3, 0.8], [1.5, 0.8]), rtol=1e-10)
_Az, _Fz = radiation.box_zone_view_factors(0.8, 1.2, 0.6, 32)
_eps = np.full(len(_Az), 0.5); _eps[:2] = 1.0
_T = np.full(len(_Az), 1000.0); _T[:2] = 773.15, 1273.15
_Q = np.zeros(len(_Az)); _Q[:2] = np.nan
_q_net = radiation.solve_radiosity_enclosure(_Az, _eps, _Fz, _T, Q=_Q)["Q"][1]
_fr, _se = reference.mc_box_exchange(0.8, 1.2, 0.6, [1, 1, 0, 0, 0, 0], 1, 2_000_000, seed=8)
_q_mc = 0.96 * radiation.SIGMA_SB * (1273.15**4 - 773.15**4) * _fr[0]
check(S, "Furnace with reradiating banded walls: network vs Monte Carlo ray tracing", "Monte Carlo", _q_net, _q_mc, rtol=4 * _se[0] / _fr[0] + 1e-3)
_R = 0.5
_Ad = math.pi * _R**2
_mc, _mse = reference.mc_cylinder_cavity(1e-9, _R, 0.6, 200_000, seed=9)
check(S, "Cylindrical cavity of zero depth: apparent emissivity equals the wall emissivity", "limit", _mc, 0.6, atol=4 * _mse)
_res = radiation.radiation_shields(1200.0, 300.0, np.ones(5), np.full(5, 0.2), np.full(5, 0.2))
check(S, "Three plane radiation shields of equal emissivity divide the heat flux by 4", "exact", _res["Q"], radiation.SIGMA_SB * (1200.0**4 - 300.0**4) / (4 * (2 / 0.2 - 1)), rtol=1e-14)


# ------------------------------------------------------------------ 7. Diffusion, porous catalysts & convective mass transfer
S = "Diffusion, porous catalysts, and convective mass transfer"
st_dil = diffusion.stefan_stagnant_film_flux(40.0, 2.5e-5, 0.01, 1e-7, 0.0)
check(S, "Stefan stagnant-film flux approaches equimolar Fick's law as x_A1 -> 0", "dilute limit", st_dil["N_A"] / st_dil["J_A_equimolar"], 1.0, rtol=1e-6)
st_50 = diffusion.stefan_stagnant_film_flux(40.0, 2.5e-5, 0.01, 0.50, 0.0)
check(S, "Stefan drift enhancement factor at x_A1 = 0.5, x_A2 = 0 equals ln(2)/0.5", "exact", st_50["drift_factor"], math.log(2.0) / 0.5, rtol=1e-14)

for geom in ("slab", "sphere"):
    check(S, f"{geom} catalyst effectiveness factor eta(phi -> 0) = 1", "reaction-controlled limit", float(diffusion.catalyst_effectiveness_factor(1e-6, geom)), 1.0, rtol=1e-9)
check(S, "slab catalyst effectiveness factor eta(phi = 20) ~ 1/phi", "diffusion-limited asymptotic", float(diffusion.catalyst_effectiveness_factor(20.0, "slab")), 1.0 / 20.0, rtol=1e-8)
check(S, "sphere catalyst effectiveness factor eta(phi = 30) ~ (3/phi)*(1 - 1/phi)", "asymptotic", float(diffusion.catalyst_effectiveness_factor(30.0, "sphere")), (3.0 / 30.0) * (1.0 - 1.0 / 30.0), rtol=1e-10)

for phi_v in [0.5, 1.5, 4.0]:
    check(
        S,
        f"spherical catalyst FD BVP matches analytical effectiveness at phi={phi_v}",
        "numerical vs analytical",
        diffusion.solve_spherical_catalyst_fd(phi_v, N=501)["effectiveness"],
        float(diffusion.catalyst_effectiveness_factor(phi_v, "sphere")),
        rtol=2e-5,
    )

check(S, "Chilton-Colburn analogy reduces to Reynolds analogy k_c = h/(rho cp) at Le = 1", "identity", convective_mass.chilton_colburn_mass_transfer_coeff(60.0, 1.2, 1000.0, 0.70, 0.70) * 1.2 * 1000.0, 60.0, rtol=1e-14)
check(S, "Stefan blowing factor ln(1 + B_M)/B_M -> 1 as B_M -> 0", "limit", convective_mass.stefan_blowing_factor(1e-9), 1.0, rtol=1e-8)
check(S, "Droplet evaporation d^2-law lifetime scales with d0^2", "d^2-law", convective_mass.droplet_evaporation_time(200e-6, 2.5e-5, 995.0, 0.018, 5000.0, 1000.0)["t_lifetime"] / convective_mass.droplet_evaporation_time(100e-6, 2.5e-5, 995.0, 0.018, 5000.0, 1000.0)["t_lifetime"], 4.0, rtol=1e-14)

# Chapter 14: Stefan tube
for _xs, _r in ((0.078, 417.0), (0.7, 10.0)):
    _b, _prof = reference.stefan_tube_exact(_xs, 0.0, _r)
    check(S, f"Exact Stefan-tube profile meets the surface value (x_s = {_xs}, r = {_r:g})", "boundary condition", float(_prof(_b)), _xs, rtol=1e-12)
_b, _prof = reference.stefan_tube_exact(0.078, 0.0, 417.0)
check(S, "Quasi-steady Stefan tube (14.8) vs exact similarity solution, ethanol-like case", "exact", math.sqrt(math.log(1 / (1 - 0.078)) / (2 * 417.0)), _b, rtol=3e-5)
_xi = np.linspace(0, 1, 101)
_D = 1.2e-5
_sol = reference.stefan_tube_march(0.078, 0.0, 417.0, _D, 2 * _b * math.sqrt(_D * 10.0), (10.0, 3600.0), n=100, x0=_prof(_xi * _b))
check(S, "Moving-grid march of the unsteady Stefan tube vs exact similarity solution", "exact", _sol.y[-1, -1], 2 * _b * math.sqrt(_D * 3600.0), rtol=1e-6)
# Chapter 15: pellets and reacting fronts
for _shape in ("slab", "cylinder", "sphere"):
    check(S, f"{_shape} pellet: finite-difference effectiveness vs Eq. (15.5) at phi = 2", "exact",
          diffusion.pellet_fd(2.0, _shape, 1.0, 320)["eta"], diffusion.effectiveness_first_order(2.0, _shape), rtol=5e-5)
check(S, "Second-order slab pellet: finite differences vs collocation (solve_bvp)", "collocation",
      diffusion.pellet_fd(2.0, "slab", 2.0, 640)["eta"], reference.pellet_bvp(2.0, "slab", 2.0)["eta"], rtol=2e-5)
check(S, "Cylinder effectiveness I1(2 phi)/(phi I0(2 phi)) vs collocation", "collocation",
      diffusion.effectiveness_first_order(1.3, "cylinder"), reference.pellet_bvp(1.3, "cylinder", 1.0)["eta"], rtol=1e-7)
_lam = reference.one_phase_front_exact(1e-4)
check(S, "Sharp front with storage tends to the shrinking-core law as St -> 0", "limit", _lam, math.sqrt(1e-4 / 2), rtol=2e-5)
check(S, "Film flux with sigma = -1 (A -> 2B) equals (cD/L) ln(1 + x1)", "exact", diffusion.film_flux(40.0, 2e-5, 1e-3, 0.5, 0.0, -1.0), 0.8 * math.log(1.5), rtol=1e-14)
# Chapter 16: boundary layer with a species, blowing, drops
check(S, "Species boundary layer at B = 0 vs Pohlhausen quadrature, Sc = 2.5", "exact",
      -reference.species_similarity(2.5)["dphi0"], convection_ext.pohlhausen_heat_transfer(2.5), rtol=1e-5)
check(S, "Species boundary layer at B = 0, Sc = 1: Sh_x/Re_x^0.5 = Blasius f''(0) = 0.332057", "Reynolds analogy",
      -reference.species_similarity(1.0)["dphi0"], 0.33205733621, rtol=1e-7)
_bl = reference.species_similarity(1.74, 2.0)
check(S, "Blowing wall condition f(0) = (2B/Sc) phi'(0) is met", "boundary condition", _bl["f0"], 2 * 2.0 / 1.74 * _bl["dphi0"], rtol=1e-9)
check(S, "Brown-Lawler drag tends to Stokes 24/Re at Re = 1e-6", "limit", convective_mass.drag_coefficient_sphere(1e-6) * 1e-6, 24.0, rtol=1e-3)
_ut, _ = convective_mass.terminal_velocity(5e-6, 1000.0, 1.2, 1.8e-5)
check(S, "Terminal velocity of a 5 um drop vs Stokes's law", "Stokes", _ut, 1000.0 * 9.81 * (5e-6) ** 2 / (18 * 1.8e-5) * (1 - 1.2 / 1000.0), rtol=1e-3)
_dr = reference.drop_evaporation_transient(0.1, 1e-3, s_max=math.log(1e4))
check(S, "Unsteady drop evaporation: late rate vs quasi-steady ln(1 + B)", "quasi-steady limit", _dr["G"][-1], _dr["G_qs"], rtol=2e-2)
# Chapter 17: moist air and the wetted pad
from engtransport import _property_fits
for _t, _rh, _W, _h, _twb in _property_fits.HUMID_AIR_REF:
    if _twb > 0.5 and _rh in (0.3, 0.6):
        _Wf = convective_mass.humidity_ratio(_t + 273.15, _rh)
        check(S, f"Adiabatic saturation temperature at {_t:g} C, {100 * _rh:g} % vs real-gas humid air", "CoolProp HAPropsSI",
              convective_mass.adiabatic_saturation_temperature(_t + 273.15, _Wf) - 273.15, _twb, atol=0.05)
        check(S, f"Humidity ratio at {_t:g} C, {100 * _rh:g} % vs real-gas humid air", "CoolProp HAPropsSI", _Wf, _W, rtol=8e-3)
_Wp = convective_mass.humidity_ratio(308.15, 0.2)
check(S, "Wetted pad, Lewis relation: marched outlet vs 1 - exp(-NTU), NTU = 1.9", "march",
      reference.wetted_pad_march(308.15, _Wp, 1.9, 1.0, n_eval=2)["T"][-1], convective_mass.evaporative_pad(308.15, _Wp, 1.9)[0], atol=0.12)
# Chapter 18: fouling from operating data and the cleaning interval
from scipy.optimize import curve_fit as _curve_fit, minimize_scalar as _minimize_scalar
_t = np.arange(0.0, 200.0)
_R = exchangers.asymptotic_fouling(_t, 1.2e-3, 80.0) + 3e-5 * np.sin(1.7 * _t)
_f = exchangers.fit_asymptotic_fouling(_t, _R)
_p, _c = _curve_fit(lambda t, a, b: a * (1 - np.exp(-t / b)), _t, _R, p0=[1e-3, 50.0])
check(S, "Kern-Seaton fit by variable projection vs scipy curve_fit: tau", "curve_fit", _f["tau"], _p[1], rtol=1e-6)
check(S, "Kern-Seaton fit: standard error of tau vs curve_fit", "curve_fit", math.sqrt(_f["cov"][1, 1]), math.sqrt(_c[1, 1]), rtol=1e-4)
_ch, _cc = 75000.0, 84000.0
_eps = exchangers.effectiveness_from_ntu(1.3, _ch / _cc, "shell_and_tube_1_2n")
_Q = _eps * _ch * 170.0
_ua = exchangers.ua_from_temperatures(30.0, 2500.0, 573.15, 573.15 - _Q / _ch, 40.0, 2100.0, 403.15, 403.15 + _Q / _cc, "shell_and_tube_1_2n")
check(S, "UA recovered from exact terminal temperatures (1-2 exchanger)", "round trip", _ua["UA"], 1.3 * _ch, rtol=1e-12)
_L = lambda t: 500.0 * (1 - np.exp(-np.asarray(t) / 80.0))
_topt, _J = exchangers.optimum_cleaning_interval(_L, 15000.0, 4000.0, 2.0)
_res = _minimize_scalar(lambda tc: exchangers.cleaning_cycle_cost(tc, _L, 15000.0, 4000.0, 2.0), bounds=(10, 2000), method="bounded", options={"xatol": 1e-7})
check(S, "Optimum cleaning interval: golden section vs scipy minimize_scalar", "minimize_scalar", _topt, _res.x, rtol=1e-5)
check(S, "Optimum cleaning interval meets L(t_c) = J(t_c), Eq. (18.8)", "optimality", float(_L(_topt)), _J, rtol=1e-4)

# The zeros of J0 used by the cylinder series:
for k_idx, z_val in enumerate(transient._J0_ZEROS[:10], start=1):
    check("Transient conduction, Heisler eigenvalues, and Stefan moving boundaries", f"Bessel J0 root #{k_idx} residual |J0(z_{k_idx})| = 0", "Abramowitz & Stegun", abs(transient.bessel_j0(z_val)), 0.0, atol=2e-7)

"""Tests for the code used in the chapters: root finders, properties and the reference solvers."""

import math

import numpy as np
import pytest

from engtransport import convection_ext, convection_int, exchangers, fins, natural_conv, numerics, phase_change, properties, reference, transient
from engtransport.constants import SIGMA_SB


def test_newton_and_bisect_find_sqrt2():
    f, df = (lambda x: x * x - 2.0), (lambda x: 2.0 * x)
    root, history = numerics.newton(f, df, 1.0)
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-14)
    assert len(history) < 8
    root, history = numerics.bisect(f, 0.0, 2.0, xtol=1e-13)
    assert root == pytest.approx(math.sqrt(2.0), abs=1e-12)


def test_bisect_needs_a_bracket():
    with pytest.raises(ValueError):
        numerics.bisect(lambda x: x * x + 1.0, -1.0, 1.0)


@pytest.mark.parametrize("T, mu, k, cp, pr", [
    (300.0, 1.85373e-05, 0.0263844657, 1006.3739076641, 0.7070636188),   # Lemmon & Jacobsen (2004)
    (600.0, 3.07687e-05, 0.0460112538, 1051.2032568664, 0.7029621396),
])
def test_air_properties_match_reference(T, mu, k, cp, pr):
    air = properties.dry_air_properties(T)
    assert air["mu"] == pytest.approx(mu, rel=1e-3)
    assert air["k"] == pytest.approx(k, rel=2e-3)
    assert air["cp"] == pytest.approx(cp, rel=1e-3)
    assert air["Pr"] == pytest.approx(pr, rel=3e-3)


def test_water_properties_match_iapws():
    w = properties.water_properties(373.15)
    assert w["rho"] == pytest.approx(958.3490516049, rel=1e-3)
    assert w["cp"] == pytest.approx(4215.6736168158, rel=3e-3)
    assert w["mu"] == pytest.approx(0.000281582, rel=1e-3)


def test_properties_accept_arrays_and_refuse_out_of_range():
    T = np.array([250.0, 500.0])
    assert properties.dry_air_properties(T)["k"].shape == (2,)
    with pytest.raises(ValueError):
        properties.water_properties(500.0)


def test_insulated_pipe_steady_matches_resistance_without_radiation():
    r1, r2, k, h, T_w, T_inf = 0.05, 0.10, 0.05, 20.0, 380.0, 250.0
    exact = (T_w - T_inf) / (math.log(r2 / r1) / (2 * math.pi * k) + 1 / (2 * math.pi * r2 * h))
    errors = [abs(reference.insulated_pipe_fv(T_w, r1, r2, k, h, 0.0, T_inf, n_cells=n)["q"] - exact)
              for n in (20, 40)]
    assert errors[1] < 1e-3 * exact
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.05)   # second order


def test_insulated_pipe_cooling_conserves_energy():
    r1, r2 = 0.05, 0.08
    t = np.linspace(0.0, 20 * 3600.0, 401)
    out = reference.insulated_pipe_fv(370.0, r1, r2, 0.04, 15.0, 0.8, 260.0, n_cells=20, t_eval=t,
                                      rho_c_ins=8.4e4, C_cladding=500.0, C_pipe=3.0e4)
    assert out["energy_error"] < 1e-5
    assert np.all(np.diff(out["T_water"]) < 0)


def test_radiation_coefficient_factorisation():
    Ts, Tsur = 300.0, 250.0
    hr = SIGMA_SB * (Ts + Tsur) * (Ts**2 + Tsur**2)
    assert hr * (Ts - Tsur) == pytest.approx(SIGMA_SB * (Ts**4 - Tsur**4), rel=1e-14)


# ---- Chapter 2


def test_steady_conduction_fv_layered_wall_is_exact():
    faces = np.concatenate((np.linspace(0, 0.1, 11), np.linspace(0.1, 0.25, 16)[1:]))
    k = np.concatenate((np.full(10, 0.7), np.full(15, 0.04)))
    out = reference.steady_conduction_fv(faces, k, "plane", left=("convection", 8.0, 293.15),
                                         right=("convection", 23.0, 253.15))
    exact = 40.0 / (1 / 8 + 0.1 / 0.7 + 0.15 / 0.04 + 1 / 23)
    assert out["Q_right"] == pytest.approx(exact, rel=1e-12)
    assert abs(out["balance"]) < 1e-9


def test_steady_conduction_fv_generation_in_a_cylinder_converges():
    R, k, qv, Ts = 0.01, 2.0, 1e6, 400.0
    errors = []
    for n in (20, 40):
        out = reference.steady_conduction_fv(np.linspace(0, R, n + 1), np.full(n, k), "cylinder", q_gen=qv,
                                             left=("symmetry",), right=("temperature", Ts))
        exact = Ts + qv * (R**2 - out["r"] ** 2) / (4 * k)
        errors.append(np.max(np.abs(out["T"] - exact)))
    assert errors[1] < 1e-3 * (qv * R**2 / (4 * k))
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)


def test_steady_conduction_fv_sphere_with_k_of_T_matches_kirchhoff():
    # k = k0 (1 + beta T): the mean-temperature rule is exact for the heat flow
    k0, beta, r1, r2, T1, T2 = 1.0, 2e-3, 0.01, 0.03, 600.0, 300.0
    out = reference.steady_conduction_fv(np.linspace(r1, r2, 201), lambda T, i: k0 * (1 + beta * T), "sphere",
                                         left=("temperature", T1), right=("temperature", T2))
    k_mean = k0 * (1 + beta * 0.5 * (T1 + T2))
    exact = 4 * math.pi * k_mean * (T1 - T2) / (1 / r1 - 1 / r2)
    assert out["Q_right"] == pytest.approx(exact, rel=1e-4)


def test_uo2_conductivity_and_dew_point():
    assert properties.uo2_conductivity(1000.0) == pytest.approx(3.467, rel=1e-3)
    assert properties.saturation_pressure_water(293.15) == pytest.approx(2339.3, rel=5e-3)   # IAPWS-95
    assert properties.dew_point(293.15, 0.5) - 273.15 == pytest.approx(9.26, abs=0.02)
    assert properties.dew_point(293.15, 1.0) == pytest.approx(293.15, abs=1e-9)


# ---- Chapter 3


def test_plate_fin_series_reduces_to_1d_fin_for_small_biot():
    L, b, k, h = 0.041, 1e-3, 200.0, 40.0
    m = math.sqrt(h / (k * b))
    q_1d = math.sqrt(h * k * b) * math.tanh(m * L)
    q_2d = reference.plate_fin_2d_series(L, b, k, h)["Q_per_K"]
    assert q_2d == pytest.approx(q_1d, rel=1e-4)
    assert q_2d < q_1d                          # the 1-D theory overestimates


def test_plate_fin_series_first_root_and_base_temperature():
    s = reference.plate_fin_2d_series(1.5, 0.09, 2.3, 23.0)
    z0 = s["lam"][0] * 0.09
    assert z0 * math.tan(z0) == pytest.approx(23.0 * 0.09 / 2.3, rel=1e-12)
    assert s["theta"](0.0, 0.03) == pytest.approx(1.0, abs=1e-3)
    assert s["theta"](1.5, 0.0) < 1e-5            # tip of a long fin: at the air temperature


def test_plate_fin_fv_converges_to_series_with_balance():
    exact = reference.plate_fin_2d_series(0.05, 1e-3, 200.0, 40.0)["Q_per_K"]
    errors = []
    for ny in (4, 8):
        fv = reference.plate_fin_2d_fv(0.05, 1e-3, 200.0, 40.0, 1.0, 0.0, nx=12 * ny, ny=ny)
        errors.append(abs(fv["Q_base"] / exact - 1))
        assert abs(fv["balance"]) < 1e-9
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.1)


def test_plate_fin_fv_convective_tip_matches_1d_exact():
    L, b, k, h = 0.04, 1e-3, 200.0, 40.0
    m = math.sqrt(h / (k * b)); r = h / (m * k)
    q_1d = math.sqrt(h * k * b) * (math.sinh(m * L) + r * math.cosh(m * L)) / (math.cosh(m * L) + r * math.sinh(m * L))
    fv = reference.plate_fin_2d_fv(L, b, k, h, 1.0, 0.0, nx=160, ny=8, tip="convection")
    assert fv["Q_base"] == pytest.approx(q_1d, rel=2e-4)


def test_heat_sink_resistance_and_limits():
    s = fins.plate_fin_heat_sink(10, 0.04, 2e-3, 0.1, 0.1, 200.0, 40.0)
    assert s["R"] == pytest.approx(0.3053, rel=1e-3)
    # very conductive fins: eta_o -> 1 and R -> 1/(h A_t)
    s_inf = fins.plate_fin_heat_sink(10, 0.04, 2e-3, 0.1, 0.1, 1e9, 40.0)
    assert s_inf["eta_o"] == pytest.approx(1.0, abs=1e-6)
    assert s_inf["R"] == pytest.approx(1 / (40.0 * s_inf["A_t"]), rel=1e-6)
    with pytest.raises(ValueError):
        fins.plate_fin_heat_sink(60, 0.04, 2e-3, 0.1, 0.1, 200.0, 40.0)


def test_schmidt_optimum():
    from scipy.optimize import brentq
    beta = brentq(lambda b: math.sinh(2 * b) - 6 * b, 0.5, 3.0)
    assert beta == pytest.approx(1.4192, abs=1e-4)


# ---- Chapter 4


def _image_theta(z, r):
    a = math.sqrt(z * z - r * r)
    return lambda X, Y: np.log((X**2 + (Y + a) ** 2) / (X**2 + (Y - a) ** 2)) / (2 * math.acosh(z / r))


def test_laplace_fd_buried_pipe_matches_image_solution():
    z, r = 1.0, 0.1125
    th = _image_theta(z, r)
    S = 2 * math.pi / math.acosh(z / r)
    errors = []
    for h in (0.04, 0.02):
        out = reference.laplace_2d_fd(3.0, 3.0, h, circles=[(0.0, z, r, 1.0)], top=("temperature", 0.0),
                                      left=("insulated",), right=("temperature", lambda y: th(3.0, y)),
                                      bottom=("temperature", lambda x: th(x, 3.0)))
        errors.append(abs(2 * out["Q_circles"][0] / S - 1))
        T = out["T"]
        assert np.nanmax(T) <= 1.0 + 1e-12 and np.nanmin(T) >= -1e-12        # maximum principle
    assert errors[1] < 1e-4
    assert errors[0] / errors[1] > 4.0                                       # second order or better


def test_laplace_fd_rectangle_matches_superposition():
    out = reference.laplace_2d_fd(1.0, 1.0, 0.025, top=("temperature", 1.0), bottom=("temperature", 0.0),
                                  left=("temperature", 0.0), right=("temperature", 0.0))
    assert out["T"][20, 20] == pytest.approx(0.25, abs=1e-12)                # centre, by symmetry


def test_laplace_fd_convective_surface_and_effective_depth():
    # with a surface coefficient, the pipe behaves as if it were k/h deeper
    z, r, H = 1.0, 0.05715, 10.0                                             # H = h/k, so k/h = 0.1 m
    q = []
    for far in (("temperature", 0.0), ("insulated",)):
        out = reference.laplace_2d_fd(8.0, 8.0, 0.04, circles=[(0.0, z, r, 1.0)], top=("convection", H, 0.0),
                                      left=("insulated",), right=far, bottom=far)
        q.append(2 * out["Q_circles"][0])
    assert np.mean(q) == pytest.approx(2 * math.pi / math.acosh((z + 0.1) / r), rel=3e-3)


def test_laplace_fd_two_pipes_symmetric():
    out = reference.laplace_2d_fd(4.0, 4.0, 0.04, circles=[(1.8, 1.0, 0.1, 1.0), (2.2, 1.0, 0.1, 1.0)],
                                  top=("temperature", 0.0), left=("temperature", 0.0), right=("temperature", 0.0),
                                  bottom=("temperature", 0.0))
    q1, q2 = out["Q_circles"]
    assert q1 == pytest.approx(q2, rel=1e-10)
    with pytest.raises(ValueError):
        reference.laplace_2d_fd(1.0, 1.0, 0.1, top=("radiation", 0.0))


# ---- Chapter 5


@pytest.mark.parametrize("geom", ["slab", "cylinder", "sphere"])
def test_crank_nicolson_matches_series_and_conserves_energy(geom):
    errors = []
    for n in (20, 40):
        out = reference.transient_conduction_fv(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, geom, n_cells=n, t_end=0.3, n_steps=2 * n)
        exact = transient.transient_1d_series(out["r"], 0.3, 1.0, geom)["theta_star"]
        errors.append(np.max(np.abs(out["T"][-1] - exact)))
        assert out["Q_lost"][-1] == pytest.approx(out["Q_stored"][-1], rel=1e-10)
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.15)


def test_lumped_limit_and_one_term():
    # small Bi: the centre follows the lumped model exp(-(m+1) Bi Fo)
    for geom, m in (("slab", 0), ("cylinder", 1), ("sphere", 2)):
        th = transient.transient_1d_series(np.array([0.0]), 50.0, 1e-3, geom)["theta_star"][0]
        assert th == pytest.approx(math.exp(-(m + 1) * 1e-3 * 50.0), rel=2e-3)
    lam = transient.sphere_eigenvalues(1.0, 3)
    assert lam[0] == pytest.approx(math.pi / 2, rel=1e-10)        # 1 - lam cot(lam) = 1


def test_transient_fv_with_temperature_dependent_h_reduces_to_constant():
    a = reference.transient_conduction_fv(0.025, 30.0, 4.68e6, 1000.0, 60.0, 850.0, "cylinder", n_cells=30,
                                          t_end=20.0, n_steps=200)
    b = reference.transient_conduction_fv(0.025, 30.0, 4.68e6, lambda Ts: 1000.0, 60.0, 850.0, "cylinder",
                                          n_cells=30, t_end=20.0, n_steps=200)
    assert np.allclose(a["T"], b["T"])
    with pytest.raises(ValueError):
        reference.transient_conduction_fv(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, "cube")


# ---- Chapter 6


def test_contact_and_periodic_solutions():
    assert transient.contact_temperature(1.0, 1.0, 10.0, 1.0, 1.0, 0.0) == pytest.approx(5.0)
    assert transient.contact_temperature(1.0, 1.0, 10.0, 100.0, 1.0, 0.0) == pytest.approx(10.0 / 11.0)
    d = math.sqrt(2 * 1e-6 / (2 * math.pi))
    assert transient.periodic_surface(d, 0.0, 1e-6, 0.0, 1.0, 1.0) == pytest.approx(math.exp(-1) * math.cos(-1))


def test_neumann_freezing_satisfies_stefan_condition_and_limits():
    kf, Cf, ku, Cu, L = 2.0, 1.78e6, 1.5, 2.29e6, 8.0e7
    sol = transient.neumann_freezing(kf, Cf, ku, Cu, L, -10.0, 0.0, 5.0)
    t = 1e7
    X = float(sol["front"](t))
    h = 1e-5
    flux_f = kf * (sol["T"](X - h, t) - sol["T"](X - 2 * h, t)) / h      # one-sided slopes at the front
    flux_u = ku * (sol["T"](X + 2 * h, t) - sol["T"](X + h, t)) / h
    dXdt = X / (2 * t)
    assert flux_f - flux_u == pytest.approx(L * dXdt, rel=2e-3)
    # starting at the freezing temperature: one-phase Neumann
    one = transient.neumann_freezing(kf, Cf, ku, Cu, L, -10.0, 0.0, 0.0)
    assert one["lam"] == pytest.approx(transient.stefan_melting_root(Cf * 10.0 / L), rel=1e-9)


def test_enthalpy_method_tracks_neumann_and_conserves_energy():
    kf, Cf, ku, Cu, L = 2.0, 1.78e6, 1.5, 2.29e6, 8.0e7
    sol = transient.neumann_freezing(kf, Cf, ku, Cu, L, -10.0, 0.0, 5.0)
    t = 30 * 86400.0
    out = reference.enthalpy_1d_fv(np.arange(0.0, 4.0 + 0.01, 0.02), kf, Cf, ku, Cu, L, 0.0, 5.0, t,
                                   left=("temperature", -10.0), right=("temperature", 5.0))
    assert out["front"][-1] == pytest.approx(float(sol["front"](t)), abs=4e-3)
    assert abs(out["balance"]) < 1e-10


# ---- Chapter 7


def test_pohlhausen_quadrature_matches_collocation_and_limits():
    for Pr in (0.03, 0.71, 30.0):
        assert convection_ext.pohlhausen_heat_transfer(Pr) == pytest.approx(reference.similarity_bvp(Pr)["dtheta0"], rel=1e-5)
    assert convection_ext.pohlhausen_heat_transfer(1.0) == pytest.approx(0.332057, rel=2e-5)   # Pr = 1: theta = f'
    assert convection_ext.pohlhausen_heat_transfer(1000.0) == pytest.approx(0.339 * 10.0, rel=2e-3)


def test_boundary_layer_march_settles_on_similarity():
    m = reference.boundary_layer_march(0.71, n_x=200, n_y=100)
    assert m["Cf_sqrtRe"][-1] == pytest.approx(0.664115, rel=5e-3)
    assert m["Nu_sqrtRe"][-1] == pytest.approx(convection_ext.pohlhausen_heat_transfer(0.71), rel=2e-3)
    # unheated starting length: larger heat transfer than a plate heated from the edge
    u = reference.boundary_layer_march(0.71, n_x=200, n_y=100, x_heat_start=0.5)
    assert u["Nu_sqrtRe"][-1] > m["Nu_sqrtRe"][-1]


# ---- Chapter 8


def test_graetz_series_eigenvalues_and_limits():
    g = convection_int.graetz_series()
    assert g["lam"][:3] == pytest.approx([2.70436, 6.67903, 10.67338], abs=2e-5)
    assert g["Nu_x"](1.0) == pytest.approx(3.6568, abs=2e-4)
    z = 0.05
    assert g["Nu_mean"](z) == pytest.approx(-np.log(g["theta_m"](z)) / (4 * z))


def test_graetz_march_matches_series_and_flux_limits():
    g = convection_int.graetz_series()
    m = reference.graetz_march(0.5, n_r=100, n_x=300)
    i = int(np.argmin(np.abs(m["zeta"] - 0.05)))
    assert m["theta_m"][i] == pytest.approx(float(g["theta_m"](m["zeta"][i])), abs=1e-4)
    f = reference.graetz_march(2.0, n_r=100, n_x=300, wall="flux")
    assert f["Nu_x"][-1] == pytest.approx(48 / 11, rel=1e-3)
    with pytest.raises(ValueError):
        reference.graetz_march(wall="radiation")


# ---- Chapter 9


def test_free_convection_shooting_matches_collocation_and_ostrach():
    s = natural_conv.free_convection_similarity(0.72)
    b = reference.free_convection_bvp(0.72)
    assert s["G"] == pytest.approx(b["G"], abs=1e-7)
    assert s["fpp0"] == pytest.approx(b["fpp0"], abs=1e-7)
    assert s["G"] == pytest.approx(0.5046, abs=1e-4)          # Ostrach (1953), Pr = 0.72
    assert reference.free_convection_limits()["large_Pr"] / np.sqrt(2) == pytest.approx(0.503, abs=1e-3)


def test_free_convection_march_settles_on_similarity():
    m = reference.free_convection_march(0.72, n_eta=300)
    assert m["G"][-1] == pytest.approx(0.504634, abs=2e-4)


def test_natural_convection_correlations():
    assert natural_conv.nu_horizontal_plate_up_lloyd_moran(1e6) == pytest.approx(0.54 * 1e6 ** 0.25)
    assert natural_conv.nu_horizontal_plate_up_lloyd_moran(1e8) == pytest.approx(0.15 * 1e8 ** (1 / 3))
    lam = natural_conv.nu_vertical_plate_churchill_chu_laminar(1e8, 0.71)
    full = natural_conv.nu_vertical_plate_churchill_chu(1e8, 0.71)
    assert lam == pytest.approx(full, rel=0.15)        # the full-range form is cruder in the laminar range


# ---- Chapter 10


def test_saturation_and_steam_properties():
    s = properties.saturated_water(373.15)
    assert s["p_sat"] == pytest.approx(101418.0, rel=1e-3)
    assert s["h_fg"] == pytest.approx(2.2564e6, rel=1e-3)
    assert s["sigma"] == pytest.approx(0.0589, rel=2e-3)
    assert properties.saturation_temperature_water(101325.0) == pytest.approx(373.124, abs=0.01)
    st = properties.steam_properties(800.0)
    assert st["k"] == pytest.approx(0.0699, rel=5e-3)
    with pytest.raises(ValueError):
        properties.steam_properties(300.0)


def test_film_condensation_subcooling_three_ways():
    for Ja in (0.05, 1.0):
        q = phase_change.film_condensation_subcooling(Ja)["ratio"]
        assert q == pytest.approx(reference.film_condensation_bvp(Ja)["ratio"], abs=1e-10)
    m = reference.film_condensation_march(0.5, n_eta=100)
    assert m["ratio"][-1] == pytest.approx(phase_change.film_condensation_subcooling(0.5)["ratio"], abs=1e-4)


def test_boiling_correlations():
    s = properties.saturated_water(373.15)
    q_max = phase_change.zuber_critical_heat_flux(s["rho"], s["rho_v"], s["h_fg"], s["sigma"])
    assert q_max == pytest.approx(1.26e6, rel=0.01)
    q = phase_change.rohsenow_heat_flux(np.array([10.0, 20.0]), s["mu"], s["h_fg"], s["rho"], s["rho_v"], s["sigma"], s["cp"], s["Pr"])
    assert q[1] / q[0] == pytest.approx(8.0)
    h_v = phase_change.h_film_condensation(958.0, 0.6, 0.68, 2.8e-4, 2.26e6, 10.0, 1.0, "vertical")
    assert h_v == pytest.approx(phase_change.h_condensation_vertical_plate(958.0, 0.6, 0.68, 2.8e-4, 2.26e6, 0.0, 373.15, 363.15, 1.0)["h_bar"], rel=1e-3)


# ---- Chapter 11


def test_crossflow_exact_and_cells():
    e = exchangers.effectiveness_crossflow_exact(1.0, 1.0)
    assert e == pytest.approx(0.476222, abs=1e-6)
    assert reference.crossflow_cells(1.0, 1.0, 80)["eps"] == pytest.approx(e, abs=1e-5)
    assert exchangers.effectiveness_crossflow_exact(1.0, 0.0) == pytest.approx(1 - np.exp(-1.0))
    ntu = exchangers.ntu_from_effectiveness(e, 1.0, "crossflow_exact")
    assert ntu == pytest.approx(1.0, rel=1e-10)


def test_shell_and_tube_and_correction_factor():
    for Cr in (0.3, 1.0):
        f = exchangers.effectiveness_from_ntu(1.5, Cr, "shell_and_tube_1_2n")
        assert reference.shell_and_tube_1_2_bvp(1.5, Cr)["eps"] == pytest.approx(f, abs=1e-10)
    assert exchangers.lmtd_correction_factor(1.0, 0.5, "counterflow") == pytest.approx(1.0)
    assert exchangers.lmtd_correction_factor(2.0, 0.5, "shell_and_tube_1_2n") < 0.8


# ---- Chapter 12


def test_planck_band_fraction_and_constants():
    from engtransport import radiation
    lt = np.array([1000.0, 2897.77, 7300.0, 1e5])
    F = radiation.blackbody_band_fraction(lt)
    assert F[1] == pytest.approx(0.250054, abs=1e-6)
    assert np.all(np.diff(F) > 0)
    assert radiation.band_fraction(0.0, 1e6, 1000.0) == pytest.approx(1.0, abs=1e-12)
    assert radiation.stefan_boltzmann_from_constants() == pytest.approx(5.670374419e-8, rel=1e-9)


def test_step_surface():
    from engtransport import radiation
    s = radiation.step_surface(2.5, 0.95, 0.05)
    assert s["absorptance"]() == pytest.approx(0.9191, abs=1e-4)
    assert s["emissivity"](333.15) == pytest.approx(0.05, abs=1e-3)


# ---- Chapter 13


def test_box_view_factors_close_and_are_reciprocal():
    from engtransport import radiation
    A, F = radiation.box_view_factors(0.8, 1.2, 0.6)
    assert np.allclose(F.sum(axis=1), 1.0, atol=1e-14)
    assert np.allclose(A[:, None] * F, (A[:, None] * F).T, atol=1e-15)
    assert F[1, 0] == pytest.approx(0.341694, abs=1e-6)
    Az, Fz = radiation.box_zone_view_factors(0.8, 1.2, 0.6, 1)
    assert np.allclose(Fz, F, atol=1e-15)
    Az, Fz = radiation.box_zone_view_factors(0.8, 1.2, 0.6, 5)
    assert np.allclose(Fz.sum(axis=1), 1.0, atol=1e-13) and Fz.min() >= 0.0


def test_crossed_strings_and_disks():
    from engtransport import radiation
    for a in np.radians([20.0, 60.0, 120.0]):
        F = radiation.view_factor_crossed_strings([0, 0], [1, 0], [0, 0], [math.cos(a), math.sin(a)])
        assert F == pytest.approx(1 - math.sin(a / 2), abs=1e-14)
    assert radiation.view_factor_coaxial_disks(1.0, 1.0, 1.0) == pytest.approx(0.381966, abs=1e-6)


def test_radiosity_with_prescribed_heat_flow():
    from engtransport import radiation
    A = np.array([1.0, 1.0])
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    first = radiation.solve_radiosity_enclosure(A, [0.8, 0.6], F, [1000.0, 400.0])
    second = radiation.solve_radiosity_enclosure(A, [0.8, 0.6], F, [0.0, 400.0], Q=[first["Q"][0], np.nan])
    assert second["T"][0] == pytest.approx(1000.0, rel=1e-12)
    q = radiation.SIGMA_SB * (1000.0**4 - 400.0**4) / (1 / 0.8 + 1 / 0.6 - 1)
    assert first["Q"][0] == pytest.approx(q, rel=1e-12)


def test_radiation_shields_plane_rule():
    from engtransport import radiation
    for n in (0, 1, 4):
        res = radiation.radiation_shields(1200.0, 300.0, np.ones(n + 2), np.full(n + 2, 0.2), np.full(n + 2, 0.2))
        q = radiation.SIGMA_SB * (1200.0**4 - 300.0**4) / ((n + 1) * (2 / 0.2 - 1))
        assert res["Q"] == pytest.approx(q, rel=1e-12)
        assert np.all(np.diff(res["T"]) < 0)


def test_monte_carlo_box_view_factors():
    from engtransport import radiation
    A, F = radiation.box_view_factors(0.8, 1.2, 0.6)
    frac, se = reference.mc_box_exchange(0.8, 1.2, 0.6, np.ones(6), 1, 200_000, seed=4)
    assert np.all(np.abs(frac - F[1]) <= 5 * se + 1e-12)


# ---- Chapter 14


def test_film_flux_limits_and_profiles():
    from engtransport import diffusion
    assert diffusion.film_flux(1.0, 1.0, 1.0, 0.5, 0.0) == pytest.approx(math.log(2.0), rel=1e-14)
    assert diffusion.film_flux(1.0, 1.0, 1.0, 0.5, 0.0, sigma=0.0) == pytest.approx(0.5, rel=1e-14)
    assert diffusion.film_flux(1.0, 1.0, 1.0, 1e-8, 0.0) == pytest.approx(1e-8, rel=1e-7)
    z = np.linspace(0, 1, 11)
    x = diffusion.film_profile(z, 0.6, 0.1)
    assert x[0] == pytest.approx(0.6) and x[-1] == pytest.approx(0.1)


def test_stefan_tube_quasi_steady_and_exact():
    from engtransport import diffusion
    s = diffusion.stefan_tube_level(np.array([0.0, 3600.0]), 0.01, 1.2e-5, 0.078, 0.0, 417.0)
    assert s[0] == pytest.approx(0.01)
    K = 2 * 1.2e-5 / 417.0 * math.log(1 / (1 - 0.078))
    assert s[1] ** 2 - 0.01**2 == pytest.approx(K * 3600.0, rel=1e-12)
    b, prof = reference.stefan_tube_exact(0.3, 0.05, 50.0)
    assert float(prof(0.0)) == pytest.approx(0.05, abs=1e-13) and float(prof(b)) == pytest.approx(0.3, rel=1e-11)


def test_solvents_and_diffusivities():
    eth = properties.solvent_saturation("ethanol", 298.15)
    assert eth["p_sat"] == pytest.approx(7886.0, rel=1e-3) and eth["rho_l"] == pytest.approx(785.0, rel=1e-3)
    assert properties.water_vapour_diffusivity_air(298.15) == pytest.approx(2.552e-5, rel=1e-3)
    assert properties.saturation_pressure_ice(273.15) == pytest.approx(611.21, rel=1e-12)
    assert properties.saturation_pressure_ice(253.15) < properties.saturation_pressure_water(253.15)
    from engtransport import diffusion
    assert diffusion.millington_quirk(0.45) == pytest.approx(0.45 ** (4 / 3), rel=1e-12)


# ---- Chapter 15


def test_effectiveness_shapes_and_limits():
    from engtransport import diffusion
    for shape in ("slab", "cylinder", "sphere"):
        assert diffusion.effectiveness_first_order(1e-6, shape) == pytest.approx(1.0, abs=1e-9)
        assert diffusion.effectiveness_first_order(200.0, shape) == pytest.approx(1 / 200.0, rel=0.01)
    assert diffusion.effectiveness_first_order(1.0, "slab") == pytest.approx(math.tanh(1.0), rel=1e-14)
    assert diffusion.i1_over_i0(1.0) == pytest.approx(0.44638997, rel=1e-7)
    # Aris modulus for a sphere is R/3 sqrt(k/D): compare with the R-based formula of `catalyst_effectiveness_factor`
    assert diffusion.effectiveness_first_order(1.0, "sphere") == pytest.approx(float(diffusion.catalyst_effectiveness_factor(3.0, "sphere")), rel=1e-12)


def test_pellet_fd_against_exact():
    from engtransport import diffusion
    for shape in ("slab", "cylinder", "sphere"):
        r = diffusion.pellet_fd(2.0, shape, 1.0, 160)
        assert r["eta"] == pytest.approx(diffusion.effectiveness_first_order(2.0, shape), rel=2e-4)
    with pytest.raises(ValueError):
        diffusion.pellet_fd(1.0, "slab", 0.0)


def test_shrinking_core_and_coating():
    from engtransport import diffusion
    x = diffusion.shrinking_core_depth(1e9, 1e-8, 0.0175, 1500.0)
    assert x == pytest.approx(math.sqrt(2 * 1e-8 * 0.0175 * 1e9 / 1500.0), rel=1e-12)
    xc = diffusion.shrinking_core_depth(1e9, 1e-8, 0.0175, 1500.0, R_outer=3e6)
    assert xc < x and xc**2 / (2e-8) + 3e6 * xc == pytest.approx(0.0175 * 1e9 / 1500.0, rel=1e-10)
    assert diffusion.observable_modulus(4.0, 1e-3, 2e-6, 1.0) == pytest.approx(2.0)


def test_mass_fractions_spalding_and_blowing_factor():
    from engtransport import convective_mass
    m = convective_mass.vapour_mass_fraction(0.5 * 101325.0, 0.018015)
    assert m == pytest.approx(0.018015 / (0.018015 + 0.02897), rel=1e-12)
    assert convective_mass.spalding_number(0.5, 0.0) == pytest.approx(1.0)
    f = convective_mass.film_blowing_factor(np.array([0.0, 1e-9, 1.0, -0.5]))
    assert f == pytest.approx([1.0, 1.0, math.log(2.0), 2 * math.log(2.0)], rel=1e-9)
    with pytest.raises(ValueError):
        convective_mass.spalding_number(1.0, 0.0)


def test_falling_drop_small_drops_follow_the_still_d2_law():
    from engtransport import convective_mass
    # a 10 um drop barely moves: its lifetime is close to that of a drop at rest (Sh close to 2)
    r = convective_mass.falling_drop(10e-6, 998.0, 1.18, 1.85e-5, 2.5e-5, 0.006, 0.61)
    t_rest = (10e-6) ** 2 / convective_mass.d2_law_rate(10e-6, 1.18, 2.5e-5, 998.0, 0.006)
    assert 0.97 * t_rest < r["t_life"] < t_rest       # its slight motion ventilates it a little
    assert np.all(np.diff(r["z"]) >= 0) and r["z_life"] < 1e-3


def test_species_similarity_suction_and_blowing():
    base = reference.species_similarity(0.61)["dphi0"]
    assert reference.species_similarity(0.61, -0.3)["dphi0"] / base > 1.0 + math.log(0.7) / -0.3 - 1.0
    blown = reference.species_similarity(0.61, 1.0)
    assert blown["f0"] < 0 and blown["dphi0"] / base < math.log(2.0)


def test_psychrometrics_round_trip_and_wet_bulb_ordering():
    from engtransport import convective_mass as cm
    T = 303.15
    W = cm.humidity_ratio(T, 0.4)
    assert cm.relative_humidity(T, W) == pytest.approx(0.4, rel=1e-12)
    # enthalpy is unchanged along adiabatic saturation, up to the small liquid term
    T_as = cm.adiabatic_saturation_temperature(T, W)
    W_s = cm.humidity_ratio(T_as, 1.0)
    assert cm.moist_air_enthalpy(T_as, W_s) - cm.moist_air_enthalpy(T, W) == pytest.approx((W_s - W) * cm.CP_LIQUID * (T_as - 273.15), rel=1e-9)
    # a smaller psychrometric ratio gives a colder wet surface; r = 1 sits close to T_as
    t1, t09 = cm.wet_surface_temperature(T, W, 1.0), cm.wet_surface_temperature(T, W, 0.9)
    assert t09 < t1 and abs(t1 - T_as) < 0.1


def test_evaporative_pad_limits():
    from engtransport import convective_mass as cm
    T, W = 308.15, cm.humidity_ratio(308.15, 0.2)
    T0, W0, e0 = cm.evaporative_pad(T, W, 0.0)
    assert T0 == pytest.approx(T) and W0 == pytest.approx(W) and e0 == 0.0
    Tinf, Winf, einf = cm.evaporative_pad(T, W, 50.0)
    assert Tinf == pytest.approx(cm.adiabatic_saturation_temperature(T, W), abs=1e-9)
    assert cm.relative_humidity(Tinf, Winf) == pytest.approx(1.0, rel=1e-9)


def test_fouling_fit_recovers_exact_law_and_cleaning_optimum_condition():
    from engtransport import exchangers as ex
    t = np.arange(0.0, 300.0)
    f = ex.fit_asymptotic_fouling(t, ex.asymptotic_fouling(t, 1.2e-3, 80.0))
    assert f["R_inf"] == pytest.approx(1.2e-3, rel=1e-8) and f["tau"] == pytest.approx(80.0, rel=1e-8)
    L = lambda s: 400.0 * (1 - np.exp(-np.asarray(s) / 60.0))
    tc, J = ex.optimum_cleaning_interval(L, 10000.0, 3000.0, 2.0)
    assert float(L(tc)) == pytest.approx(J, rel=1e-4)
    # an expensive cleaning never pays with asymptotic fouling: the optimum runs to the bound
    tc_big, _ = ex.optimum_cleaning_interval(L, 1e6, 3000.0, 2.0, t_bounds=(1.0, 1e4))
    assert tc_big > 0.99e4


def test_ua_from_temperatures_round_trip_arrays():
    from engtransport import exchangers as ex
    ch, cc = np.array([75000.0, 60000.0]), np.array([84000.0, 90000.0])
    ntu = np.array([1.0, 2.0])
    eps = np.array([ex.effectiveness_from_ntu(n, a / b, "counterflow") for n, a, b in zip(ntu, ch, cc)])
    Q = eps * ch * 150.0
    r = ex.ua_from_temperatures(ch / 2500.0, 2500.0, 450.0, 450.0 - Q / ch, cc / 2100.0, 2100.0, 300.0, 300.0 + Q / cc)
    assert r["UA"] == pytest.approx(ntu * ch, rel=1e-12)
    assert np.allclose(r["imbalance"], 0.0, atol=1e-12)

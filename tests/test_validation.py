"""Comprehensive 68-check validation suite for engtransport."""

import math
import unittest
import numpy as np


from engtransport import (
    conduction,
    convection_ext,
    convection_int,
    convective_mass,
    diffusion,
    exchangers,
    fins,
    properties,
    radiation,
    transient,
)


class ConductionAndFinsValidation(unittest.TestCase):
    def test_cylindrical_and_spherical_shells_collapse_to_plane_wall_in_thin_limit(self):
        R = 100.0
        dr = 1e-4
        k = 25.0
        r_cyl = conduction.r_cylindrical_shell(R, R + dr, L=1.0, k=k)
        r_plane_cyl = conduction.r_plane_wall(dr, k=k, A=2.0 * math.pi * R * 1.0)
        self.assertAlmostEqual(r_cyl / r_plane_cyl, 1.0, places=6)

        r_sph = conduction.r_spherical_shell(R, R + dr, k=k)
        r_plane_sph = conduction.r_plane_wall(dr, k=k, A=4.0 * math.pi * (R ** 2))
        self.assertAlmostEqual(r_sph / r_plane_sph, 1.0, places=5)

    def test_critical_insulation_radius_maximizes_heat_transfer(self):
        k_ins = 0.15
        h_ext = 10.0
        r1 = 0.005
        r_crit = conduction.critical_insulation_radius(k_ins, h_ext, "cylinder")
        self.assertAlmostEqual(r_crit, 0.015, places=12)

        def total_r(r2):
            return conduction.r_cylindrical_shell(r1, r2, 1.0, k_ins) + 1.0 / (2.0 * math.pi * r2 * h_ext)

        self.assertLess(total_r(r_crit), total_r(r_crit * 0.8))
        self.assertLess(total_r(r_crit), total_r(r_crit * 1.25))

    def test_kirchhoff_transform_conserves_flux_everywhere(self):
        x = np.linspace(0.0, 0.2, 1001)
        q_flux, T_x = conduction.kirchhoff_linear_k_wall(x, L=0.2, T1=600.0, T2=300.0, k0=20.0, beta=0.002)
        dT_dx = np.gradient(T_x, x, edge_order=2)
        k_local = 20.0 * (1.0 + 0.002 * (T_x - 600.0))
        q_local = -k_local * dT_dx
        np.testing.assert_allclose(q_local[2:-2], q_flux, rtol=1e-5)

    def test_2d_steady_laplacian_matches_exact_fourier_double_series(self):
        x, y, T_num = conduction.solve_2d_steady_rect(41, 41, W=1.0, H=1.0, T_left=300.0, T_right=300.0, T_bottom=300.0, T_top=400.0)
        T_exact = conduction.analytical_2d_rectangular_plate(x, y, W=1.0, H=1.0, T_base=300.0, T_top=400.0, n_terms=100)
        # At the center of a square plate (0.5, 0.5), symmetry requires T = 300 + (400 - 300)/4 = 325.0 K!
        self.assertAlmostEqual(T_num[20, 20], 325.0, places=10)
        self.assertAlmostEqual(T_exact[20, 20], 325.0, places=6)
        np.testing.assert_allclose(T_num[5:-5, 5:-5], T_exact[5:-5, 5:-5], rtol=2e-3)

    def test_fin_asymptotic_limits_and_array_efficiency(self):
        x = np.linspace(0.0, 0.05, 51)
        short_fin = fins.uniform_fin_solution(x, L=1e-5, P=0.04, Ac=1e-4, k=200.0, h=10.0, T_b=400.0, T_inf=300.0)
        self.assertAlmostEqual(short_fin["efficiency"], 1.0, places=8)

        long_fin = fins.uniform_fin_solution(x, L=2.0, P=0.04, Ac=1e-5, k=20.0, h=100.0, T_b=400.0, T_inf=300.0, tip_bc="adiabatic")
        inf_fin = fins.uniform_fin_solution(x, L=2.0, P=0.04, Ac=1e-5, k=20.0, h=100.0, T_b=400.0, T_inf=300.0, tip_bc="infinite")
        self.assertAlmostEqual(long_fin["q_fin"], inf_fin["q_fin"], places=6)


class TransientAndStefanValidation(unittest.TestCase):
    def test_heisler_series_converges_to_lumped_capacitance_at_small_biot(self):
        # For Bi = 1e-5: Lumped capacitance uses Bi_lumped = h * (V/As) / k:
        #   - Slab: V/As = L -> theta = exp(-Bi * Fo)
        #   - Cylinder: V/As = R/2 -> theta = exp(-2 * Bi * Fo)
        #   - Sphere: V/As = R/3 -> theta = exp(-3 * Bi * Fo)
        Bi = 1e-5
        Fo = 5000.0
        for geom, factor in [("slab", 1.0), ("cylinder", 2.0), ("sphere", 3.0)]:
            res = transient.transient_1d_series(0.0, Fo=Fo, Bi=Bi, geometry=geom, n_modes=10)
            expected = math.exp(-factor * Bi * Fo)
            self.assertAlmostEqual(float(res["theta_star"]), expected, places=5)

    def test_eigenvalues_satisfy_transcendental_characteristic_equations(self):
        Bi = 1.75
        for xi in transient.slab_eigenvalues(Bi, 10):
            self.assertAlmostEqual(xi * math.tan(xi), Bi, places=10)
        for xi in transient.sphere_eigenvalues(Bi, 10):
            self.assertAlmostEqual(1.0 - xi / math.tan(xi), Bi, places=10)
        for xi in transient.cylinder_eigenvalues(Bi, 10):
            self.assertAlmostEqual(xi * transient.bessel_j1(xi) / transient.bessel_j0(xi), Bi, places=10)

    def test_stefan_small_stefan_number_quasi_steady_limit(self):
        # As Ste -> 0, lambda * exp(lambda^2) * erf(lambda) ~ (2/sqrt(pi))*lambda^2 = Ste/sqrt(pi) => lambda ~ sqrt(Ste / 2)
        Ste = 1e-5
        lam = transient.stefan_melting_root(Ste)
        self.assertAlmostEqual(lam / math.sqrt(0.5 * Ste), 1.0, places=4)


class ConvectionAndExchangersValidation(unittest.TestCase):
    def test_blasius_shooter_reproduces_exact_howarth_wall_shear(self):
        blasius = convection_ext.solve_blasius(eta_max=10.0, n_steps=1500)
        self.assertAlmostEqual(blasius["fpp0"], 0.332057336, places=6)

    def test_power_law_nusselt_recover_newtonian_and_plug_limits(self):
        self.assertAlmostEqual(convection_int.nu_laminar_power_law_const_q(1.0), 48.0 / 11.0, places=12)
        self.assertAlmostEqual(convection_int.nu_laminar_power_law_const_q(1e-9), 8.0, places=6)

    def test_lmtd_and_epsilon_ntu_give_identical_heat_duty(self):
        for flow in ("counterflow", "parallel"):
            res = exchangers.rate_exchanger(UA=3200.0, C_h=2500.0, C_c=1800.0, T_h_in=450.0, T_c_in=310.0, flow_type=flow)
            self.assertAlmostEqual(res["Q"], res["Q_lmtd"], places=8)
            ntu_back = exchangers.ntu_from_effectiveness(res["effectiveness"], res["Cr"], flow_type=flow)
            self.assertAlmostEqual(ntu_back, res["NTU"], places=9)


class RadiationAndDiffusionValidation(unittest.TestCase):
    def test_planck_integral_equals_stefan_boltzmann_and_band_fraction_limits(self):
        T = 1500.0
        lam_grid = np.geomspace(0.1, 200.0, 4000)
        E_lam = radiation.planck_spectral_emissive_power(lam_grid, T)
        integral_E = float(np.trapezoid(E_lam, lam_grid))
        exact_sb = radiation.SIGMA_SB * (T ** 4)
        self.assertAlmostEqual(integral_E / exact_sb, 1.0, places=3)
        self.assertAlmostEqual(radiation.blackbody_band_fraction(1e6), 1.0, places=6)
        # At Wien peak lambda_max * T = 2897.77 um*K, F_(0 -> lambda_max*T) is universally 0.25003:
        self.assertAlmostEqual(radiation.blackbody_band_fraction(radiation.WIEN_B), 0.25003, places=4)

    def test_oppenheim_radiosity_enclosure_conserves_total_energy(self):
        areas = np.array([2.0, 2.0, 4.0])
        eps = np.array([0.85, 0.65, 0.50])
        F = np.array([
            [0.0, 0.4, 0.6],
            [0.4, 0.0, 0.6],
            [0.3, 0.3, 0.4],
        ])
        T = np.array([1100.0, 450.0, 300.0])
        res = radiation.solve_radiosity_enclosure(areas, eps, F, T, reradiating_mask=np.array([False, False, True]))
        self.assertAlmostEqual(res["energy_balance_residual"], 0.0, places=8)
        self.assertAlmostEqual(res["Q"][0] + res["Q"][1], 0.0, places=8)

    def test_stefan_diffusion_and_porous_catalyst_fd_match_analytical(self):
        # 1. Dilute limit of Stefan diffusion matches Fick's law
        st_dilute = diffusion.stefan_stagnant_film_flux(40.0, 2.5e-5, 0.01, 1e-6, 0.0)
        self.assertAlmostEqual(st_dilute["N_A"] / st_dilute["J_A_equimolar"], 1.0, places=5)

        # 2. Spherical Thiele modulus finite-difference BVP matches analytical effectiveness factor
        for phi_val in [0.5, 2.0, 5.0]:
            eta_exact = float(diffusion.catalyst_effectiveness_factor(phi_val, geometry="sphere"))
            eta_fd = diffusion.solve_spherical_catalyst_fd(phi_val, N=501)["effectiveness"]
            self.assertAlmostEqual(eta_fd, eta_exact, places=4)

    def test_fuller_diffusivity_and_chilton_colburn_analogy(self):
        # H2O - Air at 298.15 K, 1 atm is ~2.5e-5 m^2/s
        D_water_air = properties.fuller_diffusivity("H2O", "Air", 298.15, 101325.0)
        self.assertTrue(2.3e-5 <= D_water_air <= 2.8e-5)

        # At Sc == Pr (Le == 1), k_c * rho * cp == h_heat (Reynolds analogy)
        kc = convective_mass.chilton_colburn_mass_transfer_coeff(h_heat=50.0, rho=1.2, cp=1005.0, Pr=0.71, Sc=0.71)
        self.assertAlmostEqual(kc * 1.2 * 1005.0, 50.0, places=10)


if __name__ == "__main__":
    unittest.main()

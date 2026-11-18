# Changelog

## 0.1.0 (2026-11-18)

First release. An open book of 18 executed chapters in four parts, each with derivations,
numbered equations, three figures, four exercises and executed sample solutions, with every
result checked against an exact or independent numerical solution, and the `engtransport`
package underneath. The book grew out of an earlier library with 18 short notebook outlines.

### Part I. Conduction

- Chapter 1: heat balances and the three modes of heat transfer. An insulated
  district-heating pipe in an Astana winter: the surface balance solved by Newton's
  method, the resistance network and successive substitution, and the time for the
  water to reach 0 °C after the pumps stop. Checked against finite volumes through the
  insulation, steady (second-order convergence) and in time (with the heat stored in
  the insulation and the cladding), with the energy balance closing to 3 × 10⁻⁷.
- Chapter 2: steady conduction and thermal resistance. A brick wall in an Astana winter
  (U-value, inner surface temperature and the dew point); the critical radius and the
  ampacity of a PVC-insulated cable; heat generation in a wire and in a uranium dioxide
  fuel pellet; conductivity that depends on temperature and the Kirchhoff transform.
  Checked against finite volumes with layers, generation and k(T) (second-order
  convergence to the Kirchhoff solution); the constant-k shortcuts for the pellet err by
  up to hundreds of kelvin.
- Chapter 3: fins and extended surfaces. The fin equation, tip conditions, efficiency and
  effectiveness; a plate-fin heat sink for a 150 W power module; Schmidt's optimum fin
  for a given amount of metal; a concrete balcony slab as a thermal bridge. The 1-D fin
  theory is checked against the exact two-dimensional series solution and 2-D finite
  volumes: it overestimates the heat by about Bi/6.
- Chapter 4: two-dimensional steady conduction. Separation of variables and the decay of
  disturbances; line sources and images; a buried pre-insulated district-heating pipe and
  the supply and return pipes in one trench. Checked with a finite-difference solver
  that handles circular pipes (Shortley–Weller): second-order or faster convergence to the
  exact solutions, the error of cutting off the soil, and the soil conductances of two
  pipes, which line sources get wrong by up to 22 % while the heat losses of insulated
  pipes stay within 0.3 %.
- Chapter 5: transient conduction. The lumped model and the Biot number; series solutions
  for the slab, the cylinder and the sphere; quenching a steel bar in oil and in water, and
  why the size of the bar matters more than the quenchant. Checked with Crank–Nicolson
  finite volumes (second order in space and time, energy balance to rounding error); the
  lumped model is tested against the series.
- Chapter 6: semi-infinite solids and phase change. The error-function solution; the
  contact temperature and why a steel railing at −30 °C feels so much colder than wood;
  the seasonal temperature wave in the ground; frost under a road cleared of snow, by
  Neumann's two-phase solution (2.34 m after 176 days, against 2.45 m measured near
  Astana) and Stefan's formula. Checked with an enthalpy-method finite-volume solver
  (first-order convergence of the front, energy balance to rounding error), which also
  checks the contact temperature and the seasonal wave.
- Chapter 7: boundary layers and external flow. Scaling of the boundary layer; the
  Blasius and Pohlhausen similarity solutions, with the wall heat flux as an exact
  quadrature for any Prandtl number; turbulent and mixed flat plates; the
  Churchill–Bernstein correlation and the wind on the pipe of Chapter 1 (h = 25 W/(m² K)
  is a 6 m/s wind). Checked by collocation of the similarity equations and by a marching
  solution of the full boundary-layer equations, which settles on the similarity values.
- Chapter 8: internal flow and the Graetz problem. The bulk temperature and its exponential
  approach to the wall temperature; fully developed laminar Nusselt numbers; the Graetz
  series and Lévêque's entrance solution; a heater for viscous oil (the whole heater is in
  the thermal entry region); Gnielinski's correlation and the district-heating main of
  Chapter 1 (inside film negligible, 1.8 K/km cooling). Checked with a marching
  finite-volume solution (second order, to 10⁻⁵), including the uniform-flux result for
  power-law fluids, which replaces a check that only restated the formula.
- Chapter 9: natural convection. The Boussinesq boundary-layer equations and the Grashof
  and Rayleigh numbers; Ostrach's similarity solution for a vertical plate, by shooting,
  with its large-Prandtl limit; the correlations of Churchill and Chu and of Lloyd and
  Moran; a sealed outdoor electronics cabinet (620 W at 50 °C walls in 35 °C air, two thirds
  by radiation); the district-heating pipe on a calm night. Checked by collocation, by a
  marching solution that starts from rough profiles and settles on the similarity solution
  (second order), and by comparing the laminar correlation with the exact coefficient
  (within 0.5 % for Pr from 10⁻³ to 10³).
- Chapter 10: boiling and condensation. Nusselt's film condensation on plates and tubes;
  an exact treatment of the film's subcooling, h_fg' = h_fg (1 + 27/40 Ja + ...), against
  Rohsenow's 0.68; the condenser of a combined heat and power plant (film, wall and cooling
  water resistances, the effect of bank depth); Rohsenow's nucleate boiling, the critical
  and minimum heat fluxes, Bromley's film boiling; an immersion heater and the boiling
  crisis, compared with Nukiyama's measurements. Checked by collocation and by a march
  down the plate from Nusselt's linear profile (second order, to 10⁻⁵).
- Chapter 11: heat exchangers. Energy balances, the overall coefficient, the LMTD, and the
  effectiveness–NTU relations for counterflow, parallel flow, 1-2 shell-and-tube and
  crossflow (Mason's exact series); the LMTD correction factor; a district-heating
  substation (counterflow needed for a 55 °C return; the primary flow at low supply
  temperatures); an air-heating coil whose coldest tubes freeze at 32 % of the design water
  flow while the mixed outlet is at 13 °C. Checked with a grid of small exchangers (second
  order) and by collocation of the 1-2 three-stream equations (10⁻¹³).
- Chapter 12: black bodies and real surfaces. Planck's law, Wien's law and the
  Stefan–Boltzmann constant from h, c and k; band fractions; emissivity, absorptivity and
  Kirchhoff's law; the sun as a 5772 K black body; a selective solar absorber in Astana
  (efficiency at 60 °C 15 % with black paint, 50 % with a selective coating) and the best
  cutoff wavelength. The check against quadrature found an error of up to 1.5 × 10⁻⁶ in the
  earlier two-series band-fraction formula, which is now accurate to 10⁻¹⁵.
- Chapter 13: view factors and enclosures. The view factor, reciprocity and summation;
  disks and rectangles, view-factor algebra and the crossed-string rule; radiosity networks
  for diffuse grey enclosures, two-surface enclosures and reradiating walls; an electric
  hardening furnace whose fibre-lined walls carry half the heat to the load (a 300 kg batch
  to 850 °C in 67 minutes at 40 kW); radiation shields in a vacuum furnace. Checked by
  Monte Carlo ray tracing (view factors, and the whole furnace: the one-zone network is
  3.2 % high, the banded network within 0.2 %) and Gauss–Legendre quadrature. Exercises
  include a blackbody reference cavity, whose mouth never exceeds an apparent emissivity of
  0.975 with ε = 0.9 paint, and the electronics cabinet of Chapter 9 from the inside.
- Chapter 14: molecular diffusion. Fick's law and the drift of the mixture; Fuller's
  correlation; Stefan's law and equimolar counter-diffusion; the Stefan tube (ethanol at
  25 °C: 10 mm in 17.5 h; 6 % error per kelvin); a loam drying after rain as a Stefan tube
  with the Millington–Quirk diffusivity (12 mm of water in a month of dry air). Checked
  by an exact similarity solution of the unsteady Stefan tube with a moving level, a
  moving-grid march (second order) and Massman's review of measured diffusivities.
- Chapter 15: diffusion with reaction. The Thiele modulus and the effectiveness factor of
  slabs, cylinders and spheres on Aris's length V/A; Bischoff's modulus for other orders;
  the Weisz–Prater test; the pellet size of a fixed-bed converter against a pressure-drop
  budget (Ergun); the carbonation of concrete as a shrinking core (25 mm of cover in 85
  years with assumed typical properties). Checked by finite differences (second order) and
  collocation of the pellet equations, and by the method of lines for a front of finite
  reaction rate against the sharp one-phase Stefan solution.
- Chapter 16: convective mass transfer. The mass transfer conductance and the Sherwood
  number; the analogy with heat transfer and the Chilton–Colburn factors; Ranz and
  Marshall for drops; blowing, the Spalding number and the film model; drying a solvent
  coating (ethanol, acetone, diethyl ether: for ether the low-rate formula is 39 % low and
  the film model 40 % high); the d² law, terminal fall and Wells's curve for a
  dust-suppression spray (drops under about 150 μm evaporate before falling 3 m in hot dry
  air). Checked by collocation of the boundary layer with Pr ≠ Sc and with blowing at the
  wall, and by the method of lines for the unsteady evaporation of a drop (the d² law is
  0.2 % off for water in air).
- Chapter 17: simultaneous heat and mass transfer. Moist air per kilogram of dry air;
  the adiabatic saturation temperature and the wet surface, the psychrometric ratio and
  Lewis's relation (a ventilated wet bulb reads 0.5 K low in dry air); the wetted-pad
  cooler; evaporative precooling of a 1 MW data hall on assumed design days in Astana and
  Shymkent, limited by the dew point of the supply rather than its temperature (a chiller
  needed for 3.25 and 10.75 hours with a 15 °C limit; 6 and 13 t of water a day). Checked
  against CoolProp's real-gas humid air (wet bulb within 0.04 K) and by marching the
  nonlinear pad equations (within 0.6 % of 1 − e^−NTU).
- Chapter 18: a project, from exchanger data to a cleaning schedule. A year of synthetic
  daily readings of a fouling crude preheat exchanger (labelled synthetic, generated from
  the Kern–Seaton law with operating changes and instrument noise); UA and the fouling
  resistance by inverse ε–NTU; the propagation of measurement errors; a least-squares fit of
  the law and how many days of data it needs; the cost of a cleaning cycle and its optimum
  (139 days for the assumed costs; cleaning never pays above about 66 thousand USD). Checked
  against the synthetic truth (exact inversion, error propagation, coverage of the fitted
  intervals over 200 synthetic years) and against SciPy's curve fit and minimiser.

### The `engtransport` package

- `properties`: air and liquid water from short formulas fitted to the reference
  equations (Lemmon et al. 2000; Lemmon and Jacobsen 2004; IAPWS-95 and the IAPWS
  transport formulations) by `scripts/fit_properties.py`, largest errors 0.11 % (air)
  and 0.22 % (water); replaces the earlier correlation matched to a textbook table.
- `numerics`: bisection and Newton's method with the history of iterates.
- `reference`: finite-volume model of an insulated pipe, steady and cooling; a general
  steady one-dimensional conduction solver (layers, generation, k(T), convection and
  radiation) by finite volumes and Newton's method; two-dimensional conduction in a plate
  fin by finite volumes and by the exact series solution.
- `fins`: thermal resistance of a plate-fin heat sink.
- `reference.laplace_2d_fd`: two-dimensional steady conduction by finite differences on a
  square grid, with circular holes (pipes) at fixed temperature, and fixed, insulated or
  convective edges.
- `reference.transient_conduction_fv`: transient conduction in a slab, cylinder or sphere
  by finite volumes and the theta method (Crank–Nicolson by default), with a heat transfer
  coefficient that may depend on the surface temperature.
- `transient`: the eigenvalues of the series solutions are cached, which makes repeated
  evaluation fast; new `contact_temperature`, `periodic_surface` and `neumann_freezing`
  (two-phase Neumann solution).
- `reference.enthalpy_1d_fv`: one-dimensional conduction with freezing and thawing by the
  enthalpy method, with layers and a surface temperature that changes in time.
- `convection_ext.pohlhausen_heat_transfer`: exact laminar flat-plate heat transfer for any
  Prandtl number; `reference.similarity_bvp` and `reference.boundary_layer_march` (with an
  optional unheated starting length).
- `convection_int.graetz_series` (eigenvalues by shooting) and `reference.graetz_march`
  (uniform wall temperature or flux, power-law velocity profiles).
- `natural_conv.free_convection_similarity` (two-parameter shooting), the laminar
  Churchill–Chu and Lloyd–Moran correlations; `reference.free_convection_bvp`,
  `free_convection_limits` and `free_convection_march`.
- `phase_change`: `film_condensation_subcooling` (Gauss–Legendre quadrature), Nusselt's
  coefficients for plates and tubes (`h_film_condensation`, `nusselt_tube_constant`),
  Rohsenow's nucleate boiling, the minimum heat flux and Bromley's film boiling;
  `reference.film_condensation_bvp` and `film_condensation_march`.
- `exchangers`: `effectiveness_crossflow_exact` and the flow type `crossflow_exact`, inverse
  NTU by bisection for any arrangement, `lmtd_correction_factor`; `rate_exchanger` no longer
  fails at a pinch; `reference.crossflow_cells` and `shell_and_tube_1_2_bvp`.
- `radiation`: `blackbody_band_fraction` rewritten (vectorised, 10⁻¹⁵), `band_fraction`,
  `wien_constant`, `stefan_boltzmann_from_constants`, `solar_spectrum`, `step_surface`; the
  radiation constants are now computed from the exact SI values of h, c and k; nominal solar
  values (IAU 2015).
- `radiation` (Chapter 13): `view_factor_parallel_rectangles`, `view_factor_perpendicular_rectangles`,
  `box_view_factors`, `box_zone_view_factors` (walls in horizontal bands, by view-factor
  algebra), `view_factor_crossed_strings` from end points; `solve_radiosity_enclosure` now
  takes prescribed heat flows as well as temperatures and returns surface temperatures;
  `radiation_shields`. `reference.mc_box_exchange`, `mc_cylinder_cavity`,
  `view_factor_gauss_rectangles` and `view_factor_gauss_strips`.
- `diffusion` (Chapter 14): `film_flux` and `film_profile` for any drift ratio,
  `stefan_tube_level`, `millington_quirk`; `properties.solvent_saturation` (ethanol,
  methanol, acetone, diethyl ether, fitted to CoolProp by `scripts/fit_properties.py`),
  `water_vapour_diffusivity_air` (Massman 1998), `saturation_pressure_ice`, Fuller volumes
  of acetone and diethyl ether; `reference.stefan_tube_exact` and `stefan_tube_march`.
- `diffusion` (Chapter 15): `effectiveness_first_order` (slab, cylinder via a continued
  fraction for I1/I0, sphere), `generalized_modulus`, `pellet_fd` (any shape, order >= 0.5,
  activity profiles), `observable_modulus`, `shrinking_core_depth` (with a surface
  resistance); `reference.pellet_bvp`, `one_phase_front_exact`, `reacting_front_march`.
- `convective_mass` (Chapter 16): `vapour_mass_fraction`, `spalding_number`,
  `film_blowing_factor`, `sherwood_flat_plate`, `sherwood_sphere_ranz_marshall`,
  `mass_conductance_from_heat`, `drag_coefficient_sphere` (Brown and Lawler 2003),
  `terminal_velocity`, `d2_law_rate`, `falling_drop`; `reference.species_similarity`
  (boundary layer with a species and blowing) and `drop_evaporation_transient`.
- `convective_mass` (Chapter 17): `humidity_ratio`, `relative_humidity`,
  `moist_air_enthalpy`, `adiabatic_saturation_temperature`, `wet_surface_temperature`,
  `evaporative_pad`; `reference.wetted_pad_march`; `scripts/fit_properties.py` now also
  tabulates CoolProp humid air (`_property_fits.HUMID_AIR_REF`) for the checks.
- `exchangers` (Chapter 18): `ua_from_temperatures`, `asymptotic_fouling`,
  `fit_asymptotic_fouling` (variable projection, golden section, covariance),
  `cleaning_cycle_cost`, `optimum_cleaning_interval`.
- `properties.saturated_water` (saturation pressure, latent heat and vapour density fitted
  to IAPWS-95; the IAPWS surface tension), `saturation_temperature_water` and
  `steam_properties` (superheated steam at 1 atm, 380–1300 K).
- `properties`: conductivity of uranium dioxide (Fink 2000); saturation pressure of water
  and dew point (Alduchov and Eskridge 1996).
- `plotting`: the plotting style of the figures (boxed axes); `panel_label` passes extra
  keyword arguments to `ax.text`.
- The earlier validation script is now part of the tests (`tests/reference_checks.py`), as
  in the companion books: the property checks compare with the reference equations, and
  checks that only restated a definition or repeated a formula were removed.
- Book layout as in the companion book on variational methods: package at the top level,
  Jupyter Book 2 site, introduction, notation and references pages,
  CC BY 4.0 for text and figures, institutional contact details.

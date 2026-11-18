<div align="center">

# Heat and mass transfer through worked problems

**An open book of executed notebooks on conduction, convection, radiation and diffusion — with every
result checked against an exact or independent numerical solution.**

[![tests](https://github.com/ktwyw/engtransport/actions/workflows/tests.yml/badge.svg)](https://github.com/ktwyw/engtransport/actions/workflows/tests.yml)
[![book](https://github.com/ktwyw/engtransport/actions/workflows/book.yml/badge.svg)](https://ktwyw.github.io/engtransport/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Notebooks](https://img.shields.io/badge/notebooks-36%20executed-brightgreen.svg)](#whats-inside)
[![Code: MIT](https://img.shields.io/badge/code-MIT-green.svg)](LICENSE)
[![Text: CC BY 4.0](https://img.shields.io/badge/text-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.md)

<img src="docs/images/pipe_cooling.gif" width="820" alt="Cross-section of an insulated district-heating pipe whose temperature falls after the pumps stop, beside a graph of the water temperature against time, which reaches 0 degrees Celsius after 38 hours">

<sub>A district-heating pipe in an Astana winter, after the pumps stop. Left: the temperature in the water and through the 50 mm of mineral wool. Right: the water temperature, which reaches 0 °C after 38 hours (Chapter 1). (<code>scripts/make_readme_pipe_animation.py</code>)</sub>

</div>

Heat and mass transfer decides how thick insulation must be, how large a heat exchanger must be, how
fast a quenched part cools, and how quickly a catalyst pellet or a drying film can work. Its laws are
few: Fourier's law, Newton's law of cooling, the Stefan–Boltzmann law and Fick's law, with balances of
energy and of species. The difficulty is in applying them, and in knowing how far the answer can be
trusted. This repository teaches both through engineering problems, many of them set in Kazakhstan. It
has **18 chapters in four parts** (conduction; convection and heat exchangers; radiation;
mass transfer), each an executed notebook with **worked solutions to its four exercises** in
[`solutions/`](solutions), a small tested Python package (**`engtransport`**) whose core needs NumPy
alone, and a list of **[suggested references](docs/references.md)**.

> **Start in 60 seconds:** open [Chapter 1](notebooks/part1_conduction/01_heat_balances.ipynb) here on
> GitHub (the notebooks are stored with their outputs): how much heat an insulated pipe loses in an
> Astana winter, and how long its water takes to reach 0 °C when the pumps stop.

> **Status:** version 0.1.0. All 18 chapters are written, each with executed sample solutions.

## Why this book?

- **One problem per chapter, followed to the end.** Each chapter starts from an engineering question,
  derives the model, solves it and answers the question with numbers.
- **Every result is checked.** The model is compared with a more complete calculation that keeps what
  the model leaves out: a finite-volume solution, a converged series, the reference equations for the
  properties. The size of the difference is printed.
- **Pictures beside the theory.** Three figures in every chapter: two beside the argument, one that
  collects the checks.
- **Derivations are written out.** No step is left to the reader. Each chapter ends with four exercises,
  and every exercise has a sample solution, derived and then checked by computation.
- **Honest about accuracy.** Each chapter ends with a section on what to trust: which inputs the answer
  is sensitive to, and which simplifications would fail elsewhere.

## What the checks show

<p align="center">
<img src="docs/images/checks_ch01.png" width="820" alt="Four checks of Chapter 1: convergence of three root finders, convergence of the finite-volume heat loss, and the cooling of the water from three models with their differences">
</p>

**(a)** Newton's method, successive substitution and bisection for the surface temperature of the pipe.
**(b)** The finite-volume heat loss converges to the same answer as the square of the cell size.
**(c, d)** The water cooling after the pumps stop: the closed-form model against a finite-volume model
that keeps the heat stored in the insulation; they differ by less than 1 K.

| Check | Model | Reference | Agreement |
|---|---|---|---|
| Surface temperature of an insulated pipe | Newton's method, 4 steps | the surface balance itself | 10⁻¹² K |
| Heat loss of an insulated pipe | resistance and surface balance | finite volumes through the insulation | error ∝ Δ*r*², 2 × 10⁻⁵ W/m with 320 cells |
| Cooling of the pipe after the pumps stop | lumped closed form, Eq. (1.11) | finite volumes with the heat stored in the insulation | within 0.9 K; time to 0 °C within 1.7 % |
| Energy balance of the cooling | heat lost | fall in stored heat | 3 × 10⁻⁷ |
| House wall: U-value and inner surface temperature | series resistances, Eq. (2.1) | finite volumes through the layers | exact (rounding error) |
| Fuel pellet with *k*(*T*) | Kirchhoff transform | finite volumes with *k*(*T*) and generation | error ∝ Δ*r*², 0.04 K with 160 cells |
| Fuel pellet, constant *k* at the surface temperature | Eq. (2.7) | Kirchhoff transform | centre 82 K too cold at 20 kW/m |
| Cable: parabola in the copper, logarithm in the PVC | Eqs. (2.3), (2.7) | finite volumes, copper and PVC | 5 × 10⁻⁶ K |
| Plate fin, 1-D theory, Eq. (3.3) | Biot number 2 × 10⁻⁴ | exact 2-D series, Eq. (3.9) | 6 × 10⁻⁵ |
| Balcony slab, 2-D | finite volumes | exact 2-D series | error ∝ Δ<sup>1.75</sup>, 3 × 10⁻⁴ with 80 cells across |
| Balcony slab, 1-D fin | Eq. (3.3) | exact 2-D series | heat 10 % too high (Bi = 0.9) |
| Buried pipe, shape factor | line source and image, Eq. (4.7) | finite differences, Shortley–Weller | 3 × 10⁻⁶ with 10 mm cells |
| Supply and return pipes | line sources, Eq. (4.9) | finite differences, exact soil conductances | heat losses within 0.3 % |
| Transient slab, cylinder, sphere | series, Eqs. (5.5)–(5.7) | Crank–Nicolson finite volumes | error ∝ Δ², 10⁻⁶ with 160 cells; energy to rounding |
| Lumped model | Eq. (5.2) | series | 3–6 % at Bi<sub>c</sub> = 0.1, 26–40 % at Bi<sub>c</sub> = 1 |
| Frost under a road | Neumann solution, Eq. (6.7) | enthalpy-method finite volumes | front within 1 mm with 10 mm cells, first order |
| Contact temperature, seasonal wave | Eqs. (6.3), (6.4) | finite volumes | 10⁻³ K |
| Laminar flat plate, heat transfer | quadrature (7.5) | collocation, and a marching solution of the boundary-layer equations | 4 × 10⁻⁶; marching to 10⁻⁴, first order |
| Laminar tube flow, thermal entry | Graetz series, Eq. (8.4) | marching finite volumes | 7 × 10⁻⁶ in the bulk temperature, second order |
| Uniform-flux tube, power-law fluids | Eq. (8.8) | marching finite volumes | 4 digits for n = 1, 0.5, 0.25 |
| Laminar free convection, vertical plate | similarity solution by shooting | collocation, and a march from rough profiles | 4 × 10⁻⁸ (Pr ≤ 2); march second order, to 6 × 10⁻⁶ |
| Churchill–Chu laminar correlation | exact laminar coefficient, Eq. (9.4) | the correlation, Eq. (9.7) | within 0.5 % for Pr from 10⁻³ to 10³ |
| Subcooled condensate film | exact solution by quadrature, Eq. (10.5) | collocation, and a march down the plate from Nusselt's profile | 2 × 10⁻¹³; march second order, to 10⁻⁵ |
| Blackbody band fraction | series of Eq. (12.4) | numerical quadrature | 10⁻¹⁵ (an earlier two-series formula: 1.5 × 10⁻⁶) |
| View factors of a box | Eqs. (13.5), (13.6) | Monte Carlo ray tracing; Gauss–Legendre quadrature | within the statistical error; 10⁻¹⁵ by quadrature for opposite faces, slow for adjacent ones |
| Electric furnace, reradiating walls | radiosity network, walls in bands | Monte Carlo ray tracing of the whole furnace | 3.2 % with one zone per wall, 0.2 % with bands |
| Stefan tube, falling level | quasi-steady, Eq. (14.8) | exact similarity solution; moving-grid march | 2 × 10⁻⁵ for ethanol; march second order, to 4 × 10⁻⁸ |
| Gas diffusivities | Fuller's correlation, Eq. (14.3) | Massman's review of measurements | within 3 % for water vapour and CO₂, 7–9 % low for methane |
| Effectiveness factor of a pellet | Eq. (15.5) | finite differences; collocation | second order, 4 × 10⁻⁶ (slab, 320 cells); 10⁻⁵ against collocation |
| Carbonation front | shrinking core, Eq. (15.10) | sharp one-phase solution; method of lines with a finite reaction rate | 3 × 10⁻⁷ for concrete; lag of 0.002 L at Da = 10⁴ |
| Heat–mass analogy, Pr ≠ Sc | Sh = 0.332 Re<sup>1/2</sup> Sc<sup>1/3</sup>, Eq. (16.3) | collocation of the similarity equations | within 1.1 % for 0.6 ≤ Sc ≤ 1.8, 3.4 % down to Sc = 0.3 |
| Blowing at an evaporating plate | film model ln(1 + B)/B, Eq. (16.7) | exact boundary layer with a wall velocity, Eq. (16.8) | within 5 % for B ≤ 0.2; 21–36 % high at B = 2 |
| d² law of an evaporating drop | quasi-steady, Eq. (16.9) | method of lines for unsteady diffusion with Stefan flow | lifetime 0.2 % shorter for water in air; ≈ 0.6 [ε ln(1 + B)]<sup>1/2</sup> |
| Psychrometrics of moist air | ideal-gas formulas, Eqs. (17.1)–(17.3) | CoolProp real-gas humid air (ASHRAE RP-1485) | wet bulb within 0.04 K; W 0.5–0.7 % low (enhancement factor) |
| Wetted-pad evaporative cooler | ε = 1 − e<sup>−NTU</sup>, Eq. (17.7) | march of the nonlinear heat and vapour equations | within 0.6 % (Lewis); up to 2 % higher with r = Le<sup>2/3</sup> |
| Fouling from plant data | inverse ε–NTU, Eqs. (18.2)–(18.4) | synthetic plant records from a known law | exact to 10⁻¹⁷ without noise; scatter as Eq. (18.5) predicts; 95 % intervals cover the truth in 96 % of 200 years |
| Cleaning interval | golden-section search of Eq. (18.7) | SciPy's bounded minimiser; the condition (18.8) | to 10⁻⁴ days; residual 2 × 10⁻⁶ |
| Crossflow exchanger, both fluids unmixed | Mason's exact series, Eq. (11.9) | a grid of small exchangers | second order, to 2 × 10⁻⁶ with 160 × 160 cells |
| 1-2 shell-and-tube exchanger | Eq. (11.8) | collocation of the three-stream equations | 7 × 10⁻¹⁴ |
| Properties of air (200–1000 K) | short fitted formulas | Lemmon et al. (2000), Lemmon and Jacobsen (2004) | within 0.11 % |
| Properties of liquid water (0–200 °C) | short fitted formulas | IAPWS-95 and the IAPWS transport formulations | within 0.22 % |

## Gallery

Each chapter has three figures: two beside the theory they illustrate, and one that collects the checks.

| | |
|---|---|
| <img src="docs/images/gallery_ch01.png" width="400" alt="Heat loss of an insulated pipe against insulation thickness, and the water temperature after the circulation stops for three insulation thicknesses"><br>Insulating a district-heating pipe: the heat loss, and the time the water takes to reach 0 °C when the pumps stop ([Chapter 1](notebooks/part1_conduction/01_heat_balances.ipynb)) | <img src="docs/images/gallery_ch02.png" width="400" alt="Temperature through a brick wall with and without mineral wool, and the inner surface temperature against insulation thickness with the dew point"><br>A wall in an Astana winter: where the temperature falls, and when the inner surface stays above the dew point ([Chapter 2](notebooks/part1_conduction/02_steady_conduction.ipynb)) |
| <img src="docs/images/gallery_ch03.png" width="400" alt="Thermal resistance of a plate-fin heat sink against the number and height of fins, the heat carried by a fin of fixed mass against its thickness, and temperatures in a concrete balcony slab"><br>A heat sink for a 150 W power module, the best shape for a fin, and a balcony slab that leaks as much heat as 17 m² of wall ([Chapter 3](notebooks/part1_conduction/03_fins.ipynb)) | <img src="docs/images/gallery_ch04.png" width="400" alt="Shape factor of a buried cylinder, heat loss against soil conductivity, soil temperatures around supply and return pipes, and their heat losses against spacing"><br>Buried district-heating pipes: how much the soil matters, and how the supply pipe warms the return ([Chapter 4](notebooks/part1_conduction/04_two_dimensional.ipynb)) |
| <img src="docs/images/gallery_ch05.png" width="400" alt="Quenching a steel bar: centre and surface temperatures in oil and water, temperatures across the bar, cooling time against diameter, and against the heat transfer coefficient"><br>Quenching a steel bar: why size matters more than the quenchant ([Chapter 5](notebooks/part1_conduction/05_transient.ipynb)) | <img src="docs/images/gallery_ch06.png" width="400" alt="The seasonal temperature wave in the ground, temperature against time at several depths, frost depth under a road against time with a measured point, and temperature profiles in freezing ground"><br>The seasons in the ground and the frost under a road near Astana ([Chapter 6](notebooks/part1_conduction/06_semi_infinite.ipynb)) |
| <img src="docs/images/gallery_ch07.png" width="400" alt="Local heat transfer along a flat plate, average Nusselt number of laminar and turbulent plates, heat transfer coefficient of the Chapter 1 pipe against wind speed, and its heat loss against wind"><br>From wind speed to heat transfer coefficient: the pipe of Chapter 1 in a cross wind ([Chapter 7](notebooks/part2_convection/07_boundary_layers.ipynb)) | <img src="docs/images/gallery_ch08.png" width="400" alt="Bulk temperature of oil along a heater tube, tube length against flow, inside heat transfer coefficient of the district-heating pipe against velocity, and water temperature along insulated and bare mains"><br>A heater for viscous oil, and the water in the district-heating main ([Chapter 8](notebooks/part2_convection/08_internal_flow.ipynb)) |
| <img src="docs/images/gallery_ch09.png" width="400" alt="Heat flows from a sealed electronics cabinet against wall temperature, allowable power against air temperature, heat transfer coefficients on the cabinet, and heat loss of the district-heating pipe against wind speed"><br>A sealed electronics cabinet in summer, and the pipe on a calm night ([Chapter 9](notebooks/part2_convection/09_natural_convection.ipynb)) | <img src="docs/images/gallery_ch10.png" width="400" alt="Heat taken up per metre of condenser tube against cooling-water temperature, thermal resistances of condenser tubes, the boiling curve of water with Nukiyama's maxima, and the temperature jump of a heater at the critical heat flux"><br>A power-plant condenser and the boiling crisis of a heater ([Chapter 10](notebooks/part2_convection/10_boiling_condensation.ipynb)) |
| <img src="docs/images/gallery_ch11.png" width="400" alt="Primary flow and return temperature of a district-heating substation against network supply temperature, the water temperature field of an air-heating coil with its freezing corner, and coil temperatures against water flow"><br>A district-heating substation and the freezing corner of an air heater ([Chapter 11](notebooks/part2_convection/11_heat_exchangers.ipynb)) | <img src="docs/images/gallery_ch12.png" width="400" alt="Solar and plate spectra with a selective coating, net absorbed flux against cutoff wavelength, absorber efficiency for black paint and a selective coating, and stagnation temperatures"><br>A selective solar absorber ([Chapter 12](notebooks/part3_radiation/12_radiation_surfaces.ipynb)) |
| <img src="docs/images/gallery_ch13.png" width="400" alt="Temperatures of the heater, walls and load while a furnace batch heats, heat to the load for three kinds of wall, heating time against installed power, and the heat loss of a vacuum furnace hot zone against the number of radiation shields"><br>An electric furnace whose walls carry half the heat, and the radiation shields of a vacuum furnace ([Chapter 13](notebooks/part3_radiation/13_view_factors_enclosures.ipynb)) | <img src="docs/images/gallery_ch14.png" width="400" alt="Level of ethanol in a Stefan tube against time, depth of the dry layer in a soil after rain, daily evaporation, and water lost in a month against air humidity"><br>A Stefan tube and a field drying after rain ([Chapter 14](notebooks/part4_mass_transfer/14_molecular_diffusion.ipynb)) |
| <img src="docs/images/gallery_ch15.png" width="400" alt="Bed length and effectiveness factor against catalyst pellet diameter, pressure drop against pellet diameter with a budget, carbonation depth of concrete against age, and time for the front to reach the steel against cover"><br>Sizing catalyst pellets, and the carbonation of concrete ([Chapter 15](notebooks/part4_mass_transfer/15_diffusion_reaction.ipynb)) | <img src="docs/images/gallery_ch16.png" width="400" alt="Square of the diameter of evaporating water drops against time, their fall against time, the fall before evaporation against initial size on a summer and a mild day, and the smallest drop that falls 3 m against humidity"><br>Which drops of a dust-suppression spray reach the road ([Chapter 16](notebooks/part4_mass_transfer/16_convective_mass_transfer.ipynb)) |
| <img src="docs/images/gallery_ch17.png" width="400" alt="Outdoor, wet-bulb, pad-outlet and supply temperatures through a summer day for a data hall in Astana and in Shymkent, supply humidity ratio with dew-point limits, and water evaporated per megawatt"><br>Evaporative precooling of a data centre in Kazakhstan, limited by the water it adds ([Chapter 17](notebooks/part4_mass_transfer/17_simultaneous_heat_mass.ipynb)) | <img src="docs/images/gallery_ch18.png" width="400" alt="Average cost per day against cleaning interval for three cleaning costs, the optimum where the loss rate meets the average cost, fitted fouling time constants against days of data, and the extra cost of the chosen interval"><br>When to clean a fouling exchanger, from a year of synthetic plant data ([Chapter 18](notebooks/part4_mass_transfer/18_cleaning_schedule.ipynb)) |

## Installation

```bash
git clone https://github.com/ktwyw/engtransport.git
cd engtransport
pip install -e ".[book]"
jupyter lab
```

The notebooks also run without installation: each one adds the repository root to the path if the
package is not installed. The core of the package needs NumPy alone; SciPy is used for the reference solutions in the
checks, and Matplotlib for the figures.

## Quick start

```python
import math
from engtransport import numerics, properties
from engtransport.constants import SIGMA_SB

# surface temperature of an insulated pipe (Chapter 1): conduction in = convection + radiation out
r1, r2, k, h, eps = 0.05715, 0.10715, 0.045, 25.0, 0.9
T_w, T_air = 383.15, 238.15
R = math.log(r2 / r1) / (2 * math.pi * k)
f = lambda Ts: (T_w - Ts) / R - 2 * math.pi * r2 * (h * (Ts - T_air) + eps * SIGMA_SB * (Ts**4 - T_air**4))
df = lambda Ts: -1 / R - 2 * math.pi * r2 * (h + 4 * eps * SIGMA_SB * Ts**3)
Ts, steps = numerics.newton(f, df, x0=T_w)
print(Ts - 273.15, (T_w - Ts) / R)          # -31.6 C, 63.7 W per metre

properties.dry_air_properties(300.0)["Pr"]   # 0.707
properties.water_properties(353.15)["cp"]    # 4196 J/(kg K)
```

## What's inside

**Part I. Conduction**

| Chapter | Topics | Checked against |
|---|---|---|
| [1. Heat balances and the three modes of heat transfer](notebooks/part1_conduction/01_heat_balances.ipynb) | Fourier, Newton and Stefan–Boltzmann; the energy balance; a district-heating pipe in winter; Newton's method; resistances; lumped cooling | finite volumes through the insulation, steady and in time; energy balance; reference property equations |
| [2. Steady conduction and thermal resistance](notebooks/part1_conduction/02_steady_conduction.ipynb) | a house wall in winter and condensation; the critical radius and the ampacity of a cable; heat generation; a nuclear fuel pellet and the Kirchhoff transform | finite volumes with layers, generation and *k*(*T*); the Kirchhoff solution |
| [3. Fins and extended surfaces](notebooks/part1_conduction/03_fins.ipynb) | a heat sink for a power module; the best use of metal; a balcony slab as a thermal bridge | the exact 2-D series and 2-D finite volumes |
| [4. Two-dimensional steady conduction](notebooks/part1_conduction/04_two_dimensional.ipynb) | a buried district-heating pipe; supply and return pipes in one trench; how far a disturbance reaches | finite differences around circular pipes |
| [5. Transient conduction](notebooks/part1_conduction/05_transient.ipynb) | quenching a steel bar in oil and in water; when the lumped model works | Crank–Nicolson finite volumes |
| [6. Semi-infinite solids and phase change](notebooks/part1_conduction/06_semi_infinite.ipynb) | a steel railing at −30 °C; the seasons in the ground; frost under a road | enthalpy-method finite volumes |

**Part II. Convection and heat exchangers**

| Chapter | Topics | Checked against |
|---|---|---|
| [7. Boundary layers and external flow](notebooks/part2_convection/07_boundary_layers.ipynb) | from wind speed to *h*: the pipe of Chapter 1 in a cross wind; the Blasius and Pohlhausen solutions | collocation; a marching boundary-layer solver |
| [8. Internal flow and the Graetz problem](notebooks/part2_convection/08_internal_flow.ipynb) | a heater for viscous oil; the district-heating main; shear-thinning fluids | marching finite volumes |
| [9. Natural convection](notebooks/part2_convection/09_natural_convection.ipynb) | a sealed electronics cabinet in summer; the district-heating pipe on a calm night | collocation; a march from rough profiles |
| [10. Boiling and condensation](notebooks/part2_convection/10_boiling_condensation.ipynb) | a power-plant condenser; an immersion heater and the boiling crisis | collocation; a march down the condensate film |
| [11. Heat exchangers](notebooks/part2_convection/11_heat_exchangers.ipynb) | a district-heating substation; the freezing corner of an air-heating coil | a grid of small exchangers; collocation |

**Part III. Radiation**

| Chapter | Topics | Checked against |
|---|---|---|
| [12. Black bodies and real surfaces](notebooks/part3_radiation/12_radiation_surfaces.ipynb) | a selective solar absorber; the best cutoff wavelength | numerical quadrature of Planck's law |
| [13. View factors and enclosures](notebooks/part3_radiation/13_view_factors_enclosures.ipynb) | an electric furnace with reradiating walls; radiation shields in a vacuum furnace | Monte Carlo ray tracing; numerical integration of view factors |

**Part IV. Mass transfer**

| Chapter | Topics | Checked against |
|---|---|---|
| [14. Molecular diffusion](notebooks/part4_mass_transfer/14_molecular_diffusion.ipynb) | evaporation through a stagnant gas; the Stefan tube; a field drying after rain | an exact similarity solution and a moving-grid march of the unsteady Stefan tube; reviewed diffusivities |
| [15. Diffusion with reaction](notebooks/part4_mass_transfer/15_diffusion_reaction.ipynb) | the pellet size of a fixed-bed converter; carbonation of concrete | finite differences and collocation of the pellet equations; a reacting front of finite rate |
| [16. Convective mass transfer](notebooks/part4_mass_transfer/16_convective_mass_transfer.ipynb) | the heat–mass analogy; blowing; drying a solvent coating; which spray drops reach the ground | the exact boundary layer with Pr ≠ Sc and with blowing; unsteady evaporation of a drop without the quasi-steady assumption |
| [17. Simultaneous heat and mass transfer](notebooks/part4_mass_transfer/17_simultaneous_heat_mass.ipynb) | psychrometrics; the wet-bulb temperature and Lewis's relation; evaporative precooling of a data centre in Astana and Shymkent | a real-gas humid-air model; the nonlinear pad equations marched |
| [18. A project: from exchanger data to a cleaning schedule](notebooks/part4_mass_transfer/18_cleaning_schedule.ipynb) | fouling resistance from a year of synthetic plant readings; the Kern–Seaton law; propagation of measurement errors; the cleaning interval and how much data it needs | the synthetic truth; Monte Carlo over 200 synthetic years; SciPy's curve fit and minimiser |

Sample solutions: [Chapter 1](solutions/part1_conduction/01_solutions.ipynb) ·
[2](solutions/part1_conduction/02_solutions.ipynb) ·
[3](solutions/part1_conduction/03_solutions.ipynb) ·
[4](solutions/part1_conduction/04_solutions.ipynb) ·
[5](solutions/part1_conduction/05_solutions.ipynb) ·
[6](solutions/part1_conduction/06_solutions.ipynb) ·
[7](solutions/part2_convection/07_solutions.ipynb) ·
[8](solutions/part2_convection/08_solutions.ipynb) ·
[9](solutions/part2_convection/09_solutions.ipynb) ·
[10](solutions/part2_convection/10_solutions.ipynb) ·
[11](solutions/part2_convection/11_solutions.ipynb) ·
[12](solutions/part3_radiation/12_solutions.ipynb) ·
[13](solutions/part3_radiation/13_solutions.ipynb) ·
[14](solutions/part4_mass_transfer/14_solutions.ipynb) ·
[15](solutions/part4_mass_transfer/15_solutions.ipynb) ·
[16](solutions/part4_mass_transfer/16_solutions.ipynb) ·
[17](solutions/part4_mass_transfer/17_solutions.ipynb) ·
[18](solutions/part4_mass_transfer/18_solutions.ipynb).

## The `engtransport` package

| Module | Purpose |
|---|---|
| `numerics` | bisection and Newton's method, written out, with the history of iterates |
| `properties` | air (200–1000 K), liquid water and saturated water (0–200 °C) and steam (380–1300 K) from formulas fitted to the reference equations; the IAPWS surface tension; uranium dioxide; saturation pressure of water (and over ice) and dew point; vapour pressure and liquid density of ethanol, methanol, acetone and diethyl ether; typical solids; gas diffusivities by Fuller's correlation and for water vapour from Massman's review; liquid diffusivities |
| `reference` | more complete solutions used in the checks (SciPy): finite volumes for an insulated pipe, steady and cooling; steady one-dimensional conduction with layers, generation, *k*(*T*), convection and radiation; two-dimensional conduction in a plate fin, by finite volumes and by the exact series; two-dimensional conduction around pipes by finite differences; transient conduction by Crank–Nicolson finite volumes; freezing and thawing by the enthalpy method; laminar boundary layers by collocation and by marching; the thermal entry region of tubes by marching; free convection on a vertical plate by collocation and by marching; the subcooled condensate film by collocation and by marching; crossflow exchangers by a grid of small exchangers; the 1-2 shell-and-tube exchanger by collocation; view factors by Gauss–Legendre quadrature; Monte Carlo ray tracing in a box and in a cylindrical cavity; the unsteady Stefan tube, exactly and by a moving-grid march; catalyst pellets by collocation; a reacting front of finite rate by the method of lines; the laminar boundary layer with a species and blowing by collocation; the unsteady evaporation of a drop; air through a wetted pad |
| `constants` | physical constants (CODATA 2018) |
| `plotting` | the plotting style of the figures (Matplotlib) |
| `fins` | one-dimensional fins: tip conditions, efficiency and effectiveness; the resistance of a plate-fin heat sink |
| `convection_ext` | external flow: Blasius by shooting, exact laminar heat transfer for any Pr, correlations for plates, cylinders and spheres |
| `convection_int` | internal flow: the Graetz series, power-law fluids, Gnielinski's correlation |
| `natural_conv` | natural convection: Ostrach's similarity solution by shooting, correlations for plates and cylinders |
| `phase_change` | film condensation with the subcooling of the film, nucleate boiling, the critical and minimum heat fluxes, film boiling |
| `exchangers` | LMTD, effectiveness–NTU for counterflow, parallel, shell-and-tube and crossflow (exact series), inverse NTU, correction factor, rating; UA and fouling resistance from operating data, the Kern–Seaton law and its least-squares fit, the cost of a cleaning cycle and its optimum |
| `radiation` | Planck's law, band fractions, Wien and Stefan–Boltzmann constants from h, c and k, selective (step) surfaces; view factors of disks and rectangles, boxes with banded walls, the crossed-string rule; radiosity networks with prescribed temperatures or heat flows; radiation shields |
| `diffusion` | fluxes through films with drift (Stefan's law, equimolar counter-diffusion, surface reactions) and their profiles; the Stefan tube; the Millington–Quirk diffusivity of porous media; effectiveness factors of slabs, cylinders and spheres, Bischoff's generalized modulus, pellets with any order or activity profile by finite differences, the Weisz–Prater modulus, shrinking-core fronts |
| `convective_mass` | mass fractions, the Spalding number and the film blowing factor; Sherwood numbers of plates and drops; the Chilton–Colburn analogy; sphere drag and terminal velocity; the d² law and drops falling while they evaporate; humidity ratio, enthalpy, adiabatic saturation and wet-surface temperatures of moist air; the wetted-pad cooler |
| `conduction` | resistances of walls and shells, the critical radius, a wall with *k*(*T*), a rectangular plate by series and by finite differences |
| `transient` | lumped capacitance; series for the slab, cylinder and sphere with cached eigenvalues; semi-infinite solids, the contact temperature, the periodic surface wave; Neumann's freezing solution |

Run the tests with `pytest -q`; they include the comparisons of the package with exact solutions,
limits and reference data in `tests/reference_checks.py`.

## Repository layout

```
engtransport/                  the Python package
notebooks/part1_conduction/    Chapters 1-6, executed
notebooks/part2_convection/    Chapters 7-11
notebooks/part3_radiation/     Chapters 12-13
notebooks/part4_mass_transfer/ Chapters 14-18
solutions/                     sample solutions to the exercises, executed
tests/                         tests of the package
docs/                          introduction, notation and references of the book; images
scripts/                       property fits and the figures of this page
myst.yml                       Jupyter Book 2 configuration
```

## Suggested references

The notebooks are self-contained and do not follow any book. For further study, see the list in
[docs/references.md](docs/references.md), which starts with T. L. Bergman, A. S. Lavine, F. P.
Incropera and D. P. DeWitt, *Fundamentals of Heat and Mass Transfer*, 8th ed. (Wiley, 2017), and J. H.
Lienhard IV and J. H. Lienhard V, *A Heat Transfer Textbook*, 5th ed. (Dover, 2019), which the authors
make freely available.

## Companion books

- [onsager-variational-methods](https://github.com/ktwyw/onsager-variational-methods): variational
  methods in soft matter.
- [error-analysis](https://github.com/ktwyw/error-analysis): evaluating, propagating and reporting
  uncertainty.
- [math-perturbation-methods](https://github.com/ktwyw/math-perturbation-methods) and
  [math-asymptotic-methods](https://github.com/ktwyw/math-asymptotic-methods).

Also by the author: [engmath](https://github.com/ktwyw/engmath), [engstat](https://github.com/ktwyw/engstat),
[engnumeric](https://github.com/ktwyw/engnumeric), [engmolines](https://github.com/ktwyw/engmolines),
[fluidmech](https://github.com/ktwyw/fluidmech), [engrheo](https://github.com/ktwyw/engrheo),
[engthermo](https://github.com/ktwyw/engthermo), [engsep](https://github.com/ktwyw/engsep),
[engreact](https://github.com/ktwyw/engreact), [engbalance](https://github.com/ktwyw/engbalance),
[engcolloid](https://github.com/ktwyw/engcolloid), [polyphys](https://github.com/ktwyw/polyphys),
[engdiscrete](https://github.com/ktwyw/engdiscrete), [engregress](https://github.com/ktwyw/engregress),
[engstoch](https://github.com/ktwyw/engstoch) and [englincontrol](https://github.com/ktwyw/englincontrol).

## Scope and limitations

- Chapter 1 takes the heat-transfer coefficient as given (Chapter 7 computes it from the wind speed),
  treats the cladding as a grey surface seeing surroundings at one temperature, and describes the
  water in a stopped pipe by one temperature. Section 1.7 shows how sensitive the answers are to each
  of these.
- Chapter 2 takes typical, dry material properties for the wall (real ones vary by 10–20 % and rise
  with moisture), ignores thermal bridges (Chapter 4), uses a combined surface coefficient for the
  cable, and the conductivity of fresh fuel for the pellet.
- Chapter 3 takes one heat transfer coefficient for every surface of the heat sink and leaves out
  the thermal paste and spreading in the base; the root temperature of the balcony slab is an
  assumption, giving an upper estimate.
- Chapter 4 holds the ground surface at a fixed temperature (Exercise 2 adds a surface coefficient)
  and takes the soil as uniform and steady; line sources for two pipes are good for the heat losses
  of insulated pipes but not for the temperatures around bare pipes close together.
- Chapter 5 takes constant steel properties and an average heat transfer coefficient for the quench
  (Exercise 4 lets it change with boiling), and leaves out the latent heat of the steel's transformations.
- Chapter 6 takes assumed soil properties and a constant (or sinusoidal) surface temperature, and
  leaves out snow (except in Exercise 2), unfrozen water below 0 °C and frost heave.
- Chapter 7 uses an empirical correlation (±20 %) for the cylinder, and leaves out free-stream
  turbulence, roughness and the direction of the wind.
- Chapter 8 takes constant properties (the viscosity of oil changes several-fold across a heater)
  and a developed velocity profile in laminar flow; turbulent correlations hold for smooth pipes.
- Chapter 9 takes the cabinet walls as isothermal and the surroundings at the air temperature
  (Exercise 1 adds the sun), and uses correlations good to ±10–20 %.
- Chapter 10 takes laminar condensate films of pure vapour (no non-condensable gas) and
  pool boiling correlations that are uncertain by tens of per cent (by a factor of two for
  the nucleate heat flux).
- Chapter 11 takes a constant overall coefficient and ideal flow arrangements; the plate
  coefficient of the substation is an assumed typical value.
- Chapter 12 treats sunlight as a 5772 K black body and coatings as steps in emissivity, and
  takes an assumed sky temperature.
- Chapter 13 takes all surfaces as grey and diffuse with uniform temperature, the furnace load
  as a flat plate at one temperature, and leaves out the heat stored in the lining; emissivities
  of the elements, lining and shields are assumed typical values.
- Chapter 14 takes the gas phase as isothermal and still, with Fick's law for two species; the soil
  model has a sharp drying front and no capillary first stage, and its porosity and water content are
  assumed values.
- Chapter 15 takes isothermal pellets with first-order kinetics and no film around them; the rate
  constant, diffusivities and the carbonation properties are assumed typical values.
- Chapter 16 takes constant gas properties, a coating held at a set temperature, and isolated drops
  falling at their terminal velocity in still air; the drying and spray conditions are assumed.
- Chapter 17 treats moist air as an ideal-gas mixture with constant heat capacities, the water film
  of a pad at one local temperature, and idealised design days; pad NTU and supply limits are assumed.
- Chapter 18 uses synthetic plant data from an assumed asymptotic fouling law, assumed noise levels
  and assumed costs; real data need the same steps with more care about drift and the clean coefficient.
- The property formulas are fitted for dry air at atmospheric pressure from 200 to 1000 K and for
  saturated liquid water from 0 to 200 °C; outside these ranges the functions refuse to answer.

## Roadmap

Version 0.1.0 is the complete book. Planned next: more exercises, hourly weather data in place of
the design days of Chapter 17, and real plant data for Chapter 18 where they can be shared.
Corrections and suggestions are welcome through the
[issue tracker](https://github.com/ktwyw/engtransport/issues).

## Author

**Yanwei Wang**, Department of Chemical and Materials Engineering, School of Engineering, Nazarbayev
University, Astana, Kazakhstan · yanwei.wang@nu.edu.kz ·
ORCID [0000-0002-8488-9833](https://orcid.org/0000-0002-8488-9833)

## Citing

Please cite the material using the metadata in [`CITATION.cff`](CITATION.cff).

## License

Code (`engtransport/`, `tests/`, `scripts/` and the code cells of the notebooks):
[MIT](LICENSE). Text, figures and notebook outputs: [CC BY 4.0](DATA_LICENSE.md).

# Heat and mass transfer through worked problems

*Conduction, convection, radiation and diffusion, taught through engineering problems,
with every result checked against an exact or independent numerical solution.*

Heat and mass transfer is the part of engineering that decides how thick insulation must
be, how large a heat exchanger must be, how fast a part cools after quenching, and how
quickly a catalyst pellet or a drying film can work. Its laws are few: Fourier's law,
Newton's law of cooling, the Stefan–Boltzmann law and Fick's law, together with
balances of energy and of species. The difficulty is in applying them: in choosing the
control volume, the simplifications and the method of solution, and in knowing how far
the answer can be trusted.

Each chapter is an executed Jupyter notebook built around one engineering problem, many
of them set in Kazakhstan: a district-heating pipe in an Astana winter, frost in the
ground under a road, the cooling of a data centre in the steppe. The chapter derives the
model, solves it, and then checks it against a more complete calculation that keeps what
the model leaves out: a finite-volume solution, a series that converges, a reference
equation for the properties. The notebooks share a small Python package, `engtransport`,
written to be read as much as to be used; its core needs NumPy alone.

The level is that of an undergraduate course in heat and mass transfer for chemical,
materials, mechanical and energy engineers, and the chapters assume calculus, ordinary
differential equations and a first course in thermodynamics. Each chapter has three
numbered figures and ends with four exercises, and a sample solution to every exercise
is given in the *Solutions* part of the book. In the tables below, the *core* chapters
make up a first course and are best read in order within each part; the last chapter is a
*project* that puts the tools to work on plant data. Symbols are collected in
[Notation and conventions](notation.md), and books and articles for further study in
[Suggested references](references.md).

## Part I. Conduction

| Chapter | Track | Topics |
|---|:-:|---|
| [1. Heat balances and the three modes of heat transfer](../notebooks/part1_conduction/01_heat_balances.ipynb) | core | Fourier, Newton and Stefan–Boltzmann; the energy balance; a district-heating pipe in winter; Newton's method; resistances; lumped cooling |
| [2. Steady conduction and thermal resistance](../notebooks/part1_conduction/02_steady_conduction.ipynb) | core | a house wall in winter and condensation; contact resistance; the critical radius and the ampacity of a cable; heat generation; a nuclear fuel pellet and the Kirchhoff transform |
| [3. Fins and extended surfaces](../notebooks/part1_conduction/03_fins.ipynb) | core | the fin equation; efficiency and effectiveness; a heat sink for a power module; the best use of metal; a balcony slab and when a fin is one-dimensional |
| [4. Two-dimensional steady conduction](../notebooks/part1_conduction/04_two_dimensional.ipynb) | core | separation of variables and superposition; line sources and images; a buried district-heating pipe; supply and return pipes in one trench; finite differences around pipes |
| [5. Transient conduction](../notebooks/part1_conduction/05_transient.ipynb) | core | the Biot number and the lumped model; series solutions for the slab, cylinder and sphere; quenching a steel bar in oil and water; Crank–Nicolson |
| [6. Semi-infinite solids and phase change](../notebooks/part1_conduction/06_semi_infinite.ipynb) | core | the error-function solution; contact temperature and a steel railing at −30 °C; the seasons in the ground; frost under a road and the Stefan problem; the enthalpy method |

## Part II. Convection and heat exchangers

| Chapter | Track | Topics |
|---|:-:|---|
| [7. Boundary layers and external flow](../notebooks/part2_convection/07_boundary_layers.ipynb) | core | scaling; the Blasius and Pohlhausen solutions; turbulent plates; a cylinder in a cross wind and the pipe of Chapter 1; a marching boundary-layer solver |
| [8. Internal flow and the Graetz problem](../notebooks/part2_convection/08_internal_flow.ipynb) | core | the bulk temperature; fully developed flow; the Graetz series and the entry region; a heater for viscous oil; turbulent pipe flow and the district-heating main; shear-thinning fluids |
| [9. Natural convection](../notebooks/part2_convection/09_natural_convection.ipynb) | core | buoyancy and the Rayleigh number; Ostrach's similarity solution for a vertical plate; correlations for plates and cylinders; a sealed electronics cabinet; the district-heating pipe on a calm night |
| [10. Boiling and condensation](../notebooks/part2_convection/10_boiling_condensation.ipynb) | core | Nusselt's film condensation and the subcooling of the film; a power-plant condenser; nucleate boiling, the critical heat flux and film boiling; the boiling crisis of an electric heater |
| [11. Heat exchangers](../notebooks/part2_convection/11_heat_exchangers.ipynb) | core | the log-mean temperature difference; effectiveness and NTU for counterflow, shell-and-tube and crossflow; a district-heating substation; the freezing corner of an air-heating coil |

## Part III. Radiation

| Chapter | Track | Topics |
|---|:-:|---|
| [12. Black bodies and real surfaces](../notebooks/part3_radiation/12_radiation_surfaces.ipynb) | core | Planck's law, Wien's law and the Stefan–Boltzmann law; band fractions; emissivity, absorptivity and Kirchhoff's law; a selective solar absorber |
| [13. View factors and enclosures](../notebooks/part3_radiation/13_view_factors_enclosures.ipynb) | core | view factors, reciprocity and summation; view-factor algebra and crossed strings; radiosity networks; reradiating walls; an electric furnace; radiation shields |

## Part IV. Mass transfer

| Chapter | Track | Topics |
|---|:-:|---|
| [14. Molecular diffusion](../notebooks/part4_mass_transfer/14_molecular_diffusion.ipynb) | core | Fick's law and the drift of the mixture; gas diffusivities; diffusion through a stagnant gas; the Stefan tube; a soil drying after rain |
| [15. Diffusion with reaction](../notebooks/part4_mass_transfer/15_diffusion_reaction.ipynb) | core | the Thiele modulus and the effectiveness factor; shapes and orders; the Weisz–Prater test; pellet size of a converter; carbonation of concrete |
| [16. Convective mass transfer](../notebooks/part4_mass_transfer/16_convective_mass_transfer.ipynb) | core | mass-transfer coefficients and the heat–mass analogy; blowing and the Spalding number; drying a solvent coating; the d² law and Wells's curve for spray drops |
| [17. Simultaneous heat and mass transfer](../notebooks/part4_mass_transfer/17_simultaneous_heat_mass.ipynb) | core | psychrometrics of moist air; the wet-bulb and adiabatic saturation temperatures; Lewis's relation; wetted-pad coolers; evaporative precooling of a data centre |
| [18. A project: from exchanger data to a cleaning schedule](../notebooks/part4_mass_transfer/18_cleaning_schedule.ipynb) | project | fouling resistance from synthetic plant data by inverse rating; the Kern–Seaton law; propagation of measurement errors; when to clean, and how much data the decision needs |

## Where to start

| If you are… | Read |
|---|---|
| new to the subject | Chapters 1, 2 and 5 |
| designing insulation | Chapters 1, 2 and 4 |
| sizing a heat exchanger | Chapters 7, 8 and 11 |
| working with furnaces or solar collectors | Chapters 12 and 13 |
| working with reactors, dryers or cooling towers | Chapters 14 to 17 |

## Companion books

[*Variational methods in soft matter*](https://github.com/ktwyw/onsager-variational-methods),
[*Error analysis and propagation*](https://github.com/ktwyw/error-analysis),
[*Perturbation methods for differential equations*](https://github.com/ktwyw/math-perturbation-methods)
and [*Asymptotic methods: expansions, integrals and sums*](https://github.com/ktwyw/math-asymptotic-methods).

## Licence and citation

Code: MIT. Text and figures: CC BY 4.0. Please cite the material using the
metadata in `CITATION.cff` in the repository.

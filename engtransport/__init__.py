"""engtransport: heat and mass transfer in Python, companion package to the book.

The modules need NumPy alone, except `reference` (SciPy, for the checks) and
`plotting` (Matplotlib, imported explicitly: `from engtransport.plotting import use_style`).
"""

from . import (
    conduction,
    constants,
    convection_ext,
    convection_int,
    convective_mass,
    diffusion,
    exchangers,
    fins,
    natural_conv,
    numerics,
    phase_change,
    properties,
    radiation,
    reference,
    transient,
)

__all__ = [
    "conduction",
    "constants",
    "convection_ext",
    "convection_int",
    "convective_mass",
    "diffusion",
    "exchangers",
    "fins",
    "natural_conv",
    "numerics",
    "phase_change",
    "properties",
    "radiation",
    "reference",
    "transient",
]
__version__ = "0.1.0"

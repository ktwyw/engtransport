# Licence for text, figures and data

Copyright (c) 2026 Yanwei Wang

The following material in this repository is licensed under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):

- the explanatory text (Markdown cells) of the notebooks in `notebooks/` and `solutions/`;
- the figures and other outputs stored in those notebooks;
- `README.md` and the pages in `docs/`.

You are free to share and adapt this material for any purpose, provided that
you give appropriate credit, provide a link to the licence, and indicate if
changes were made. A suggested form of credit is given in `CITATION.cff`.

The source code (`engtransport/`, `tests/`, `scripts/`, and the code cells of the
notebooks) is licensed separately under the MIT licence; see `LICENSE`.

All numerical results in the notebooks are computed within the notebooks
themselves. The property formulas in `engtransport/_property_fits.py` are fitted
by `scripts/fit_properties.py` to values computed with CoolProp from the published
reference equations for air, water and four solvents, which are cited in the docstrings
and in `docs/references.md`; the same script tabulates a few states of moist air for the
checks of Chapter 17. Other material properties are typical values, quoted for
illustration, and are marked as assumed where they are. The plant record of Chapter 18
is synthetic and is generated in the notebook. No third-party data sets are included in
this release.

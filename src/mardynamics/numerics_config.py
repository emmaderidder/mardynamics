"""Shared numerical constants of the estimator and of the test machinery.

These constants belong to the mathematics of the methodology: the stability
boundary the estimator projects onto, the simulation size of the
Nyblom-Hansen limit law, and the number of draws of the two resampling
checks. They are shared by the empirical pipeline and by the simulation
study, and none of them refers to a country, a variable, a date or a path.

The empirical constants (panel composition, sample split, event annotations
and input/output directories) live in :mod:`mardynamics.config`. A pure
numerical function reads this module, not that one. Anything that needs a
path or a label takes it as an argument.
"""

from __future__ import annotations

# --- Estimator -------------------------------------------------------------
# Equation (4): the fitted pair is kept inside rho(A) rho(B) <= 1 - eps_bar, so
# a forecast does not diverge with the horizon. The contraction is applied
# in the amplitude direction (Appendix, shared machinery).
ENFORCE_STABILITY = True
STABILITY_MAX_RADIUS = 0.98

# --- Nyblom-Hansen limit law -----------------------------------------------
# Appendix "Simulated Critical Values": 50,000 draws of a k-dimensional
# Brownian bridge on 2,000 grid points. The law is pivotal and depends only on
# the tested dimension k, so (k, n_draws, n_grid_points, seed) is the complete
# identity of a simulated law.
LIMITLAW_N_DRAWS = 50000
LIMITLAW_N_GRID = 2000
LIMITLAW_SEED = 20260709

# --- Fixed-regressor wild bootstrap ----------------------------------------
# Appendix, shared machinery: 499 draws. A fixed design constant. It is
# not reduced at run time, so a reported p-value cannot depend on how
# fast the machine happened to be.
N_BOOT = 499
BOOT_SEED = 20260710

# --- Kronecker specification check -----------------------------------------
# Appendix, shared machinery: moving blocks of length round(T^(1/3)) and 499
# replications.
N_SPEC_BOOT = 499
SPEC_SEED = 20260814

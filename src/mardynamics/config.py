"""Project-wide paths shared by the empirical and the simulation study.

The empirical study's own constants (panel composition, sample split, model
grids) live in :mod:`mardynamics.application.emp_config`, and the simulation
study's own constants live in :mod:`mardynamics.experiments.sim_config`. This
module holds only the paths both of them read.

The numerical constants of the estimator and of the test machinery (the
stability boundary, the limit-law simulation size and seed, the bootstrap
draws) live in :mod:`mardynamics.numerics_config`, and the pure numerical
functions read only that module.
"""

from __future__ import annotations

from pathlib import Path

# OUTPUT_DIR stays relative on purpose: the pipeline is run from the project
# root, and OUTPUT_DIR is written into the metadata JSON that the later
# scripts read, so an absolute path would bake this machine's layout into a
# stored artifact.
OUTPUT_DIR = "outputs"

# Project root, resolved from this file: ``<root>/src/mardynamics/config.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

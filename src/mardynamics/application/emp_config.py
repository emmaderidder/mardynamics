"""Constants for the empirical study.

Defines the panel (countries and indicators), the sample and forecast
windows, the tuning grids and the numerical settings of the estimators and
tests. The modules in mardynamics.application read their settings from here.
"""

from __future__ import annotations

# =========================================================
# The panel
# =========================================================
# Countries, in column order.
COUNTRIES = ("USA", "DEU", "FRA", "GBR", "CAN")

# Indicators, in row order.
INDICATORS = ("STIR", "LTIR", "PROD", "CPI", "UNRATE")

INDICATOR_LABELS = {
    "STIR": "Short-term interest rate",
    "LTIR": "Ten-year government-bond yield",
    "PROD": "Industrial production",
    "CPI": "Consumer price index",
    "UNRATE": "Unemployment rate",
}

COUNTRY_LABELS = {
    "USA": "United States",
    "DEU": "Germany",
    "FRA": "France",
    "GBR": "United Kingdom",
    "CAN": "Canada",
}

# The OECD source file for each indicator.
INDICATOR_FILES = {
    "STIR": "SHORT_TERM_INTEREST_RATE.csv",
    "LTIR": "LONG_TERM_INTEREST_RATE.csv",
    "PROD": "PRODUCTION_VOLUME.csv",
    "CPI": "CONSUMER_PRICE_INDEX.csv",
    "UNRATE": "MONTHLY_UNEMPLOYMENT_RATE.csv",
}

M_DIM = len(INDICATORS)
N_DIM = len(COUNTRIES)

# Dimension of the canonical parameter.
K_CANONICAL = M_DIM * M_DIM + N_DIM * N_DIM - 1

# Dimensions of the three canonical blocks.
BLOCK_DIMENSIONS = {
    "row": M_DIM * M_DIM - 1,
    "column": N_DIM * N_DIM - 1,
    "amplitude": 1,
}

# Parameter count of the unrestricted one-lag VAR.
K_VAR = (M_DIM * N_DIM) ** 2

# =========================================================
# Transformations
# =========================================================

LOG_DIFF_INDICATORS = ("PROD", "CPI")
DIFF_INDICATORS = ("STIR", "LTIR", "UNRATE")

# CPI is the only indicator with additional seasonal adjustment.
SEASONAL_INDICATORS = ("CPI",)

# ADF test settings: constant, no trend, Schwert maximum lag
# floor(12 (T/100)^(1/4)) = 17, AIC lag selection, MacKinnon p-values.
ADF_REGRESSION = "c"
ADF_MAX_LAG = 17
ADF_AUTOLAG = "AIC"

# =========================================================
# Sample and preprocessing windows
# =========================================================
# Raw data period.
RAW_START = "1991-01"
RAW_END = "2025-09"
# Transformed sample.
SAMPLE_START = "1991-02"
SAMPLE_END = "2025-09"
# Expected number of months and month-to-month transitions.
EXPECTED_MATRICES = 416
EXPECTED_TRANSITIONS = 415

# Validation forecasts
VALIDATION_START = "2007-02"
VALIDATION_END = "2016-12"
# Test forecasts
TEST_START = "2017-01"
TEST_END = "2025-09"

PLANNED_TEST_ORIGINS = 105

# The three preprocessing parameter sets.
PREPROCESSOR_CUTOFFS = {
    "full_sample": SAMPLE_END,
    "validation": "2007-01",
    "test": "2016-12",
}

# Which preprocessing version each analysis uses.
SECTION_PREPROCESSING = {
    "5.1_adf": "full_sample",
    "5.2_static_reference": "full_sample",
    "5.2_innovation_variance": "full_sample",
    "5.3_constancy": "full_sample",
    "5.4_validation": "validation",
    "5.4_test": "test",
    "5.5_descriptions": "test",
}

# Scale used for the test-period comparisons. A canonical shape is not
# invariant to the per-indicator scale. The comparison uses the benchmark
# refitted on this scale.
SECTION_55_BENCHMARK_VERSION = "test"

# =========================================================
# Source data and the one manual substitution
# =========================================================
# The original OECD download date is unavailable.
DOWNLOAD_DATE = "unknown_original_download_date"

# One missing observation, replaced with a value from an alternative source.
# The raw OECD file itself is left unchanged.
APPROVED_SUBSTITUTIONS = (
    {
        "country": "USA",
        "indicator": "STIR",
        "period": "2020-04",
        "value": 0.14,
        "units": "percent",
        "source_series": "TB3MS",
        "source": "Federal Reserve Board of Governors, H.15",
        "retrieval_date": "2026-08-21",
        "reason": (
            "The OECD short-term interest rate file has no observation for the "
            "United States in April 2020. This single documented substitution "
            "is approved; interpolation, forward filling and any old manual "
            "imputation are prohibited."
        ),
    },
)

# =========================================================
# Forecast design
# =========================================================
# The six models of the 3x2 design.
BENCHMARK_MODEL = "static_full"
MODELS = (
    "static_full",
    "static_reduced",
    "rolling_full",
    "rolling_reduced",
    "ms_full",
    "ms_reduced",
)
MODEL_LABELS = {
    "static_full": "Static MAR(1)",
    "static_reduced": "Static reduced-rank MAR(1)",
    "rolling_full": "Rolling MAR(1)",
    "rolling_reduced": "Rolling reduced-rank MAR(1)",
    "ms_full": "Two-state MS-MAR(1)",
    "ms_reduced": "Two-state reduced-rank MS-MAR(1)",
}
# The five models compared against the benchmark.
CHALLENGERS = tuple(model for model in MODELS if model != BENCHMARK_MODEL)

# Tuning grids
RANK_GRID = (1, 2, 3, 4)
WINDOW_GRID = (96, 120, 144, 168, 192)
FULL_RANK = 5
TIE_TOLERANCE = 1e-12

# Two regimes, fixed rather than selected.
N_REGIMES = 2

# =========================================================
# Numerical settings
# =========================================================
MASTER_SEED = 20260821

# The static fit is routed through the canonical NumPy ALS kernel
# (mardynamics.estimation.mar_fit.fit_static_mar_tensor_numpy).
STATIC_MAX_ITER = 200
STATIC_TOL = 1e-6
STATIC_STOPPING_RULE = "coefficient"

# Reduced-rank and rolling ALS.
ALS_MAX_ITER = 100
ALS_TOL = 1e-8
ALS_STOPPING_RULE = "criterion"

# Stability enforcement.
ENFORCE_STABILITY = True
MAX_RADIUS = 0.98

# Eight starts for every MS-MAR fit. Start 0 is the deterministic
# projection and residual-split construction. Starts 1 to 7 are independent
# tangent perturbations of Frobenius norm 0.10 for each canonical shape
# through the sphere exponential map, an N(0, 0.10^2) perturbation of kappa
# and an iid Bernoulli(0.5) initial regime split.
N_STARTS = 8
START_PERTURBATION = 0.10
START_LABEL_PROBABILITY = 0.5

# EM controls.
EM_MAX_ITER = 300
EM_MIN_ITER = 10
EM_LOGLIK_TOL = 1e-8
EM_PARAM_TOL = 1e-8
EM_PARAM_RULE = "canonical"
EM_ALS_ITER_PER_MSTEP = 5

# Probability floor and GEM rollback tolerance, recorded here so the
# manifest states the values a run used.
PROBABILITY_FLOOR = 1e-12
GEM_ROLLBACK_TOLERANCE = 1e-8

# Limit law.
LIMITLAW_N_DRAWS = 50000
LIMITLAW_N_GRID = 2000
LIMITLAW_SEED = 20260709

# Wild bootstrap.
WILD_BOOT_DRAWS_FULL = 499
WILD_BOOT_DRAWS_PREVIEW = 49
WILD_BOOT_SEED = 20260710

# Newey-West HAC lag of the forecast tests.
HAC_LAG = 3

# Maximum number of parallel workers.
MAX_WORKERS = 10

# Nominal levels.
NOMINAL_LEVELS = (0.01, 0.05, 0.10)
PRIMARY_LEVEL = 0.05

# =========================================================
# Interpretation thresholds
# =========================================================
# A regime with an occupancy share below this counts as poorly occupied.
REGIME_MIN_OCCUPANCY_SHARE = 0.05

# A regime with an effective sample size below this counts as too small.
# Set to the canonical dimension k = 49: below it a regime carries fewer
# weighted observations than the model has parameter directions.
REGIME_MIN_EFFECTIVE_SAMPLE = K_CANONICAL

# Two starts count as reaching the same optimum when their final
# log-likelihoods differ by less than REGIME_MULTISTART_LOGLIK_TOL times
# (1 + |best|). Set two orders of magnitude looser than the EM convergence
# tolerance of 1e-8, so one mode is not split into two by convergence noise.
REGIME_MULTISTART_LOGLIK_TOL = 1e-6

# Multi-start instability holds when the fraction of starts agreeing with
# the best final log-likelihood, at the tolerance above, falls below this.
REGIME_MULTISTART_MIN_AGREEING_FRACTION = 0.5

# The four thresholds above, carried with a label for the regime tables.
REGIME_INTERPRETATION_THRESHOLDS = {
    "provenance": (
        "author decision, not a value the methodology fixes: these "
        "conditions are named but no numeric threshold is fixed for any "
        "of them"
    ),
    "REGIME_MIN_OCCUPANCY_SHARE": REGIME_MIN_OCCUPANCY_SHARE,
    "REGIME_MIN_EFFECTIVE_SAMPLE": REGIME_MIN_EFFECTIVE_SAMPLE,
    "REGIME_MULTISTART_LOGLIK_TOL": REGIME_MULTISTART_LOGLIK_TOL,
    "REGIME_MULTISTART_MIN_AGREEING_FRACTION": (
        REGIME_MULTISTART_MIN_AGREEING_FRACTION
    ),
}

# The one simulation cell used as a finite-sample reference point.
SIMULATION_BENCHMARK = {
    "empirical_k": 49,
    "empirical_N": 415,
    "empirical_k_over_N": 49 / 415,
    "benchmark_cell": "(m,n,T)=(5,5,400)",
    "benchmark_k": 49,
    "benchmark_N": 399,
    "benchmark_k_over_N": 49 / 399,
    "benchmark_size_at_5pct": 0.038,
    "statement": (
        "The test was mildly conservative in that controlled cell. k/N is a "
        "dimension-to-information indicator, not a validity threshold, and it "
        "does not determine size or power."
    ),
}

# Run modes.
RUN_MODES = ("preview", "full")
# Preview uses every fourth ordered origin, starting from the first.
PREVIEW_ORIGIN_STRIDE = 4

# Schema version written into every output.
SCHEMA_VERSION = "1.0.0"


def wild_bootstrap_draws(mode: str) -> int:
    """Return the number of wild-bootstrap draws for a run mode.

    Parameters
    ----------
    mode : {"preview", "full"}
        Run mode.

    Returns
    -------
    int
        49 in preview mode, 499 in full mode.
    """
    if mode not in RUN_MODES:
        raise ValueError(f"unknown run mode {mode!r}")
    return WILD_BOOT_DRAWS_PREVIEW if mode == "preview" else WILD_BOOT_DRAWS_FULL

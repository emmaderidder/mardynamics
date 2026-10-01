"""Matrix autoregressive (MAR) models for macroeconomic panels.

``scripts/run_application.py`` (application study) and
``scripts/run_experiments.py`` (experiment study) are the two entry
points; every reusable piece of shared logic lives in this package. The
application study is the ``mardynamics.application`` subpackage and the
experiment study is ``mardynamics.experiments``. The shared modules are:

``config``
    Project-wide paths shared by the empirical and the simulation study.
``numerics_config``
    Shared numerical constants: the stability boundary, the limit-law
    simulation size and seed, the bootstrap draws. The pure numerical functions
    read this module, not ``config``.
``core.mar_tensor``
    Slicing the ``(T, m, n)`` tensor into one-step lagged predictor/response
    pairs and mapping forecast matrices back to labelled columns.
``core.mar_canonical``
    Canonical coordinates ``(A_bar, B_breve, gamma)`` that resolve the bilinear
    scale and sign indeterminacy, the unrestricted transition-matrix estimate
    and its Kronecker projection, and the truncated-SVD rank projection.
``estimation.mar_fit``
    The tensor-facing MAR(1) estimators: the static NumPy
    projection-initialised ALS fit and the regime-switching EM estimator with
    its one-step forecast rule.
``estimation.regime_em``
    EM machinery behind the regime-switching estimator: the weighted ALS
    update, shared by every MAR-LSE route, the static warm start with
    residual-based regime initialisation, and the Gaussian hidden-Markov
    filter/smoother.
``core.stability``
    Spectral-radius diagnostics and the soft stability projection.
``core.structural_tests``
    The constancy test: chart scores, the Nyblom-Hansen statistic and its three
    influence blocks, the Holm adjustment, the simulated limit law, the wild
    bootstrap and the specification checks.
"""

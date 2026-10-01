"""Estimator calls at the settings used by the empirical study.

Each function forwards to the shared NumPy ALS kernel or the EM estimator
with a fixed set of numerical settings, so every model in the empirical
study is fit through one implementation of the estimator, the rank
projection, the stability rule and the EM step.
"""

from __future__ import annotations

import numpy as np

import mardynamics.application.emp_config as emp_config
import mardynamics.application.starts as starts_mod
import mardynamics.estimation.mar_fit as mar_fit

# Identity stamped into every telemetry record, naming the estimator route.
ESTIMATOR_IDENTITY = "numpy_als_kronecker_projection_init"


def static_fit(tensor: np.ndarray, return_telemetry: bool = False):
    """Fit the static MAR(1) with the alternating least-squares kernel.

    Starts from the Kronecker projection and iterates to STATIC_MAX_ITER
    or STATIC_TOL, enforcing the stability bound MAX_RADIUS.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process, shape (T, m, n).
    return_telemetry : bool, default False
        Whether to return the solver diagnostics alongside the pair.

    Returns
    -------
    tuple
        (A, B), or (A, B, telemetry) when return_telemetry is set.
    """
    result = mar_fit.fit_static_mar_tensor_numpy(
        tensor,
        max_iter=emp_config.STATIC_MAX_ITER,
        tol=emp_config.STATIC_TOL,
        stopping_rule=emp_config.STATIC_STOPPING_RULE,
        enforce_stability=emp_config.ENFORCE_STABILITY,
        max_radius=emp_config.MAX_RADIUS,
        return_telemetry=return_telemetry,
    )
    if not return_telemetry:
        return result
    A, B, telemetry = result
    telemetry = dict(telemetry)
    telemetry["estimator_identity"] = ESTIMATOR_IDENTITY
    return A, B, telemetry


def als_fit(
    tensor: np.ndarray, rank: int | None = None, return_telemetry: bool = False
):
    """Fit a static or rolling MAR(1), optionally at a reduced rank.

    Iterates to ALS_MAX_ITER or ALS_TOL, enforcing the stability bound
    MAX_RADIUS. A rank imposes the same rank on both operators.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process, shape (T, m, n).
    rank : int or None, optional
        Common rank of the two operators. None leaves them unrestricted.
    return_telemetry : bool, default False
        Whether to return the solver diagnostics alongside the pair.

    Returns
    -------
    tuple
        (A, B), or (A, B, telemetry) when return_telemetry is set.
    """
    result = mar_fit.fit_static_mar_tensor_numpy(
        tensor,
        max_iter=emp_config.ALS_MAX_ITER,
        tol=emp_config.ALS_TOL,
        stopping_rule=emp_config.ALS_STOPPING_RULE,
        rank_A=rank,
        rank_B=rank,
        enforce_stability=emp_config.ENFORCE_STABILITY,
        max_radius=emp_config.MAX_RADIUS,
        return_telemetry=return_telemetry,
    )
    if not return_telemetry:
        return result
    A, B, telemetry = result
    telemetry = dict(telemetry)
    telemetry["estimator_identity"] = ESTIMATOR_IDENTITY
    return A, B, telemetry


def regime_fit(tensor: np.ndarray, fit_label: str, rank: int | None = None) -> dict:
    """Fit a two-state regime-switching MAR(1) from several EM starts.

    Every call builds its own starts from the given data, rather than
    reusing a previous fit's parameters.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process, shape (T, m, n).
    fit_label : str
        Identity of this fit, used to seed its starts.
    rank : int or None, optional
        Common rank of the regime operators.

    Returns
    -------
    dict
        The fit result, including the per-start diagnostics.
    """
    eight = starts_mod.build_starts(tensor, fit_label, rank=rank)
    result = mar_fit.fit_regime_switching_mar_tensor(
        tensor,
        n_regimes=emp_config.N_REGIMES,
        max_iter=emp_config.EM_MAX_ITER,
        min_iter=emp_config.EM_MIN_ITER,
        tol=emp_config.EM_LOGLIK_TOL,
        param_tol=emp_config.EM_PARAM_TOL,
        param_rule=emp_config.EM_PARAM_RULE,
        n_als_iter=emp_config.EM_ALS_ITER_PER_MSTEP,
        als_tol=emp_config.ALS_TOL,
        als_stopping_rule=emp_config.ALS_STOPPING_RULE,
        rank_A=rank,
        rank_B=rank,
        starts=eight,
        enforce_stability=emp_config.ENFORCE_STABILITY,
        max_radius=emp_config.MAX_RADIUS,
        store_init_diagnostics=True,
    )
    result["estimator_identity"] = ESTIMATOR_IDENTITY
    return result

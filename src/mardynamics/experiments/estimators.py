"""Estimator calls and numerical telemetry of the simulation study.

This module owns no estimation mathematics. It exists so every
experiment reaches the same shared estimator with the same numerical
controls, and so the numerical-reliability record is assembled in one
place.

The static fit is the NumPy route used for the simulation: the same
alternating recursion and the same projection starting value as the
empirical route, whose agreement the shared regression test checks. The multiple-start check reaches the
same recursion through the same shared function, passing an explicit
start instead of the projection one.
"""

from __future__ import annotations

import numpy as np

import mardynamics.estimation.mar_fit as mm
import mardynamics.core.mar_tensor as mt
import mardynamics.estimation.regime_em as regime_em
import mardynamics.experiments.sim_config as simcfg
import mardynamics.core.structural_tests as structural_tests
from mardynamics.core.mar_canonical import canonical_triple


def static_fit(
    tensor: np.ndarray,
    return_telemetry: bool = False,
    A_init: np.ndarray | None = None,
    B_init: np.ndarray | None = None,
):
    """Fit the static MAR(1) with the simulation's shared controls.

    With ``A_init`` and ``B_init`` omitted this is the ordinary fit: the
    shared alternating recursion started at the shared Kronecker
    projection estimator. Passing a start routes the same recursion from
    a different point, which is what the multiple-start check needs and
    the only difference between its starts.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(T, m, n)``.
    return_telemetry : bool, default False
        If True, also return the fit record.
    A_init, B_init : numpy.ndarray or None, optional
        An explicit start. Both must be given together.

    Returns
    -------
    tuple
        ``(A, B)``, or ``(A, B, telemetry)``.

    Raises
    ------
    ValueError
        If only one of the two start matrices is given.
    """
    if (A_init is None) != (B_init is None):
        raise ValueError(
            "static_fit: an explicit start needs both A_init and B_init."
        )

    if A_init is None:
        return mm.fit_static_mar_tensor_numpy(
            tensor,
            max_iter=simcfg.STATIC_ALS_MAX_ITER,
            tol=simcfg.STATIC_ALS_TOL,
            stopping_rule=simcfg.STATIC_ALS_STOPPING_RULE,
            enforce_stability=simcfg.ENFORCE_STABILITY,
            max_radius=simcfg.MAX_RADIUS,
            return_telemetry=return_telemetry,
        )

    X_prev, Y = mt.split_mar_lagged_tensor(np.asarray(tensor, dtype=float))
    A, B, _, telemetry = regime_em.estimate_weighted_mar_parameters(
        X_prev=X_prev,
        Y=Y,
        weights=np.ones(len(X_prev)),
        max_iter=simcfg.STATIC_ALS_MAX_ITER,
        A_init=A_init,
        B_init=B_init,
        tol=simcfg.STATIC_ALS_TOL,
        stopping_rule=simcfg.STATIC_ALS_STOPPING_RULE,
        enforce_stability=simcfg.ENFORCE_STABILITY,
        max_radius=simcfg.MAX_RADIUS,
        return_telemetry=True,
    )

    if not return_telemetry:
        return A, B

    telemetry = dict(telemetry)
    telemetry["projection_regressor_rank"] = -1
    telemetry["projection_minimum_norm"] = False
    return A, B, telemetry


def perturbed_start(
    tensor: np.ndarray, rng: np.random.Generator, perturbation: float
) -> dict:
    """Build one deterministic perturbation start of the multiple-start check.

    The ordinary fit is moved along a seeded geodesic in each shape and
    by a seeded step in the log magnitude, and the result is mapped back
    through the canonical map and contracted into the stability region.
    The start satisfies the canonical norm restriction (both shapes have
    unit Frobenius norm), the sign restriction (the canonical map
    anchors the designated coordinate) and the stability restriction, by
    construction, not by a check applied afterwards.

    Parameters
    ----------
    tensor : numpy.ndarray
        The dataset the start is built for.
    rng : numpy.random.Generator
        The replication-and-start specific generator.
    perturbation : float
        Geodesic displacement of each shape and the standard deviation of the
        log-magnitude step.

    Returns
    -------
    dict
        Keys ``A``, ``B``, ``rho`` and ``geodesic_displacement``.
    """
    from mardynamics.experiments.design import sphere_exponential
    from mardynamics.core.stability import project_to_stable, spectral_radius_from_matrices

    A_static, B_static = static_fit(tensor)
    A_bar, B_breve, gamma = canonical_triple(A_static, B_static)

    moved = {}
    for name, base in (("A", A_bar), ("B", B_breve)):
        flat = base.flatten(order="F")
        tangent = rng.standard_normal(flat.size)
        tangent = tangent - (tangent @ flat) * flat
        moved[name] = sphere_exponential(flat, tangent, perturbation).reshape(
            base.shape, order="F"
        )

    kappa = float(np.log(gamma)) + float(rng.normal(0.0, perturbation))
    A_bar_start, B_breve_start, gamma_start = canonical_triple(
        moved["A"], np.exp(kappa) * moved["B"]
    )
    A_start, B_start = project_to_stable(
        A_bar_start, gamma_start * B_breve_start, max_radius=simcfg.MAX_RADIUS
    )

    return {
        "A": A_start,
        "B": B_start,
        "rho": float(spectral_radius_from_matrices(A_start, B_start)),
        "geodesic_displacement": float(perturbation),
    }


def numerical_telemetry(
    tensor: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    fit_telemetry: dict,
    quantities: dict | None = None,
) -> dict:
    """Assemble the numerical-reliability record of one replication.

    A fixed list is reported per stable-DGP replication. Everything the
    shared estimator already reports is passed through under an
    unambiguous name. Everything else is computed here from the same
    one fit.

    The two inversion indicators are honest about what the shared
    statistics do: no covariance inside a statistic is ever
    pseudo-inverted or regularised, so those two indicators are always
    false and the inversion-failure indicators are what carries the
    information. They are still reported, because a reader should be
    able to see that they are false, not assume it.

    Parameters
    ----------
    tensor : numpy.ndarray
        The dataset.
    A, B : numpy.ndarray
        The fitted operator pair.
    fit_telemetry : dict
        The shared estimator's record of the fit.
    quantities : dict or None, optional
        A previously computed chart-score result for the same tensor and pair.
        Passing it is how a caller keeps every statistic of a replication on one
        score loop; omitting it recomputes them.

    Returns
    -------
    dict
        The numerical-reliability record, flat and directly storable. When a
        required covariance cannot be inverted the corresponding failure
        indicator is set and the quantities that depend on the inversion are
        missing rather than zero.
    """
    record = {
        "converged": bool(fit_telemetry.get("converged", False)),
        "n_iterations": int(fit_telemetry.get("als_iterations", -1)),
        "initial_objective": float(fit_telemetry.get("criterion_before", np.nan)),
        "final_objective": float(fit_telemetry.get("criterion_after", np.nan)),
        "gem_step_accepted": bool(fit_telemetry.get("accepted", True)),
        "stability_contracted": bool(fit_telemetry.get("stability_contracted", False)),
        "n_stability_contractions": int(
            fit_telemetry.get("n_stability_contractions", 0)
        ),
        "projection_regressor_rank": int(
            fit_telemetry.get("projection_regressor_rank", -1)
        ),
        "projection_minimum_norm": bool(
            fit_telemetry.get("projection_minimum_norm", False)
        ),
        # The shared statistics apply neither a ridge nor a
        # pseudo-inverse to any covariance that enters a statistic. The
        # check is the Cholesky factorisation, and it raises where the
        # statistic is undefined.
        "covariance_regularised": False,
        "pseudoinverse_used": False,
        "hessian_inversion_failed": False,
        "score_covariance_inversion_failed": False,
        "foc_norm_canonical": np.nan,
        "min_eigenvalue_H": np.nan,
        "cond_H": np.nan,
        "min_eigenvalue_Omega": np.nan,
        "cond_Omega": np.nan,
        "rho_fitted": float(
            np.max(np.abs(np.linalg.eigvals(np.kron(B, A))))
        ),
    }
    record["stability_binding"] = bool(
        record["rho_fitted"] >= simcfg.MAX_RADIUS - 1e-9
    )

    try:
        if quantities is None:
            quantities = structural_tests.chart_score_quantities(tensor, A, B)
    except structural_tests.SingularCovarianceError:
        record["hessian_inversion_failed"] = True
        return record

    H_hat = quantities["H_hat"]
    Omega_hat = quantities["Omega_hat"]

    record["min_eigenvalue_H"] = float(np.min(np.linalg.eigvalsh(H_hat)))
    record["cond_H"] = float(np.linalg.cond(H_hat))
    record["min_eigenvalue_Omega"] = float(np.min(np.linalg.eigvalsh(Omega_hat)))
    record["cond_Omega"] = float(np.linalg.cond(Omega_hat))
    record["score_covariance_inversion_failed"] = bool(
        record["min_eigenvalue_Omega"] <= 0.0
    )
    # The first-order condition in canonical coordinates: the mean chart
    # score, which is the gradient of the criterion along the identified
    # directions and is zero at an interior optimum.
    record["foc_norm_canonical"] = float(
        np.linalg.norm(quantities["g"].mean(axis=0))
    )
    return record

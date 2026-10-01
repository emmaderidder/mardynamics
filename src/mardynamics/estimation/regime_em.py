"""EM machinery for the regime-switching MAR estimator.

This module holds the estimation steps that
``mar_fit.fit_regime_switching_mar_tensor`` runs: the weighted
alternating-least-squares update for a single regime, the static-MAR
warm start with its residual-based regime initialisation, the Gaussian
hidden-Markov forward-backward filter/smoother with its E-step
quantities, the transition-matrix initialisation and the regime
ordering convention.

It contains pure NumPy/pandas logic. The canonical-coordinate algebra
lives in ``mar_canonical`` and the tensor/panel reshaping in
``mar_tensor``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from mardynamics.core.mar_canonical import normalise_ab, project_low_rank
import mardynamics.core.mar_tensor as mt
import mardynamics.core.stability as st


def _perturb_static_fit_matrices(
    A_static: np.ndarray,
    B_static: np.ndarray,
    tau_init: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Frobenius-normalised, Gaussian-perturbed copy of the static fit.

    Parameters
    ----------
    A_static, B_static : numpy.ndarray
        Static-MAR coefficient matrices.
    tau_init : float
        Standard deviation of the additive Gaussian noise.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    tuple of numpy.ndarray
        The perturbed, normalised ``(A_start, B_start)``.
    """
    A_start = A_static + rng.normal(loc=0.0, scale=tau_init, size=A_static.shape)
    B_start = B_static + rng.normal(loc=0.0, scale=tau_init, size=B_static.shape)

    A_start, B_start = normalise_ab(A_start, B_start)

    return A_start, B_start


def static_residual_norms(
    X_prev: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Per-observation Frobenius norms of the static one-step residuals.

    The residual-split initialisation orders the observations by how badly the
    single static fit explains them, so this is the quantity that decides which
    observation starts in which regime.

    Parameters
    ----------
    X_prev, Y : numpy.ndarray
        Lagged predictors and one-step responses, shape ``(n_obs, m, n)``.
    A, B : numpy.ndarray
        The static operator pair.

    Returns
    -------
    numpy.ndarray
        The ``n_obs`` residual norms.
    """
    fitted = np.array([A @ X_prev[t] @ B.T for t in range(len(X_prev))])
    return np.sqrt(((Y - fitted) ** 2).sum(axis=(1, 2)))


def _weighted_criterion(
    X_prev: np.ndarray,
    Y: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted least-squares criterion ``sum_t w_t ||Y_t - A X_t B'||_F^2``.

    This is the objective the M-step minimises for one regime, with ``w_t`` the
    smoothed responsibility of that regime.

    Parameters
    ----------
    X_prev, Y : numpy.ndarray
        Lagged predictors and responses, shape ``(n_obs, m, n)``.
    A, B : numpy.ndarray
        Operator pair to evaluate.
    weights : numpy.ndarray
        Per-observation responsibilities.

    Returns
    -------
    float
        The weighted criterion.
    """
    resid = Y - A @ X_prev @ B.T
    return float((weights[:, None, None] * resid**2).sum())


def estimate_weighted_mar_parameters(
    X_prev: np.ndarray,
    Y: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 20,
    A_init: np.ndarray | None = None,
    B_init: np.ndarray | None = None,
    tol: float = 1e-6,
    rank_A: int | None = None,
    rank_B: int | None = None,
    enforce_stability: bool = False,
    max_radius: float = 0.99,
    stopping_rule: str = "coefficient",
    return_telemetry: bool = False,
):
    """Estimate one regime's ``(A, B, sigma2)`` by weighted alternating least squares.

    Alternates the closed-form weighted least-squares update for ``A`` (given ``B``)
    and ``B`` (given ``A``), optionally projecting each onto a low rank, with
    Frobenius normalisation of ``A`` each iteration. ``sigma2`` is the
    weight-averaged residual variance per element.

    The step is started from the previous iterate and its output is accepted
    only when it does not increase the weighted criterion. Otherwise the
    incoming pair is returned unchanged. That acceptance is what makes this
    a generalised EM step: the rank projection is the closest rank-``r``
    matrix in the ordinary Frobenius sense and therefore not the minimiser
    of the rank-constrained conditional problem, and the stability
    rescaling is not a minimiser either, so neither is guaranteed on its
    own to leave the criterion no worse.

    Parameters
    ----------
    X_prev, Y : numpy.ndarray
        Lagged predictors and one-step responses, shape ``(n_obs, m, n)``.
    weights : numpy.ndarray
        Per-observation regime responsibilities (length ``n_obs``).
    max_iter : int, default 20
        Maximum ALS iterations.
    A_init, B_init : numpy.ndarray or None, optional
        Warm-start matrices; identity if omitted.
    tol : float, default 1e-6
        Convergence tolerance, read according to ``stopping_rule``.
    rank_A, rank_B : int or None, optional
        Optional low-rank constraints.
    stopping_rule : {"coefficient", "criterion"}, default "coefficient"
        What ``tol`` measures. ``"coefficient"`` stops when the larger of
        the two Frobenius coefficient changes falls below ``tol``.
        ``"criterion"`` stops when the relative change in the weighted
        criterion falls below ``tol``. The default preserves the
        behaviour every empirical call site was written against. The
        simulation study passes ``"criterion"`` because it states its
        tolerances in criterion units.
    return_telemetry : bool, default False
        If True, return a fourth element with the generalised-EM record of this
        step: the criterion before and after, whether the proposal was accepted,
        the iterations used, the weight mass, the effective sample size, whether
        the recursion stopped on its tolerance, and how many sweeps had to be
        contracted back into the stable region. The last three are observational
        additions for the simulation study's numerical-reliability reporting and
        do not change what the estimator computes.

    Returns
    -------
    tuple
        ``(A, B, sigma2)``, or ``(A, B, sigma2, telemetry)`` when
        ``return_telemetry`` is set.
    """
    if stopping_rule not in ("coefficient", "criterion"):
        raise ValueError(f"unknown stopping_rule={stopping_rule!r}")
    n_obs, m, n = Y.shape

    weights = np.asarray(weights, dtype=float)
    weights = np.clip(weights, 1e-12, None)

    if A_init is None:
        A = np.eye(m)
    else:
        A = A_init.copy()

    if B_init is None:
        B = np.eye(n)
    else:
        B = B_init.copy()

    A, B = normalise_ab(A, B)

    # Generalised EM: remember the incoming pair and its criterion so the step
    # can be rejected if the sweep, the rank projection or the stability
    # rescaling leaves the weighted criterion worse than it found it.
    A_in, B_in = A.copy(), B.copy()
    criterion_in = _weighted_criterion(X_prev, Y, A_in, B_in, weights)

    criterion_old = criterion_in
    n_iter = 0
    # Purely observational counters. They record what the recursion already
    # does and change nothing about it: how many sweeps had to be contracted
    # back into the stable region, and whether the loop stopped on its
    # tolerance or exhausted its iteration budget.
    n_stability_contractions = 0
    converged = False

    for _ in range(max_iter):
        n_iter += 1
        A_old = A.copy()
        B_old = B.copy()

        Z = np.array([X_prev[t] @ B.T for t in range(n_obs)])

        cross_A = np.zeros((m, m))
        gram_A = np.zeros((m, m))

        for t in range(n_obs):
            cross_A += weights[t] * (Y[t] @ Z[t].T)
            gram_A += weights[t] * (Z[t] @ Z[t].T)

        # Pseudo-inverse: the minimum-norm solution lets the
        # responsibility-weighted update return a value instead of an error
        # for a regime with low occupancy. This applies to the M-step only;
        # no covariance inside a statistic is pseudo-inverted.
        A = cross_A @ np.linalg.pinv(gram_A)

        if rank_A is not None:
            A = project_low_rank(A, rank_A)

        W = np.array([A @ X_prev[t] for t in range(n_obs)])

        cross_C = np.zeros((n, n))
        gram_C = np.zeros((n, n))

        for t in range(n_obs):
            cross_C += weights[t] * (W[t].T @ Y[t])
            gram_C += weights[t] * (W[t].T @ W[t])

        C = np.linalg.pinv(gram_C) @ cross_C  # minimum norm; see the A update
        B = C.T

        if rank_B is not None:
            B = project_low_rank(B, rank_B)

        A, B = normalise_ab(A, B)

        if enforce_stability:
            A_pre, B_pre = A, B
            A, B = st.project_to_stable(A, B, max_radius=max_radius)
            # The projection is soft and returns its inputs unchanged when the
            # pair is already stable, so identity is the exact test.
            if A is not A_pre or B is not B_pre:
                n_stability_contractions += 1

        if stopping_rule == "coefficient":
            delta = max(
                np.linalg.norm(A - A_old, ord="fro"),
                np.linalg.norm(B - B_old, ord="fro"),
            )
            if delta < tol:
                converged = True
                break
        else:
            criterion_new = _weighted_criterion(X_prev, Y, A, B, weights)
            rel_change = abs(criterion_old - criterion_new) / (1.0 + abs(criterion_old))
            criterion_old = criterion_new
            if rel_change < tol:
                converged = True
                break

    criterion_out = _weighted_criterion(X_prev, Y, A, B, weights)
    accepted = criterion_out <= criterion_in
    if not accepted:
        A, B = A_in, B_in
        criterion_out = criterion_in

    fitted = np.array([A @ X_prev[t] @ B.T for t in range(n_obs)])
    resid = Y - fitted

    sigma2 = (weights[:, None, None] * resid**2).sum() / (weights.sum() * m * n)
    sigma2 = float(max(sigma2, 1e-8))

    if not return_telemetry:
        return A, B, sigma2

    weight_mass = float(weights.sum())
    telemetry = {
        "criterion_before": float(criterion_in),
        "criterion_after": float(criterion_out),
        "accepted": bool(accepted),
        "als_iterations": int(n_iter),
        "occupancy": weight_mass,
        "n_eff": float(weight_mass**2 / float((weights**2).sum())),
        # Added for the simulation study's numerical-reliability reporting.
        # Both are observational: they describe the recursion above without
        # altering it, and every existing caller reads this dict by key.
        "converged": bool(converged),
        "n_stability_contractions": int(n_stability_contractions),
        "stability_contracted": bool(n_stability_contractions > 0),
    }
    return A, B, sigma2, telemetry


def initialise_regimes_from_static_mar(
    prepared_init: dict,
    n_regimes: int,
    n_als_iter: int,
    tau_init: float,
    random_state: int,
    init_id: int,
    rank_A: int | None = None,
    rank_B: int | None = None,
    enforce_stability: bool = False,
    max_radius: float = 0.99,
) -> tuple:
    """Build an initial set of regime parameters from a static-MAR residual split.

    Observations are soft-assigned to regimes by quantiles of the static-MAR
    residual norm (low residual -> low-variance regime), and each regime's
    ``(A, B, sigma2)`` is estimated by weighted ALS from a noisy perturbation of
    the static fit, giving distinct EM starting points.

    Parameters
    ----------
    prepared_init : dict
        The prepared initialisation input, with keys ``X_prev``, ``Y``,
        ``A_static``, ``B_static`` and ``resid_norm``.
    n_regimes : int
        Number of regimes.
    n_als_iter : int
        Inner ALS iterations per regime.
    tau_init : float
        Standard deviation of the Gaussian perturbation applied to the static fit.
    random_state, init_id : int
        Seed components. The RNG is seeded with ``random_state + init_id``.
    rank_A, rank_B : int or None, optional
        Optional low-rank constraints on ``A``/``B``.

    Returns
    -------
    tuple
        ``(gamma, A_list, B_list, sigma2_list, init_info)``.
    """
    X_prev = prepared_init["X_prev"]
    Y = prepared_init["Y"]
    A_static = prepared_init["A_static"]
    B_static = prepared_init["B_static"]
    resid_norm = prepared_init["resid_norm"]

    rng = np.random.default_rng(random_state + init_id)

    gamma = np.full((len(Y), n_regimes), 0.10 / max(n_regimes - 1, 1))

    if n_regimes == 2:
        threshold = np.median(resid_norm)
        low_resid_mask = resid_norm <= threshold

        gamma[low_resid_mask, 0] = 0.90
        gamma[low_resid_mask, 1] = 0.10
        gamma[~low_resid_mask, 0] = 0.10
        gamma[~low_resid_mask, 1] = 0.90

        split_rule = f"median residual norm = {threshold:.6f}"

    else:
        labels = pd.qcut(
            pd.Series(resid_norm),
            q=n_regimes,
            labels=False,
            duplicates="drop",
        ).to_numpy()

        if len(np.unique(labels)) < n_regimes:
            rank_order = pd.Series(resid_norm).rank(method="first").to_numpy()
            labels = pd.qcut(rank_order, q=n_regimes, labels=False).astype(int)

        for t in range(len(Y)):
            gamma[t, :] = 0.10 / max(n_regimes - 1, 1)
            gamma[t, labels[t]] = 0.90

        split_rule = "quantile split on static-MAR residual norms"

    gamma /= gamma.sum(axis=1, keepdims=True)

    A_list = []
    B_list = []
    sigma2_list = []
    A_start_list = []
    B_start_list = []

    for k in range(n_regimes):
        A_start_k, B_start_k = _perturb_static_fit_matrices(
            A_static=A_static,
            B_static=B_static,
            tau_init=tau_init,
            rng=rng,
        )

        A_k, B_k, sigma2_k = estimate_weighted_mar_parameters(
            X_prev=X_prev,
            Y=Y,
            weights=gamma[:, k],
            max_iter=n_als_iter,
            rank_A=rank_A,
            rank_B=rank_B,
            A_init=A_start_k,
            B_init=B_start_k,
            enforce_stability=enforce_stability,
            max_radius=max_radius,
        )

        A_start_list.append(A_start_k)
        B_start_list.append(B_start_k)
        A_list.append(A_k)
        B_list.append(B_k)
        sigma2_list.append(sigma2_k)

    init_info = {
        "method": "static_mar_residual_split_with_noisy_A_B_starts",
        "A_static": A_static,
        "B_static": B_static,
        "residual_norm": resid_norm,
        "split_rule": split_rule,
        "initial_gamma": gamma.copy(),
        "initial_probs": np.full(n_regimes, 1.0 / n_regimes),
        "tau_init": tau_init,
        "init_id": init_id,
        "A_start_list": A_start_list,
        "B_start_list": B_start_list,
    }

    return gamma, A_list, B_list, sigma2_list, init_info


def compute_regime_log_likelihoods(
    X_prev: np.ndarray,
    Y: np.ndarray,
    A_list: list[np.ndarray],
    B_list: list[np.ndarray],
    sigma2_list: list[float],
) -> np.ndarray:
    """Compute per-observation Gaussian log-emission densities for each regime.

    Parameters
    ----------
    X_prev, Y : numpy.ndarray
        Lagged predictors and responses, shape ``(n_obs, m, n)``.
    A_list, B_list : list of numpy.ndarray
        Per-regime coefficient matrices.
    sigma2_list : list of float
        Per-regime scalar noise variances.

    Returns
    -------
    numpy.ndarray
        Log-emission matrix of shape ``(n_obs, n_regimes)``.
    """
    n_obs, m, n = Y.shape
    d = m * n
    n_regimes = len(A_list)

    log_emissions = np.zeros((n_obs, n_regimes))

    for k in range(n_regimes):
        fitted_k = np.array([A_list[k] @ X_prev[t] @ B_list[k].T for t in range(n_obs)])
        rss_k = ((Y - fitted_k) ** 2).sum(axis=(1, 2))
        sigma2_k = max(float(sigma2_list[k]), 1e-8)

        log_emissions[:, k] = -0.5 * (
            d * np.log(2.0 * np.pi * sigma2_k) + rss_k / sigma2_k
        )

    return log_emissions


def forward_backward_filter_smoother(
    log_emissions: np.ndarray,
    transition_matrix: np.ndarray,
    initial_probs: np.ndarray,
) -> tuple:
    """Run the Gaussian HMM forward-backward filter/smoother.

    Uses the column-stochastic convention ``transition_matrix[i, j] =
    P(S_t = i | S_{t-1} = j)`` so that ``predicted = transition_matrix @ filtered``.
    Returns filtered, one-step predicted and smoothed regime probabilities, the
    two-slice smoothed joints (indexed ``[current, previous]``) and the log-likelihood.

    Parameters
    ----------
    log_emissions : numpy.ndarray
        Log-emission matrix of shape ``(n_obs, n_regimes)``.
    transition_matrix : numpy.ndarray
        Column-stochastic transition matrix.
    initial_probs : numpy.ndarray
        Initial regime distribution.

    Returns
    -------
    tuple
        ``(filtered, predicted, smoothed, joint_smoothed, loglik)``.
    """
    n_obs, n_regimes = log_emissions.shape

    filtered = np.zeros((n_obs, n_regimes))
    predicted = np.zeros((n_obs, n_regimes))
    smoothed = np.zeros((n_obs, n_regimes))
    joint_smoothed = np.zeros((n_obs, n_regimes, n_regimes))

    predicted[0] = np.clip(initial_probs, 1e-12, None)
    predicted[0] /= predicted[0].sum()

    log_w0 = np.log(predicted[0]) + log_emissions[0]
    c0 = logsumexp(log_w0)
    filtered[0] = np.exp(log_w0 - c0)

    loglik = c0

    for t in range(1, n_obs):
        predicted[t] = transition_matrix @ filtered[t - 1]
        predicted[t] = np.clip(predicted[t], 1e-12, None)
        predicted[t] /= predicted[t].sum()

        log_w = np.log(predicted[t]) + log_emissions[t]
        ct = logsumexp(log_w)
        filtered[t] = np.exp(log_w - ct)
        loglik += ct

    smoothed[-1] = filtered[-1]

    for t in range(n_obs - 2, -1, -1):
        ratio = smoothed[t + 1] / np.clip(predicted[t + 1], 1e-12, None)
        smoothed[t] = filtered[t] * (transition_matrix.T @ ratio)
        smoothed[t] = np.clip(smoothed[t], 1e-12, None)
        smoothed[t] /= smoothed[t].sum()

    for t in range(1, n_obs):
        numer = (
            transition_matrix
            * smoothed[t][:, None]
            * filtered[t - 1][None, :]
            / np.clip(predicted[t][:, None], 1e-12, None)
        )
        numer = np.clip(numer, 1e-12, None)
        joint_smoothed[t] = numer / numer.sum()

    return filtered, predicted, smoothed, joint_smoothed, float(loglik)


def initialise_transition_matrix(n_regimes: int, stay_prob: float = 0.90) -> np.ndarray:
    """Build a transition matrix with constant diagonal stay probability.

    Parameters
    ----------
    n_regimes : int
        Number of regimes.
    stay_prob : float, default 0.90
        Probability of remaining in the same regime; the remainder is split
        uniformly over the other regimes.

    Returns
    -------
    numpy.ndarray
        The ``(n_regimes, n_regimes)`` transition matrix.
    """
    if n_regimes == 1:
        return np.ones((1, 1))

    off_diag = (1.0 - stay_prob) / (n_regimes - 1)
    P = np.full((n_regimes, n_regimes), off_diag)
    np.fill_diagonal(P, stay_prob)

    return P


def sort_regimes_by_sigma2(
    A_list: list[np.ndarray],
    B_list: list[np.ndarray],
    sigma2_list: list[float],
    P: np.ndarray,
    rho: np.ndarray,
    filtered: np.ndarray,
    predicted: np.ndarray,
    smoothed: np.ndarray,
    joint_smoothed: np.ndarray,
) -> tuple:
    """Relabel regimes in ascending order of ``sigma2`` for identifiability.

    Reorders every regime-indexed object consistently (coefficient lists, the
    transition matrix on both axes, the initial distribution and all probability
    arrays) so that regime 1 is the lowest-variance regime.

    Parameters
    ----------
    A_list, B_list : list of numpy.ndarray
        Per-regime coefficient matrices.
    sigma2_list : list of float
        Per-regime variances used as the sort key.
    P, rho : numpy.ndarray
        Transition matrix and initial distribution.
    filtered, predicted, smoothed, joint_smoothed : numpy.ndarray
        Filter/smoother outputs to reorder.

    Returns
    -------
    tuple
        The same objects, reordered by ascending ``sigma2``.
    """
    order = np.argsort(np.asarray(sigma2_list))

    A_new = [A_list[i] for i in order]
    B_new = [B_list[i] for i in order]
    s_new = [sigma2_list[i] for i in order]
    P_new = P[np.ix_(order, order)]
    rho_new = rho[order]
    filtered_new = filtered[:, order]
    predicted_new = predicted[:, order]
    smoothed_new = smoothed[:, order]

    joint_new = joint_smoothed.copy()
    for t in range(joint_new.shape[0]):
        joint_new[t] = joint_new[t][np.ix_(order, order)]

    return (
        A_new,
        B_new,
        s_new,
        P_new,
        rho_new,
        filtered_new,
        predicted_new,
        smoothed_new,
        joint_new,
    )

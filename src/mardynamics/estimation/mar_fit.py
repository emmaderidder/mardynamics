"""Matrix autoregressive (MAR) model fitting and forecasting.

Implements the active tensor-facing MAR(1) estimators: the static MAR(1)
NumPy projection-initialised alternating-least-squares fit, and the
regime-switching MAR(1) EM estimator (a Gaussian hidden Markov model
over bilinear MAR components) with its one-step forecast rule.

The weighted-ALS building blocks (weighted ALS, HMM filter/smoother,
initialisation) live in ``regime_em``. The canonical coordinates and
the low-rank projection live in ``mar_canonical``. Tensor reshaping
lives in ``mar_tensor``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mardynamics.core.mar_canonical as mc
import mardynamics.core.mar_tensor as mt
import mardynamics.estimation.regime_em as regime_em

# A generalised-EM iteration whose observed log-likelihood falls by more than
# ``GEM_ROLLBACK_TOLERANCE * (1 + |loglik_before|)`` is undone and logged. The
# relative form makes the threshold scale with the likelihood's own magnitude
# instead of with the sample size.
GEM_ROLLBACK_TOLERANCE = 1e-8


def fit_static_mar_tensor_numpy(
    train_tensor: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    stopping_rule: str = "coefficient",
    rank_A: int | None = None,
    rank_B: int | None = None,
    enforce_stability: bool = False,
    max_radius: float = 0.99,
    return_telemetry: bool = False,
):
    """Fit the static MAR(1) least-squares estimator in NumPy.

    This is the canonical MAR-LSE route: projection-initialised
    alternating least squares, the same recursion the simulation study
    uses and every active empirical production route (static full-rank,
    rolling, reduced-rank and regime-switching M-steps, and the
    fixed-regressor wild-bootstrap refit) shares.

    It is not a second alternating-least-squares implementation. The
    recursion is the shared weighted one of
    :func:`mardynamics.estimation.regime_em.estimate_weighted_mar_parameters` at unit
    weights, and the starting value is the shared Kronecker projection
    estimator of :func:`mardynamics.core.mar_canonical.projection_estimator`.

    Parameters
    ----------
    train_tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``.
    max_iter : int, default 200
        Maximum alternating sweeps.
    tol : float, default 1e-6
        Convergence tolerance, read according to ``stopping_rule``.
    stopping_rule : {"coefficient", "criterion"}, default "coefficient"
        What ``tol`` measures.
    rank_A, rank_B : int or None, optional
        Optional low-rank constraints.
    enforce_stability : bool, default False
        Whether the pair is contracted into the stable region.
    max_radius : float, default 0.99
        The stability boundary used when it is.
    return_telemetry : bool, default False
        If True, also return the alternating-least-squares record together with
        the projection estimator's design rank and minimum-norm flag.

    Returns
    -------
    tuple
        ``(A, B)``, or ``(A, B, telemetry)`` when ``return_telemetry`` is set.
    """
    X_prev, Y = mt.split_mar_lagged_tensor(np.asarray(train_tensor, dtype=float))
    projection = mc.projection_estimator(train_tensor)

    A, B, _, telemetry = regime_em.estimate_weighted_mar_parameters(
        X_prev=X_prev,
        Y=Y,
        weights=np.ones(len(X_prev)),
        max_iter=max_iter,
        A_init=projection.A,
        B_init=projection.B,
        tol=tol,
        rank_A=rank_A,
        rank_B=rank_B,
        enforce_stability=enforce_stability,
        max_radius=max_radius,
        stopping_rule=stopping_rule,
        return_telemetry=True,
    )

    if not return_telemetry:
        return A, B

    telemetry = dict(telemetry)
    telemetry["projection_regressor_rank"] = projection.regressor_rank
    telemetry["projection_minimum_norm"] = projection.minimum_norm
    return A, B, telemetry


def _canonical_parameter_delta(
    A_list: list[np.ndarray],
    B_list: list[np.ndarray],
    A_prev: list[np.ndarray],
    B_prev: list[np.ndarray],
) -> float:
    """Maximum absolute change of the canonical parameters across regimes.

    The canonical parameter of a regime is ``(A_bar, B_breve, kappa)``, so this
    is the largest absolute entrywise change of either shape or of the
    log-amplitude, maximised over regimes. It is scale- and sign-free because
    both pairs pass through :func:`mardynamics.core.mar_canonical.canonical_triple` first.

    Parameters
    ----------
    A_list, B_list : list of numpy.ndarray
        Current per-regime operators.
    A_prev, B_prev : list of numpy.ndarray
        The same operators one iteration earlier.

    Returns
    -------
    float
        The maximum absolute canonical-parameter change.
    """
    delta = 0.0
    for A_now, B_now, A_old, B_old in zip(A_list, B_list, A_prev, B_prev):
        A_bar, B_breve, gamma = mc.canonical_triple(A_now, B_now)
        A_bar_old, B_breve_old, gamma_old = mc.canonical_triple(A_old, B_old)
        delta = max(
            delta,
            float(np.max(np.abs(A_bar - A_bar_old))),
            float(np.max(np.abs(B_breve - B_breve_old))),
            abs(float(np.log(gamma) - np.log(gamma_old))),
        )
    return delta


def fit_regime_switching_mar_tensor(
    mar_tensor: np.ndarray,
    static_pair: tuple[np.ndarray, np.ndarray] | None = None,
    n_regimes: int = 2,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 10,
    n_als_iter: int = 20,
    rank_A: int | None = None,
    rank_B: int | None = None,
    tau_init: float = 0.02,
    random_state: int = 42,
    warm_start_params: dict | None = None,
    store_init_diagnostics: bool = True,
    min_iter: int = 10,
    param_tol: float = 1e-3,
    enforce_stability: bool = False,
    max_radius: float = 0.99,
    als_tol: float = 1e-6,
    als_stopping_rule: str = "coefficient",
    param_rule: str = "mixed",
    starts: list[dict] | None = None,
) -> dict:
    """Fit a regime-switching MAR(1) by generalised EM directly on a tensor.

    Each regime is a bilinear MAR(1) with an isotropic scalar noise variance.
    The algorithm alternates a forward-backward E-step over the regime
    sequence with M-steps that update the transition matrix and initial
    distribution in closed form and each regime's ``(A, B)`` by weighted
    alternating least squares.

    It is a generalised EM step in two nested senses, both required by the
    methodology. Inside a regime, the complete operator proposal (conditional
    update, rank projection, Frobenius normalisation, stability contraction)
    is accepted only when it does not increase the variance-weighted
    complete-data operator criterion, and the closed-form variance update
    follows the operator decision. Across the whole iteration, the observed
    log-likelihood is recomputed at the proposed parameters: a numerical
    decrease larger than ``1e-8 * (1 + |loglik_before|)`` is rolled back, the
    iteration is undone, the fit stops with reason ``"rollback"``, and the event
    is recorded. Continuing after a rollback would repeat the identical
    deterministic step, so the stop is not a policy choice.

    Parameters
    ----------
    mar_tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``, at least three time points.
    static_pair : tuple of numpy.ndarray or None, optional
        Full-sample static ``(A, B)`` used to build the residual-split starts.
        Fitted with :func:`fit_static_mar_tensor_numpy` when omitted.
    n_regimes : int, default 2
        Number of regimes.
    max_iter, min_iter : int
        Maximum and minimum EM iterations per start.
    tol : float, default 1e-4
        Relative log-likelihood convergence tolerance.
    n_init : int, default 10
        Number of starts, ignored when ``starts`` is given.
    n_als_iter : int, default 20
        Inner weighted-ALS iterations per M-step.
    rank_A, rank_B : int or None, optional
        Optional low-rank constraints on the per-regime operators.
    tau_init : float, default 0.02
        Perturbation scale of the residual-split starts; ``0.0`` gives the plain
        projection/residual-split start.
    random_state : int, default 42
        Base RNG seed of the residual-split starts.
    warm_start_params : dict or None, optional
        Parameters from a previous fit, used as the first start if given.
    store_init_diagnostics : bool, default True
        If True, attach per-start log-likelihood/convergence summaries.
    param_tol : float, default 1e-3
        Parameter-change tolerance, read according to ``param_rule``.
    enforce_stability : bool, default False
        Whether each operator proposal is contracted into the stable region.
    max_radius : float, default 0.99
        The stability boundary used when it is.
    als_tol : float, default 1e-6
        Tolerance handed to the weighted ALS M-step.
    als_stopping_rule : {"coefficient", "criterion"}, default "coefficient"
        What ``als_tol`` measures inside the M-step.
    param_rule : {"mixed", "canonical"}, default "mixed"
        What ``param_tol`` measures. ``"mixed"`` is the pre-existing rule (the
        largest change over the operator Frobenius norms, the variances, the
        transition matrix and the initial distribution); ``"canonical"`` is the
        maximum absolute canonical-parameter change of
        :func:`_canonical_parameter_delta`.
    starts : list of dict or None, optional
        Explicit starting parameter sets, each with ``A_list``, ``B_list``,
        ``sigma2_list``, ``transition_matrix``, ``initial_probs`` and
        ``init_info``. When given, these replace the residual-split starts
        entirely and ``n_init`` is ignored.

    Returns
    -------
    dict
        The best fit: per-regime ``A_list``/``B_list``/``sigma2_list``, the
        ``transition_matrix`` and ``initial_probs``, the filtered/predicted/
        smoothed (and joint) regime probabilities, ``loglik``, convergence
        diagnostics including ``convergence_reason`` and ``rolled_back``, the
        per-iteration ``gem_telemetry`` records, ``last_observation`` and the
        rank metadata.
    """
    if param_rule not in ("mixed", "canonical"):
        raise ValueError(f"unknown param_rule={param_rule!r}")

    mar_tensor = np.asarray(mar_tensor, dtype=float)
    if mar_tensor.shape[0] < 3:
        raise ValueError("Need at least 3 time points to fit MS-MAR(1).")

    X_prev, Y = mt.split_mar_lagged_tensor(mar_tensor)

    prepared_init = None
    if starts is None:
        if static_pair is None:
            static_pair = fit_static_mar_tensor_numpy(
                mar_tensor,
                max_iter=n_als_iter if n_als_iter > 0 else 200,
                tol=als_tol,
                stopping_rule=als_stopping_rule,
                enforce_stability=enforce_stability,
                max_radius=max_radius,
            )
        prepared_init = {
            "mar_tensor": mar_tensor,
            "X_prev": X_prev,
            "Y": Y,
            "A_static": static_pair[0],
            "B_static": static_pair[1],
            "resid_norm": regime_em.static_residual_norms(X_prev, Y, *static_pair),
        }

    best_result = None
    best_loglik = -np.inf
    init_logliks = [] if store_init_diagnostics else None
    init_summaries = [] if store_init_diagnostics else None

    n_starts = len(starts) if starts is not None else max(n_init, 1)

    for init_id in range(n_starts):
        if starts is not None:
            start = starts[init_id]
            A_list = [a.copy() for a in start["A_list"]]
            B_list = [b.copy() for b in start["B_list"]]
            sigma2_list = [float(s) for s in start["sigma2_list"]]
            transition_matrix = np.asarray(
                start["transition_matrix"], dtype=float
            ).copy()
            initial_probs = np.asarray(start["initial_probs"], dtype=float).copy()
            initial_probs = np.clip(initial_probs, 1e-12, None)
            initial_probs /= initial_probs.sum()
            init_info = dict(start.get("init_info", {}))
            init_info.setdefault("init_id", init_id)

        elif init_id == 0 and warm_start_params is not None:
            A_list = [a.copy() for a in warm_start_params["A_list"]]
            B_list = [b.copy() for b in warm_start_params["B_list"]]
            sigma2_list = [float(s) for s in warm_start_params["sigma2_list"]]
            transition_matrix = warm_start_params["transition_matrix"].copy()

            initial_probs = warm_start_params["initial_probs"].copy()
            initial_probs = np.clip(initial_probs, 1e-12, None)
            initial_probs /= initial_probs.sum()

            init_info = {
                "method": "warm_start_A_B_P_rho",
                "initial_probs": initial_probs.copy(),
                "tau_init": tau_init,
                "init_id": init_id,
            }

        else:
            _, A_list, B_list, sigma2_list, init_info = (
                regime_em.initialise_regimes_from_static_mar(
                    prepared_init=prepared_init,
                    n_regimes=n_regimes,
                    n_als_iter=n_als_iter,
                    rank_A=rank_A,
                    rank_B=rank_B,
                    tau_init=tau_init,
                    random_state=random_state,
                    init_id=init_id,
                    enforce_stability=enforce_stability,
                    max_radius=max_radius,
                )
            )

            transition_matrix = regime_em.initialise_transition_matrix(
                n_regimes, stay_prob=0.90
            )
            initial_probs = np.full(n_regimes, 1.0 / n_regimes)

        converged = False
        rel_loglik_change = np.inf
        param_delta = np.inf
        rolled_back = False
        convergence_reason = "max_iter"
        gem_telemetry = []
        iteration = 0

        log_emissions = regime_em.compute_regime_log_likelihoods(
            X_prev=X_prev, Y=Y, A_list=A_list, B_list=B_list, sigma2_list=sigma2_list
        )
        filtered, predicted, smoothed, joint_smoothed, loglik = (
            regime_em.forward_backward_filter_smoother(
                log_emissions=log_emissions,
                transition_matrix=transition_matrix,
                initial_probs=initial_probs,
            )
        )
        # The log-likelihood at the starting point, before the first E step.
        # It is the only quantity that distinguishes two starts that converge to
        # the same optimum from two starts that were not different to begin
        # with, so it is reported alongside the converged value.
        initial_loglik = loglik

        for iteration in range(1, max_iter + 1):
            loglik_before = loglik
            A_prev_iter = [a.copy() for a in A_list]
            B_prev_iter = [b.copy() for b in B_list]
            sigma2_prev_iter = np.asarray(sigma2_list, dtype=float).copy()
            P_prev_iter = transition_matrix.copy()
            rho_prev_iter = initial_probs.copy()
            smoothed_before = smoothed

            initial_probs = np.clip(smoothed[0].copy(), 1e-12, None)
            initial_probs /= initial_probs.sum()

            denom = smoothed[:-1].sum(axis=0)
            P_new = transition_matrix.copy()

            for j in range(n_regimes):
                if denom[j] > 1e-12:
                    P_new[:, j] = joint_smoothed[1:, :, j].sum(axis=0) / denom[j]

            P_new = np.clip(P_new, 1e-12, None)
            P_new /= P_new.sum(axis=0, keepdims=True)
            transition_matrix = P_new

            step_records = []
            for k in range(n_regimes):
                A_k, B_k, sigma2_k, step = regime_em.estimate_weighted_mar_parameters(
                    X_prev=X_prev,
                    Y=Y,
                    weights=smoothed_before[:, k],
                    max_iter=n_als_iter,
                    tol=als_tol,
                    rank_A=rank_A,
                    rank_B=rank_B,
                    A_init=A_list[k],
                    B_init=B_list[k],
                    enforce_stability=enforce_stability,
                    max_radius=max_radius,
                    stopping_rule=als_stopping_rule,
                    return_telemetry=True,
                )
                A_list[k] = A_k
                B_list[k] = B_k
                sigma2_list[k] = sigma2_k
                step_records.append(step)

            log_emissions = regime_em.compute_regime_log_likelihoods(
                X_prev=X_prev,
                Y=Y,
                A_list=A_list,
                B_list=B_list,
                sigma2_list=sigma2_list,
            )
            filtered, predicted, smoothed, joint_smoothed, loglik = (
                regime_em.forward_backward_filter_smoother(
                    log_emissions=log_emissions,
                    transition_matrix=transition_matrix,
                    initial_probs=initial_probs,
                )
            )

            drop = loglik_before - loglik
            record = {
                "iteration": iteration,
                "loglik_before": float(loglik_before),
                "loglik_after": float(loglik),
                "rolled_back": bool(
                    drop > GEM_ROLLBACK_TOLERANCE * (1.0 + abs(loglik_before))
                ),
            }
            for k, step in enumerate(step_records):
                for key, value in step.items():
                    record[f"regime{k}_{key}"] = value
            gem_telemetry.append(record)

            if record["rolled_back"]:
                A_list = A_prev_iter
                B_list = B_prev_iter
                sigma2_list = list(sigma2_prev_iter)
                transition_matrix = P_prev_iter
                initial_probs = rho_prev_iter
                log_emissions = regime_em.compute_regime_log_likelihoods(
                    X_prev=X_prev,
                    Y=Y,
                    A_list=A_list,
                    B_list=B_list,
                    sigma2_list=sigma2_list,
                )
                filtered, predicted, smoothed, joint_smoothed, loglik = (
                    regime_em.forward_backward_filter_smoother(
                        log_emissions=log_emissions,
                        transition_matrix=transition_matrix,
                        initial_probs=initial_probs,
                    )
                )
                rolled_back = True
                convergence_reason = "rollback"
                break

            rel_loglik_change = abs(loglik - loglik_before) / (1.0 + abs(loglik_before))

            if param_rule == "canonical":
                param_delta = _canonical_parameter_delta(
                    A_list, B_list, A_prev_iter, B_prev_iter
                )
            else:
                A_deltas = [
                    np.linalg.norm(A_list[k] - A_prev_iter[k], ord="fro")
                    for k in range(n_regimes)
                ]
                B_deltas = [
                    np.linalg.norm(B_list[k] - B_prev_iter[k], ord="fro")
                    for k in range(n_regimes)
                ]
                param_delta = max(
                    max(A_deltas),
                    max(B_deltas),
                    float(
                        np.max(
                            np.abs(
                                np.asarray(sigma2_list, dtype=float) - sigma2_prev_iter
                            )
                        )
                    ),
                    float(np.max(np.abs(transition_matrix - P_prev_iter))),
                    float(np.max(np.abs(initial_probs - rho_prev_iter))),
                )

            if (
                iteration >= min_iter
                and rel_loglik_change < tol
                and param_delta < param_tol
            ):
                converged = True
                convergence_reason = "loglik_and_param_tol"
                break

        (
            A_list,
            B_list,
            sigma2_list,
            transition_matrix,
            initial_probs,
            filtered,
            predicted,
            smoothed,
            joint_smoothed,
        ) = regime_em.sort_regimes_by_sigma2(
            A_list=A_list,
            B_list=B_list,
            sigma2_list=sigma2_list,
            P=transition_matrix,
            rho=initial_probs,
            filtered=filtered,
            predicted=predicted,
            smoothed=smoothed,
            joint_smoothed=joint_smoothed,
        )

        result = {
            "n_regimes": n_regimes,
            "A_list": A_list,
            "B_list": B_list,
            "sigma2_list": sigma2_list,
            "transition_matrix": transition_matrix,
            "initial_probs": initial_probs,
            "filtered_probs": filtered,
            "predicted_probs": predicted,
            "smoothed_probs": smoothed,
            "joint_smoothed_probs": joint_smoothed,
            "loglik": loglik,
            "initial_loglik": initial_loglik,
            "n_iter": iteration,
            "converged": converged,
            "convergence_reason": convergence_reason,
            "rolled_back": rolled_back,
            "gem_telemetry": gem_telemetry,
            "gem_rejected_steps": int(
                sum(
                    1
                    for rec in gem_telemetry
                    for key, value in rec.items()
                    if key.endswith("_accepted") and not value
                )
            ),
            "occupancy": smoothed.sum(axis=0),
            "n_eff": (smoothed.sum(axis=0) ** 2) / (smoothed**2).sum(axis=0),
            "last_observation": mt.get_last_mar_tensor_observation(mar_tensor),
            "init_info": init_info,
            "init_id": init_id,
            "rank_A": rank_A,
            "rank_B": rank_B,
            "final_rel_loglik_change": rel_loglik_change,
            "final_param_delta": param_delta,
        }

        if store_init_diagnostics:
            init_logliks.append(
                {"init_id": init_id, "loglik": loglik, "converged": converged}
            )
            init_summaries.append(
                {
                    "init_id": init_id,
                    "loglik": loglik,
                    "converged": converged,
                    "n_iter": iteration,
                    "p11": transition_matrix[0, 0] if n_regimes >= 1 else np.nan,
                    "p22": transition_matrix[1, 1] if n_regimes >= 2 else np.nan,
                }
            )

        if result["loglik"] > best_loglik:
            best_loglik = result["loglik"]
            best_result = result

    if store_init_diagnostics:
        best_result["all_init_logliks"] = (
            pd.DataFrame(init_logliks)
            .sort_values("loglik", ascending=False)
            .reset_index(drop=True)
        )
        best_result["all_init_summaries"] = pd.DataFrame(init_summaries)

    return best_result


def forecast_next_from_regime_switching_fit(
    fit_result: dict,
) -> tuple[pd.Series, np.ndarray]:
    """Compute the one-step-ahead forecast from a fitted regime-switching MAR.

    The next-period regime distribution is ``P @ filtered[-1]`` (using the
    filtered, not smoothed, probabilities, so no future information is used), and
    the forecast is the regime-probability-weighted mean ``sum_k p_k A_k X_T B_kᵀ``.

    Parameters
    ----------
    fit_result : dict
        A fit returned by :func:`fit_regime_switching_mar_tensor`.

    Returns
    -------
    forecast_series : pandas.Series
        One-step-ahead forecast indexed by ``variable_country`` labels.
    next_regime_probs : numpy.ndarray
        The predicted next-period regime distribution.
    """
    next_regime_probs = (
        fit_result["transition_matrix"] @ fit_result["filtered_probs"][-1]
    )
    next_regime_probs = np.clip(next_regime_probs, 1e-12, None)
    next_regime_probs /= next_regime_probs.sum()

    X_last = fit_result["last_observation"]
    forecast_matrix = np.zeros_like(X_last, dtype=float)

    for k in range(fit_result["n_regimes"]):
        A_k = fit_result["A_list"][k]
        B_k = fit_result["B_list"][k]
        forecast_matrix += next_regime_probs[k] * (A_k @ X_last @ B_k.T)

    return (
        mt.flatten_mar_forecast_matrix(
            forecast_matrix=forecast_matrix,
            variables=fit_result["variables"],
            countries=fit_result["countries"],
        ),
        next_regime_probs,
    )

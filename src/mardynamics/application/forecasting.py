"""Tuning grids and recursive forecasts.

Every model is re-estimated at every origin from information available
before the forecasted observation. The estimation slice of a target month
is ``tensor[:index]``, so the observation being forecast and everything
after it are unreachable by construction. The rolling models take the
last ``w`` matrices of that slice, giving an effective sample size of
``w-1``.

MS-MAR fits rebuild all eight fresh indexed starts at every origin and
do not warm-start from the previous one. Their forecasts use the one-step
predicted state distribution through the shared forecast rule, which
averages the regime-specific conditional means using that distribution.

The hard-failure list determines which conditions end a unit. A
successful stability contraction is telemetry, not a failure. A finite
admissible fit that stopped at the iteration cap is soft non-convergence:
its forecast is retained and flagged, but not used for structural
interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import mardynamics.application.data as data_mod
import mardynamics.application.emp_config as emp_config
import mardynamics.application.estimators as estimators
import mardynamics.application.run_infrastructure as runinfra
import mardynamics.estimation.mar_fit as mar_fit
import mardynamics.core.mar_tensor as mar_tensor


class HardFailure(RuntimeError):
    """A hard failure of one model at one origin."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(f"{reason}: {message}" if message else reason)
        self.reason = reason


@dataclass(frozen=True)
class ModelSpec:
    """One candidate configuration of the 3 by 2 design.

    Attributes
    ----------
    model : str
        Model key from :data:`mardynamics.application.emp_config.MODELS`.
    rank : int or None
        Common rank of the two operators, for reduced-rank models.
    window : int or None
        Rolling window length in matrices, for rolling models.
    """

    model: str
    rank: int | None = None
    window: int | None = None

    @property
    def label(self) -> str:
        """Full logical identity of the configuration.

        Returns
        -------
        str
            A stable label used for seeding and reporting.
        """
        parts = [self.model]
        if self.window is not None:
            parts.append(f"w{self.window}")
        if self.rank is not None:
            parts.append(f"r{self.rank}")
        return "_".join(parts)


def candidate_specs() -> list[ModelSpec]:
    """Return every candidate the validation stage must evaluate.

    Builds the reduced-rank static and MS-MAR candidates over RANK_GRID,
    the rolling full-rank candidates over WINDOW_GRID, and the rolling
    reduced-rank candidates over every ``(window, rank)`` pair in
    WINDOW_GRID x RANK_GRID. FULL_RANK is the separate full-rank comparator
    and not a reduced-rank candidate.

    Returns
    -------
    list of ModelSpec
        The candidates, grouped by model.
    """
    specs = [ModelSpec("static_full")]
    specs += [ModelSpec("static_reduced", rank=r) for r in emp_config.RANK_GRID]
    specs += [ModelSpec("rolling_full", window=w) for w in emp_config.WINDOW_GRID]
    specs += [
        ModelSpec("rolling_reduced", window=w, rank=r)
        for w in emp_config.WINDOW_GRID
        for r in emp_config.RANK_GRID
    ]
    specs.append(ModelSpec("ms_full"))
    specs += [ModelSpec("ms_reduced", rank=r) for r in emp_config.RANK_GRID]
    return specs


def _joint_spectral_radius(A: np.ndarray, B: np.ndarray) -> float:
    """Return ``rho(A) rho(B)``.

    Parameters
    ----------
    A, B : numpy.ndarray
        The operator pair.

    Returns
    -------
    float
        The joint spectral radius.
    """
    return float(
        np.max(np.abs(np.linalg.eigvals(A))) * np.max(np.abs(np.linalg.eigvals(B)))
    )


def _validate_operator_pair(A: np.ndarray, B: np.ndarray) -> float:
    """Validate one fitted pair against the hard-failure list.

    Parameters
    ----------
    A, B : numpy.ndarray
        The operator pair.

    Returns
    -------
    float
        The joint spectral radius, for telemetry.

    Raises
    ------
    HardFailure
        On a malformed dimension, a non-finite parameter or a post-correction
        radius above the maximum.
    """
    if A.shape != (emp_config.M_DIM, emp_config.M_DIM) or B.shape != (emp_config.N_DIM, emp_config.N_DIM):
        raise HardFailure("malformed_dimension", f"{A.shape} and {B.shape}")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise HardFailure("non_finite_parameter")
    radius = _joint_spectral_radius(A, B)
    if radius > emp_config.MAX_RADIUS + 1e-8:
        raise HardFailure("post_correction_radius_above_maximum", f"{radius:.6f}")
    return radius


# =========================================================
# Inferred solver telemetry
# =========================================================
def contraction_target() -> float:
    """Return the joint spectral radius a stability contraction lands on.

    ``mardynamics.core.stability.project_to_stable`` is a soft projection: it
    returns the pair untouched while the radius is below ``max_radius``, and
    when it acts it rescales until the radius reaches
    ``max_radius * (1 - safety)``. The safety margin is read from that
    function's signature.

    Returns
    -------
    float
        The contraction target.
    """
    import inspect

    import mardynamics.core.stability as stability

    safety = inspect.signature(stability.project_to_stable).parameters[
        "safety"
    ].default
    return float(emp_config.MAX_RADIUS * (1.0 - safety))


# Written into ``convergence_reason`` wherever a fitting route's
# convergence was not observed. No active route currently produces this
# value; it is the fallback label used by
# mardynamics.application.descriptions.telemetry_table's aggregate branch
# if a future route were added without convergence reporting.
NOT_ASSESSABLE_REASON = "not_assessable_by_any_active_route"

# The status written for a fit whose convergence was not observed.
NOT_ASSESSABLE_STATUS = (
    "not_assessable: this route's convergence was never observed; it is "
    "neither certified reliable nor flagged as non-convergent"
)

# Stated wherever the contraction flag is inferred instead of reported.
CONTRACTION_PROVENANCE = (
    "inferred: the estimator exposes no contraction flag, so the value "
    "is inferred from the joint spectral radius landing on the fixed "
    "contraction target max_radius * (1 - safety)"
)

# Relative tolerance of the fingerprint. A contraction reaches the target
# to about 3e-15, so this is many orders of magnitude tighter than any
# radius an uncontracted fit could reach by coincidence.
CONTRACTION_TOLERANCE = 1e-12


def inferred_contraction(radius: float) -> bool:
    """Report whether a fitted radius carries the contraction fingerprint.

    Parameters
    ----------
    radius : float
        The fitted joint spectral radius.

    Returns
    -------
    bool
        True when the radius equals the contraction target.
    """
    target = contraction_target()
    return bool(
        abs(float(radius) - target) <= CONTRACTION_TOLERANCE * max(1.0, target)
    )


def _als_stopping(
    history: np.ndarray,
    rank: int | None,
    iterations: int,
    max_iter: int = emp_config.ALS_MAX_ITER,
    tol: float = emp_config.ALS_TOL,
    stopping_rule: str = emp_config.ALS_STOPPING_RULE,
) -> tuple:
    """Decide whether an ALS fit stopped on its criterion or at the cap.

    The solver reports only how many iterations it completed. Fewer than
    the cap can only mean the relative criterion change fell below the
    tolerance. Exactly the cap is ambiguous: the criterion may have been
    met on the very last iteration, or it may not have been met at all. To
    resolve this, the same deterministic solve is repeated once with one
    extra iteration allowed. If it then uses that extra iteration the cap
    was binding; if it still stops at the cap, the criterion was met there.

    The probe is diagnostic only and does not affect the returned fit: the
    fit that produces the forecast is the one run at the iteration cap.
    Shared by every route through the canonical NumPy kernel: the static
    full-rank route probes at ``emp_config.STATIC_MAX_ITER``, the
    reduced-rank and rolling routes at ``emp_config.ALS_MAX_ITER``.

    Parameters
    ----------
    history : numpy.ndarray
        The estimation slice the fit used.
    rank : int or None
        The common rank the fit imposed.
    iterations : int
        Iterations the fit completed.
    max_iter : int, optional
        The route's iteration cap.
    tol : float, optional
        The route's tolerance.
    stopping_rule : str, optional
        The route's stopping rule.

    Returns
    -------
    tuple
        ``(converged, convergence_reason)``.
    """
    if iterations < max_iter:
        return True, "criterion"

    import mardynamics.estimation.mar_fit as models

    probe = models.fit_static_mar_tensor_numpy(
        history,
        max_iter=max_iter + 1,
        tol=tol,
        stopping_rule=stopping_rule,
        rank_A=rank,
        rank_B=rank,
        enforce_stability=emp_config.ENFORCE_STABILITY,
        max_radius=emp_config.MAX_RADIUS,
        return_telemetry=True,
    )[2]
    if int(probe["als_iterations"]) > max_iter:
        return False, "max_iter"
    return True, "criterion"


def fit_and_forecast(spec: ModelSpec, tensor: np.ndarray, index: int) -> dict:
    """Fit one configuration at one origin and forecast the next matrix.

    Parameters
    ----------
    spec : ModelSpec
        The configuration.
    tensor : numpy.ndarray
        The standardised tensor of the phase's preprocessing version.
    index : int
        Position of the month being forecast. Only ``tensor[:index]`` is used.

    Returns
    -------
    dict
        The forecast matrix and the fit's telemetry.

    Raises
    ------
    HardFailure
        On any condition the hard-failure list covers.
    """
    history = tensor[:index]
    if history.shape[0] < 3:
        raise HardFailure("insufficient_history", f"{history.shape[0]} matrices")

    telemetry = {
        "converged": True,
        "convergence_reason": "",
        "n_iter": 0,
        "gem_rejected_steps": 0,
        "rolled_back": False,
        "likelihood_monotone": True,
        "stability_contracted": False,
        "soft_non_convergence": False,
        "convergence_assessed": True,
        "effective_sample_size": float(history.shape[0] - 1),
    }

    if spec.model.startswith("rolling"):
        if spec.window is None or history.shape[0] < spec.window:
            raise HardFailure(
                "window_infeasible",
                f"w={spec.window} needs {spec.window} matrices, "
                f"{history.shape[0]} available",
            )
        history = history[-spec.window :]
        telemetry["effective_sample_size"] = float(spec.window - 1)

    last = history[-1]

    if spec.model == "static_full":
        A, B, solver = estimators.static_fit(history, return_telemetry=True)
        # Routed through the same canonical NumPy kernel as every
        # reduced-rank and rolling fit, so convergence is observable here:
        # the solver reports its own iteration count and criterion history,
        # at the STATIC_MAX_ITER/STATIC_TOL/STATIC_STOPPING_RULE settings.
        # This route is the benchmark every relative MSFE and forecast-test
        # differential is measured against.
        iterations = int(solver["als_iterations"])
        converged, reason = _als_stopping(
            history,
            None,
            iterations,
            max_iter=emp_config.STATIC_MAX_ITER,
            tol=emp_config.STATIC_TOL,
            stopping_rule=emp_config.STATIC_STOPPING_RULE,
        )
        telemetry["n_iter"] = iterations
        telemetry["converged"] = converged
        telemetry["convergence_reason"] = reason
        telemetry["soft_non_convergence"] = reason == "max_iter"
        telemetry["als_criterion_before"] = float(solver["criterion_before"])
        telemetry["als_criterion_after"] = float(solver["criterion_after"])
        telemetry["als_step_accepted"] = bool(solver["accepted"])
        telemetry["estimator_identity"] = solver["estimator_identity"]
    elif spec.model in ("static_reduced", "rolling_full", "rolling_reduced"):
        A, B, solver = estimators.als_fit(
            history, rank=spec.rank, return_telemetry=True
        )
        # The solver's telemetry has six keys: criterion_before,
        # criterion_after, accepted, als_iterations, occupancy and n_eff,
        # plus the two projection fields the tensor wrapper adds. It has no
        # iteration-count, convergence or stopping-rule key, so those are
        # computed separately below.
        iterations = int(solver["als_iterations"])
        converged, reason = _als_stopping(history, spec.rank, iterations)
        telemetry["n_iter"] = iterations
        telemetry["converged"] = converged
        telemetry["convergence_reason"] = reason
        # A finite admissible fit that stopped at the iteration cap is soft
        # non-convergence: retained, flagged, and not used for structural
        # interpretation.
        telemetry["soft_non_convergence"] = reason == "max_iter"
        telemetry["als_criterion_before"] = float(solver["criterion_before"])
        telemetry["als_criterion_after"] = float(solver["criterion_after"])
        telemetry["als_step_accepted"] = bool(solver["accepted"])
        telemetry["estimator_identity"] = solver["estimator_identity"]
    elif spec.model in ("ms_full", "ms_reduced"):
        return _regime_forecast(spec, history, index, telemetry)
    else:  # pragma: no cover - the registry cannot produce another key
        raise HardFailure("unknown_model", spec.model)

    radius = _validate_operator_pair(np.asarray(A), np.asarray(B))
    telemetry["joint_spectral_radius"] = radius
    telemetry["stability_contracted"] = inferred_contraction(radius)
    telemetry["stability_contracted_provenance"] = CONTRACTION_PROVENANCE
    forecast = np.asarray(A) @ last @ np.asarray(B).T
    if not np.isfinite(forecast).all():
        raise HardFailure("non_finite_forecast")

    # The canonical triple of every origin travels with its forecast
    # record, so the rolling canonical path can be built from the stored
    # fits.
    telemetry.update(_canonical_fields(np.asarray(A), np.asarray(B)))

    return {"forecast": forecast, "A": np.asarray(A), "B": np.asarray(B), **telemetry}


def _canonical_fields(A: np.ndarray, B: np.ndarray) -> dict:
    """Flatten the canonical triple of one fit into record columns.

    Parameters
    ----------
    A, B : numpy.ndarray
        The fitted operator pair.

    Returns
    -------
    dict
        The amplitude and the two shape matrices, entry by entry.
    """
    from mardynamics.core.mar_canonical import canonical_triple

    A_bar, B_breve, gamma = canonical_triple(A, B)
    fields = {"gamma": float(gamma)}
    for i in range(emp_config.M_DIM):
        for j in range(emp_config.M_DIM):
            fields[f"A_bar_{i}_{j}"] = float(A_bar[i, j])
    for i in range(emp_config.N_DIM):
        for j in range(emp_config.N_DIM):
            fields[f"B_breve_{i}_{j}"] = float(B_breve[i, j])
    return fields


def _regime_forecast(
    spec: ModelSpec, history: np.ndarray, index: int, telemetry: dict
) -> dict:
    """Fit the MS-MAR at one origin and forecast from the predicted state.

    Parameters
    ----------
    spec : ModelSpec
        The configuration.
    history : numpy.ndarray
        The estimation slice.
    index : int
        Position of the month being forecast, part of the fit's identity.
    telemetry : dict
        Telemetry accumulator.

    Returns
    -------
    dict
        The forecast and the fit's telemetry.

    Raises
    ------
    HardFailure
        On no admissible start, a non-finite likelihood, an invalid probability
        object, a GEM monotonicity violation or a non-finite forecast.
    """
    fit_label = f"{spec.label}@{index}"
    fit = estimators.regime_fit(history, fit_label, rank=spec.rank)

    if not np.isfinite(fit["loglik"]):
        raise HardFailure("non_finite_likelihood")
    if fit.get("convergence_reason") == "rollback":
        raise HardFailure("gem_monotonicity_violation", "rolled back")

    probabilities = np.asarray(fit["filtered_probs"], dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or probabilities.min() < 0.0
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise HardFailure("invalid_probability_object")

    # _validate_operator_pair returns the joint spectral radius it validated.
    # Keeping the maximum over the regimes gives the MS models the same
    # telemetry column every other model already carries.
    radii = [
        _validate_operator_pair(np.asarray(A), np.asarray(B))
        for A, B in zip(fit["A_list"], fit["B_list"])
    ]
    telemetry["joint_spectral_radius"] = float(max(radii))
    telemetry["stability_contracted"] = any(
        inferred_contraction(value) for value in radii
    )
    telemetry["stability_contracted_provenance"] = CONTRACTION_PROVENANCE

    labelled = dict(fit)
    labelled["variables"] = list(emp_config.INDICATORS)
    labelled["countries"] = list(emp_config.COUNTRIES)
    flat, predicted = mar_fit.forecast_next_from_regime_switching_fit(labelled)
    forecast = flat.to_numpy(dtype=float).reshape(emp_config.M_DIM, emp_config.N_DIM, order="C")
    if not np.isfinite(forecast).all():
        raise HardFailure("non_finite_forecast")

    occupancy = np.asarray(fit["occupancy"], dtype=float)
    telemetry.update(
        {
            "converged": bool(fit["converged"]),
            "convergence_reason": str(fit["convergence_reason"]),
            "n_iter": int(fit["n_iter"]),
            "gem_rejected_steps": int(fit["gem_rejected_steps"]),
            "rolled_back": bool(fit["rolled_back"]),
            "likelihood_monotone": not bool(fit["rolled_back"]),
            "soft_non_convergence": fit["convergence_reason"] == "max_iter",
            "loglik": float(fit["loglik"]),
            "retained_init_id": int(fit["init_id"]),
            "n_starts_used": int(len(fit["all_init_summaries"])),
            "effective_sample_size": float(np.min(fit["n_eff"])),
        }
    )
    # The conditioning variable is the ex-ante predicted probability of the
    # high-variance regime. The shared estimator orders regimes by
    # increasing innovation variance, so it is the last entry.
    telemetry["p_high_predicted"] = float(predicted[-1])
    telemetry["predicted_probs"] = [float(p) for p in predicted]
    telemetry["estimator_identity"] = fit["estimator_identity"]

    return {"forecast": forecast, "fit": fit, **telemetry}


def _run_one_forecast_unit(
    spec: ModelSpec,
    fit_tensor: np.ndarray,
    eval_tensor: np.ndarray,
    fit_preprocessor,
    eval_preprocessor,
    index: int,
    period: str,
) -> dict:
    """Execute one forecast unit and convert a hard failure into a record.

    The model is fit on ``fit_tensor`` (``fit_preprocessor``'s scale). The
    prediction is mapped onto ``eval_preprocessor``'s scale through
    :func:`mardynamics.application.data.map_scale` and differenced against
    the actual read from ``eval_tensor``, not derived by mapping the
    fitting tensor's own actual. For validation, ``fit_tensor is
    eval_tensor`` and the two preprocessors are the same object, so the
    mapping is the identity.

    Parameters
    ----------
    spec : ModelSpec
        The configuration.
    fit_tensor : numpy.ndarray
        The standardised tensor of the phase's fitting preprocessing version.
    eval_tensor : numpy.ndarray
        The standardised tensor of the phase's evaluation preprocessing
        version; its ``[index]`` matrix is the evaluation-scale actual.
    fit_preprocessor, eval_preprocessor : mardynamics.application.data.Preprocessor
        The parameter sets of the fitting and evaluation scales.
    index : int
        Position of the forecast target.
    period : str
        The target month.

    Returns
    -------
    dict
        One record, successful or failed.
    """
    runinfra.enforce_thread_limits()
    record = {
        "model": spec.model,
        "candidate": spec.label,
        "rank": spec.rank,
        "window": spec.window,
        "period": period,
        "target_index": index,
        "fitting_preprocessor": fit_preprocessor.version,
        "evaluation_preprocessor": eval_preprocessor.version,
    }
    try:
        outcome = fit_and_forecast(spec, fit_tensor, index)
    except HardFailure as failure:
        return {
            **record,
            "status": "failed",
            "failure_reason": failure.reason,
            "failure_message": str(failure),
        }
    except Exception as error:  # noqa: BLE001 - part of the hard-failure list
        return {
            **record,
            "status": "failed",
            "failure_reason": "exception",
            "failure_message": f"{type(error).__name__}: {error}",
        }

    prediction_fit_scale = outcome["forecast"]
    prediction_eval_scale = data_mod.map_scale(
        prediction_fit_scale, fit_preprocessor, eval_preprocessor, period
    )
    actual_eval_scale = eval_tensor[index]
    error_matrix = actual_eval_scale - prediction_eval_scale
    loss = float(np.sum(error_matrix**2))
    if not np.isfinite(loss):
        return {
            **record,
            "status": "failed",
            "failure_reason": "non_finite_loss",
            "failure_message": "",
        }

    cellwise = (error_matrix**2).reshape(-1, order="C")
    return {
        **record,
        "status": "ok",
        "failure_reason": "",
        "failure_message": "",
        "loss": loss,
        **{
            f"se_{indicator}_{country}": float(cellwise[i * emp_config.N_DIM + j])
            for i, indicator in enumerate(emp_config.INDICATORS)
            for j, country in enumerate(emp_config.COUNTRIES)
        },
        **{
            f"pred_fit_{indicator}_{country}": float(
                prediction_fit_scale[i, j]
            )
            for i, indicator in enumerate(emp_config.INDICATORS)
            for j, country in enumerate(emp_config.COUNTRIES)
        },
        **{
            f"pred_eval_{indicator}_{country}": float(
                prediction_eval_scale[i, j]
            )
            for i, indicator in enumerate(emp_config.INDICATORS)
            for j, country in enumerate(emp_config.COUNTRIES)
        },
        **{
            f"actual_eval_{indicator}_{country}": float(
                actual_eval_scale[i, j]
            )
            for i, indicator in enumerate(emp_config.INDICATORS)
            for j, country in enumerate(emp_config.COUNTRIES)
        },
        **{
            key: value
            for key, value in outcome.items()
            if key not in ("forecast", "fit", "A", "B", "predicted_probs")
        },
    }


def run_units(
    specs: list[ModelSpec],
    fit_tensor: np.ndarray,
    eval_tensor: np.ndarray,
    fit_preprocessor,
    eval_preprocessor,
    periods: list[str],
    targets: list[str],
    workers: int,
) -> pd.DataFrame:
    """Execute every (configuration, origin) unit of one phase.

    Parameters
    ----------
    specs : list of ModelSpec
        Configurations to evaluate.
    fit_tensor : numpy.ndarray
        The standardised tensor of the phase's fitting preprocessing version.
    eval_tensor : numpy.ndarray
        The standardised tensor of the phase's evaluation preprocessing
        version.
    fit_preprocessor, eval_preprocessor : mardynamics.application.data.Preprocessor
        The parameter sets of the fitting and evaluation scales.
    periods : list of str
        The ordered months of the tensors (shared by every preprocessing
        version, since all three stand on the same date index).
    targets : list of str
        The months to forecast.
    workers : int
        Process workers, capped at MAX_WORKERS.

    Returns
    -------
    pandas.DataFrame
        One record per unit.
    """
    from joblib import Parallel, delayed

    position = {period: index for index, period in enumerate(periods)}
    jobs = [
        (spec, position[target], target) for spec in specs for target in targets
    ]
    n_jobs = max(1, min(int(workers), emp_config.MAX_WORKERS))

    if n_jobs == 1:
        records = [
            _run_one_forecast_unit(
                spec,
                fit_tensor,
                eval_tensor,
                fit_preprocessor,
                eval_preprocessor,
                index,
                period,
            )
            for spec, index, period in jobs
        ]
    else:
        records = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(_run_one_forecast_unit)(
                spec,
                fit_tensor,
                eval_tensor,
                fit_preprocessor,
                eval_preprocessor,
                index,
                period,
            )
            for spec, index, period in jobs
        )

    return pd.DataFrame(records)

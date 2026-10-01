"""Describes the selected rolling and regime-switching estimates.

Builds the rolling canonical path and its canonical increments from the
stored test-origin fits, and describes the full-sample regime-switching
fit: the per-regime canonical triples, the transition matrix, the expected
regime durations, the multi-start likelihoods and the solver diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mardynamics.application.emp_config as emp_config
import mardynamics.application.forecasting as forecasting
import mardynamics.core.structural_tests as structural_tests
from mardynamics.core.mar_canonical import (
    align_canonical_path,
    canonical_path_increments,
    canonical_triple,
)

DISTANCE_REFERENCE = (
    "static forecast benchmark refitted on the test-preprocessing scale; not "
    "the full-sample static reference, which is on a different scale and "
    "may not be differenced against these estimates"
)

NOT_EXPOSED = "not exposed by the estimator"


def _matrix_from_record(row: pd.Series, prefix: str, size: int) -> np.ndarray:
    """Rebuild one stored shape matrix from its flattened record columns.

    Parameters
    ----------
    row : pandas.Series
        One stored forecast record.
    prefix : str
        Column prefix, "A_bar" or "B_breve".
    size : int
        Side length of the square matrix.

    Returns
    -------
    numpy.ndarray
        The matrix.
    """
    return np.array(
        [[float(row[f"{prefix}_{i}_{j}"]) for j in range(size)] for i in range(size)]
    )


def _shape_entries(A_bar: np.ndarray, B_breve: np.ndarray) -> dict:
    """Flatten one aligned canonical pair into its named entries.

    Parameters
    ----------
    A_bar : numpy.ndarray
        Aligned row shape, indicators by indicators.
    B_breve : numpy.ndarray
        Aligned column shape, countries by countries.

    Returns
    -------
    dict
        Mapping from column name to entry value.
    """
    entries = {}
    for i, predicted in enumerate(emp_config.INDICATORS):
        for j, predictor in enumerate(emp_config.INDICATORS):
            entries[f"A_bar_{predicted}_{predictor}"] = float(A_bar[i, j])
    for i, predicted in enumerate(emp_config.COUNTRIES):
        for j, predictor in enumerate(emp_config.COUNTRIES):
            entries[f"B_breve_{predicted}_{predictor}"] = float(B_breve[i, j])
    return entries


# =========================================================
# The rolling MAR path
# =========================================================
def rolling_description(
    records: pd.DataFrame, window: int, static_pair
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the rolling canonical path and its canonical increments.

    The path is sequentially sign-aligned before the increments are taken:
    the canonical sign can jump between origins when the sign-defining
    coordinate crosses zero, and aligning both shapes together leaves the
    Kronecker product unchanged.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records of the selected rolling candidate.
    window : int
        Rolling window length, held fixed over all test origins.
    static_pair : tuple of numpy.ndarray
        Static benchmark fit used as the reference point for the two
        distance columns.

    Returns
    -------
    tuple of pandas.DataFrame
        The rolling path and the rolling increments.
    """
    ordered = records[records["status"] == "ok"].sort_values("period")
    A_static, B_static = canonical_triple(*static_pair)[:2]

    A_list, B_list, gammas, periods = [], [], [], []
    for _, row in ordered.iterrows():
        A_list.append(_matrix_from_record(row, "A_bar", emp_config.M_DIM))
        B_list.append(_matrix_from_record(row, "B_breve", emp_config.N_DIM))
        gammas.append(float(row["gamma"]))
        periods.append(str(row["period"]))

    A_list, B_list = align_canonical_path(A_list, B_list)
    increments = canonical_path_increments(A_list, B_list, gammas, index=periods)

    path_rows = []
    for position, (period, row) in enumerate(zip(periods, ordered.itertuples())):
        A_bar, B_breve = A_list[position], B_list[position]
        path_rows.append(
            {
                "period": period,
                "window": int(window),
                "gamma": gammas[position],
                "log_gamma": float(np.log(gammas[position])),
                "joint_spectral_radius": float(
                    getattr(row, "joint_spectral_radius", np.nan)
                ),
                "row_shape_distance_to_test_scale_benchmark": float(
                    np.linalg.norm(A_bar - A_static, ord="fro")
                ),
                "column_shape_distance_to_test_scale_benchmark": float(
                    np.linalg.norm(B_breve - B_static, ord="fro")
                ),
                "distance_reference": DISTANCE_REFERENCE,
                "effective_sample_size": float(window - 1),
                **_shape_entries(A_bar, B_breve),
                "stability_contracted": bool(
                    getattr(row, "stability_contracted", False)
                ),
                "stability_contracted_provenance": forecasting.CONTRACTION_PROVENANCE,
                "converged": bool(getattr(row, "converged", True)),
                "convergence_reason": str(getattr(row, "convergence_reason", "")),
                "n_iter": int(getattr(row, "n_iter", 0)),
                "soft_non_convergence": bool(
                    getattr(row, "soft_non_convergence", False)
                ),
            }
        )

    interpretation = (
        "path of the forecast-selected recency-pooling rule at a fixed window; "
        "overlapping windows make consecutive canonical increments strongly "
        "dependent, the window delays and smooths movement, peaks are not break "
        "estimates and the path is not the true parameter path"
    )
    increment_rows = pd.DataFrame(
        {
            "period": periods,
            "window": int(window),
            "d_row": increments["d_A"].to_numpy(dtype=float),
            "d_column": increments["d_B"].to_numpy(dtype=float),
            "d_gamma": increments["d_gamma"].to_numpy(dtype=float),
            "forecast_selected_pooling_rule": interpretation,
        }
    )

    return pd.DataFrame(path_rows), increment_rows


# =========================================================
# The regime-switching fit
# =========================================================
def interpretation_conditions(fit: dict) -> dict:
    """Evaluate four conditions on a regime-switching fit.

    The conditions are poor regime occupancy, low effective sample size,
    multi-start instability and an EM rollback. Each threshold is a
    constant in emp_config, reported alongside the outcome so the fired
    conditions and their values are both visible.

    Parameters
    ----------
    fit : dict
        Result of a regime-switching fit.

    Returns
    -------
    dict
        Each condition's boolean, the thresholds, the measured quantities
        and their combination.
    """
    occupancy = np.asarray(fit["occupancy"], dtype=float)
    n_eff = np.asarray(fit["n_eff"], dtype=float)
    occupancy_share = occupancy / occupancy.sum()

    summaries = fit["all_init_summaries"]
    logliks = np.asarray(summaries["loglik"], dtype=float)
    n_built = int(logliks.size)
    best = float(np.max(logliks))
    tolerance = emp_config.REGIME_MULTISTART_LOGLIK_TOL * (1.0 + abs(best))
    n_agreeing = int(np.sum(np.abs(logliks - best) < tolerance))
    agreeing_fraction = n_agreeing / n_built if n_built else 0.0

    poor_occupancy = bool(occupancy_share.min() < emp_config.REGIME_MIN_OCCUPANCY_SHARE)
    low_effective_sample = bool(n_eff.min() < emp_config.REGIME_MIN_EFFECTIVE_SAMPLE)
    multistart_instability = bool(
        agreeing_fraction < emp_config.REGIME_MULTISTART_MIN_AGREEING_FRACTION
    )
    gem_problem = bool(fit["rolled_back"]) or fit["convergence_reason"] == "max_iter"

    fired = [
        name
        for name, value in (
            ("poor_occupancy", poor_occupancy),
            ("low_effective_sample", low_effective_sample),
            ("multistart_instability", multistart_instability),
            ("gem_problem", gem_problem),
        )
        if value
    ]

    return {
        "condition_poor_occupancy": poor_occupancy,
        "condition_low_effective_sample": low_effective_sample,
        "condition_multistart_instability": multistart_instability,
        "condition_gem_problem": gem_problem,
        "prohibits_economic_interpretation": bool(fired),
        "conditions_fired": fired or ["none"],
        "min_occupancy_share": float(occupancy_share.min()),
        "min_effective_sample": float(n_eff.min()),
        "n_starts_built": n_built,
        "n_starts_agreeing_with_best": n_agreeing,
        "multistart_agreeing_fraction": float(agreeing_fraction),
        "threshold_provenance": emp_config.REGIME_INTERPRETATION_THRESHOLDS["provenance"],
        "threshold_min_occupancy_share": emp_config.REGIME_MIN_OCCUPANCY_SHARE,
        "threshold_min_effective_sample": emp_config.REGIME_MIN_EFFECTIVE_SAMPLE,
        "threshold_multistart_loglik_tol": emp_config.REGIME_MULTISTART_LOGLIK_TOL,
        "threshold_multistart_min_agreeing_fraction": (
            emp_config.REGIME_MULTISTART_MIN_AGREEING_FRACTION
        ),
    }


def regime_description(fit: dict, periods: list[str]) -> dict:
    """Describe a full-sample regime-switching fit.

    Parameters
    ----------
    fit : dict
        Result of the regime-switching fit.
    periods : list of str
        Ordered months of the tensor the model was fitted on.

    Returns
    -------
    dict
        The regime structure, the transition matrix, the effective sample
        sizes and the smoothed regime probabilities.
    """
    n_regimes = int(fit["n_regimes"])
    sigma2 = [float(s) for s in fit["sigma2_list"]]
    occupancy = np.asarray(fit["occupancy"], dtype=float)
    n_eff = np.asarray(fit["n_eff"], dtype=float)

    # Regimes are ordered by increasing innovation variance.
    labels = ["low_variance", "high_variance"]
    conditions = interpretation_conditions(fit)
    poor = conditions["prohibits_economic_interpretation"]
    status = (
        "no_economic_interpretation: "
        + ", ".join(conditions["conditions_fired"])
        + " prohibits economic interpretation of the regime-specific operators"
        if poor
        else "interpretation_conditional_on_variance_ordering: the working "
        "likelihood orders regimes by variance, so any regime reading is "
        "conditional on a partition driven first by volatility; timing near an "
        "event does not establish a structural economic state"
    )

    structure_rows = []
    neff_rows = []
    for regime in range(n_regimes):
        A = np.asarray(fit["A_list"][regime], dtype=float)
        B = np.asarray(fit["B_list"][regime], dtype=float)
        A_bar, B_breve, gamma = canonical_triple(A, B)
        jsr = float(
            np.max(np.abs(np.linalg.eigvals(A)))
            * np.max(np.abs(np.linalg.eigvals(B)))
        )

        for name, matrix, row_labels, column_labels in (
            ("row_shape_A_bar", A_bar, emp_config.INDICATORS, emp_config.INDICATORS),
            ("column_shape_B_breve", B_breve, emp_config.COUNTRIES, emp_config.COUNTRIES),
        ):
            for i, row_label in enumerate(row_labels):
                for j, column_label in enumerate(column_labels):
                    structure_rows.append(
                        {
                            "regime": regime,
                            "regime_label": labels[regime],
                            "quantity": name,
                            "row": row_label,
                            "column": column_label,
                            "value": float(matrix[i, j]),
                            "regime_interpretation_status": status,
                        }
                    )

        for name, value in (
            ("amplitude_gamma", gamma),
            ("joint_spectral_radius", jsr),
            ("innovation_variance", sigma2[regime]),
            ("occupancy", float(occupancy[regime])),
            ("effective_sample_size", float(n_eff[regime])),
        ):
            structure_rows.append(
                {
                    "regime": regime,
                    "regime_label": labels[regime],
                    "quantity": name,
                    "row": "",
                    "column": "",
                    "value": float(value),
                    "regime_interpretation_status": status,
                }
            )

        neff_rows.append(
            {
                "model": "ms_full",
                "regime": regime,
                "regime_label": labels[regime],
                "occupancy": float(occupancy[regime]),
                "occupancy_share": float(occupancy[regime] / occupancy.sum()),
                "effective_sample_size": float(n_eff[regime]),
                "innovation_variance": sigma2[regime],
                "canonical_dimension": int(emp_config.K_CANONICAL),
                **conditions,
                "regime_interpretation_status": status,
            }
        )

    # The stored transition matrix is column-stochastic.
    column_stochastic = np.asarray(fit["transition_matrix"], dtype=float)
    row_stochastic = column_stochastic.T
    transition_rows = []
    for i in range(n_regimes):
        for j in range(n_regimes):
            transition_rows.append(
                {
                    "from_regime": i,
                    "from_regime_label": labels[i],
                    "to_regime": j,
                    "to_regime_label": labels[j],
                    "probability": float(row_stochastic[i, j]),
                    "expected_duration": (
                        float(1.0 / (1.0 - row_stochastic[i, i]))
                        if row_stochastic[i, i] < 1.0
                        else float("inf")
                    ),
                }
            )

    smoothed = np.asarray(fit["smoothed_probs"], dtype=float)
    score_periods = periods[1 : smoothed.shape[0] + 1]
    smoothed_rows = [
        {
            "period": score_periods[t],
            "regime": regime,
            "regime_label": labels[regime],
            "smoothed_probability": float(smoothed[t, regime]),
            "smoothed_never_used_in_forecasts": (
                "retrospective description only; smoothed probabilities never "
                "enter forecasts or formal conditional forecast tests"
            ),
        }
        for t in range(smoothed.shape[0])
        for regime in range(n_regimes)
    ]

    return {
        "regime_structure": pd.DataFrame(structure_rows),
        "transition_matrix": pd.DataFrame(transition_rows),
        "regime_neff": pd.DataFrame(neff_rows),
        "regime_probabilities_smoothed": pd.DataFrame(smoothed_rows),
        "regime_interpretation_status": status,
        "interpretation_conditions": conditions,
    }


def predicted_probabilities(records: pd.DataFrame) -> pd.DataFrame:
    """Collect the one-step-ahead predicted regime probabilities.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records of the regime-switching model.

    Returns
    -------
    pandas.DataFrame
        One row per period and regime.
    """
    ordered = records[records["status"] == "ok"].sort_values("period")
    rows = []
    for _, row in ordered.iterrows():
        high = float(row["p_high_predicted"])
        for regime, label, value in (
            (0, "low_variance", 1.0 - high),
            (1, "high_variance", high),
        ):
            rows.append(
                {
                    "period": str(row["period"]),
                    "regime": regime,
                    "regime_label": label,
                    "predicted_probability": value,
                }
            )
    return pd.DataFrame(rows)


# =========================================================
# Solver diagnostics
# =========================================================
def initialisation_table(fits: dict) -> pd.DataFrame:
    """Report the final likelihood and retained start of every EM start.

    Parameters
    ----------
    fits : dict
        Mapping from (model, fit_label) to a fit result.

    Returns
    -------
    pandas.DataFrame
        One row per fit and start.
    """
    rows = []
    for (model, fit_label), fit in fits.items():
        summaries = fit["all_init_summaries"]
        retained = int(fit["init_id"])
        for _, summary in summaries.iterrows():
            init_id = int(summary["init_id"])
            rows.append(
                {
                    "model": model,
                    "fit_label": fit_label,
                    "init_id": init_id,
                    "start_construction": (
                        "deterministic_projection_residual_split"
                        if init_id == 0
                        else "tangent_perturbation_with_bernoulli_split"
                    ),
                    "final_loglik": float(summary["loglik"]),
                    "converged": bool(summary["converged"]),
                    "n_iter": int(summary["n_iter"]),
                    "retained": init_id == retained,
                    "convergence_reason": str(fit["convergence_reason"]),
                    "n_starts_built": int(len(summaries)),
                    "n_starts_contracted": int(emp_config.N_STARTS),
                }
            )
    return pd.DataFrame(rows)


def telemetry_table(fits: dict, records: pd.DataFrame) -> pd.DataFrame:
    """Report the solver and EM diagnostics of every retained fit.

    Parameters
    ----------
    fits : dict
        Mapping from (model, fit_label) to a fit result.
    records : pandas.DataFrame
        Forecast unit records, for the rolling and static solver diagnostics.

    Returns
    -------
    pandas.DataFrame
        One row per fit, plus one aggregate row per model.
    """
    rows = []
    for (model, fit_label), fit in fits.items():
        monotone = not bool(fit["rolled_back"])
        radii = [
            float(
                np.max(np.abs(np.linalg.eigvals(np.asarray(A, dtype=float))))
                * np.max(np.abs(np.linalg.eigvals(np.asarray(B, dtype=float))))
            )
            for A, B in zip(fit["A_list"], fit["B_list"])
        ]
        radius = max(radii)
        contracted = any(forecasting.inferred_contraction(v) for v in radii)
        rows.append(
            {
                "model": model,
                "fit_label": fit_label,
                "n_iter": int(fit["n_iter"]),
                "converged": bool(fit["converged"]),
                "convergence_reason": str(fit["convergence_reason"]),
                "gem_rejected_steps": int(fit["gem_rejected_steps"]),
                "rolled_back": bool(fit["rolled_back"]),
                "likelihood_monotone": monotone,
                "gem_rollback_tolerance": emp_config.GEM_ROLLBACK_TOLERANCE,
                # Stability contraction is applied inside the M-step and is
                # not itself returned, so it is inferred from the radius:
                # a contraction lands the radius on the same fixed value.
                "stability_contracted": bool(contracted),
                "stability_contracted_provenance": (
                    forecasting.CONTRACTION_PROVENANCE
                ),
                "joint_spectral_radius": radius,
                "final_loglik": float(fit["loglik"]),
                "n_origins": 1,
                "n_origins_contracted": int(bool(contracted)),
                "n_origins_soft_non_convergent": int(
                    fit["convergence_reason"] == "max_iter"
                ),
                "convergence_assessed": True,
                "numerical_reliability_status": (
                    "reliable"
                    if monotone and fit["converged"]
                    else "soft_non_convergence_or_rollback: forecast retained, "
                    "structural interpretation prohibited"
                ),
            }
        )

    successful = records[records["status"] == "ok"]
    for model, block in successful.groupby("model", sort=False):
        assessed = (
            bool(block["convergence_assessed"].all())
            if "convergence_assessed" in block
            else True
        )
        rows.append(
            {
                "model": model,
                "fit_label": f"{model}:test_origins_aggregate",
                "n_iter": int(block["n_iter"].max()) if "n_iter" in block else 0,
                "converged": (None if not assessed else bool(block["converged"].all())),
                "convergence_assessed": assessed,
                "convergence_reason": (
                    "aggregate_over_test_origins"
                    if assessed
                    else forecasting.NOT_ASSESSABLE_REASON
                ),
                "gem_rejected_steps": (
                    int(block["gem_rejected_steps"].sum())
                    if "gem_rejected_steps" in block
                    else 0
                ),
                "rolled_back": (
                    bool(block["rolled_back"].any())
                    if "rolled_back" in block
                    else False
                ),
                "likelihood_monotone": (
                    bool(block["likelihood_monotone"].all())
                    if "likelihood_monotone" in block
                    else True
                ),
                "gem_rollback_tolerance": emp_config.GEM_ROLLBACK_TOLERANCE,
                "stability_contracted": (
                    bool(block["stability_contracted"].any())
                    if "stability_contracted" in block
                    else NOT_EXPOSED
                ),
                "stability_contracted_provenance": (
                    forecasting.CONTRACTION_PROVENANCE
                    if "stability_contracted" in block
                    else NOT_EXPOSED
                ),
                "joint_spectral_radius": (
                    float(block["joint_spectral_radius"].max())
                    if "joint_spectral_radius" in block
                    else float("nan")
                ),
                "final_loglik": float("nan"),
                # Counts how often the stability rule fired and a fit
                # reached the iteration cap, not only whether it ever did.
                "n_origins": int(len(block)),
                "n_origins_contracted": (
                    int(block["stability_contracted"].sum())
                    if "stability_contracted" in block
                    else 0
                ),
                "n_origins_soft_non_convergent": (
                    int(block["soft_non_convergence"].fillna(False).sum())
                    if "soft_non_convergence" in block
                    else 0
                ),
                "numerical_reliability_status": (
                    forecasting.NOT_ASSESSABLE_STATUS
                    if not assessed
                    else "reliable"
                    if bool(block["converged"].all())
                    else "soft_non_convergence: forecasts retained, structural "
                    "interpretation prohibited"
                ),
            }
        )
    return pd.DataFrame(rows)


# =========================================================
# Reduced rank
# =========================================================
def rank_energy_table(
    tuning: pd.DataFrame, static_pair, regime_fits: dict
) -> pd.DataFrame:
    """Relate the selected ranks to singular-value energy and sample size.

    Parameters
    ----------
    tuning : pandas.DataFrame
        Complete validation tuning table.
    static_pair : tuple of numpy.ndarray
        Full-sample static fit, whose spectra define the energy shares.
    regime_fits : dict
        Full-sample regime-switching fits, for the effective sample sizes.

    Returns
    -------
    pandas.DataFrame
        One row per reduced-rank candidate.
    """
    A, B = static_pair
    energy_A = np.linalg.svd(np.asarray(A), compute_uv=False) ** 2
    energy_B = np.linalg.svd(np.asarray(B), compute_uv=False) ** 2
    cumulative_A = np.cumsum(energy_A) / energy_A.sum()
    cumulative_B = np.cumsum(energy_B) / energy_B.sum()

    m, n = emp_config.M_DIM, emp_config.N_DIM
    reduced_models = ("static_reduced", "rolling_reduced", "ms_reduced")

    neff_by_rank = {}
    for (model, _), fit in regime_fits.items():
        if model == "ms_reduced":
            neff_by_rank[model] = float(np.min(np.asarray(fit["n_eff"], dtype=float)))

    rows = []
    for _, entry in tuning[tuning["model"].isin(reduced_models)].iterrows():
        rank = int(entry["rank"])
        rows.append(
            {
                "model": str(entry["model"]),
                "model_label": emp_config.MODEL_LABELS[str(entry["model"])],
                "candidate": str(entry["candidate"]),
                "rank": rank,
                "window": (
                    None if pd.isna(entry["window"]) else int(entry["window"])
                ),
                "canonical_parameter_count": int(
                    rank * (2 * m - rank) + rank * (2 * n - rank) - 1
                ),
                "full_rank_parameter_count": int(emp_config.K_CANONICAL),
                "singular_value_energy_A": float(cumulative_A[rank - 1]),
                "singular_value_energy_B": float(cumulative_B[rank - 1]),
                "validation_msfe": float(entry["validation_msfe"]),
                "feasible": bool(entry["feasible"]),
                "selected": bool(entry["selected"]),
                "regime_effective_sample_size": neff_by_rank.get(
                    str(entry["model"]), float("nan")
                ),
                "regularisation_not_true_rank": (
                    "reduced rank is forecast regularisation, not a third "
                    "time-variation mechanism; a gain at low effective sample "
                    "size may be described as consistent with variance "
                    "reduction, never as proof that the true rank is low"
                ),
            }
        )
    return pd.DataFrame(rows)

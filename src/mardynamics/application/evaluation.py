"""Losses and the two forecast comparisons.

The common-origin rule is enforced first: if any of the six models
hard-fails at an origin, that origin leaves every comparative loss and
forecast-test calculation. A failed forecast is not replaced, and the
planned, attempted, successful and failed counts and the dropped-origin
list are reported alongside the results.

Diebold-Mariano uses a two-sided equal-predictive-accuracy null, Newey-West
HAC with lag three and the Harvey-Leybourne-Newbold correction for horizon
one. Giacomini-White conditions on the continuous ex-ante predicted
high-variance probability of the full-rank MS-MAR and tests the intercept
and state coefficient jointly. Both families are Holm-adjusted over the
five challengers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import mardynamics.application.emp_config as emp_config
import mardynamics.core.structural_tests as structural_tests


def common_origins(records: pd.DataFrame, targets: list[str]) -> tuple[list, list]:
    """Split the planned origins into the common set and the dropped set.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records of all six models.
    targets : list of str
        The planned target months.

    Returns
    -------
    tuple
        The ordered common origins and the ordered dropped origins.
    """
    common, dropped = [], []
    for period in targets:
        slice_ = records[records["period"] == period]
        complete = len(slice_) == len(emp_config.MODELS)
        successful = bool((slice_["status"] == "ok").all()) and complete
        (common if successful else dropped).append(period)
    return common, dropped


def accuracy_tables(
    records: pd.DataFrame, targets: list[str], common: list[str], dropped: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ``forecast_accuracy.csv`` and ``relative_msfe.csv``.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records.
    targets : list of str
        The planned target months of this run mode.
    common : list of str
        The common origins.
    dropped : list of str
        The dropped origins.

    Returns
    -------
    tuple of pandas.DataFrame
        The accuracy table and the relative-MSFE table.
    """
    status = (
        "complete_common_sample"
        if not dropped
        else f"reduced_common_sample: {len(dropped)} origin(s) dropped for all "
        f"six models because at least one model hard-failed there"
    )

    rows = []
    for model in emp_config.MODELS:
        model_records = records[records["model"] == model]
        on_common = model_records[model_records["period"].isin(common)]
        losses = on_common["loss"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "model_label": emp_config.MODEL_LABELS[model],
                "n_planned": len(targets),
                "n_attempted": int(len(model_records)),
                "n_successful": int((model_records["status"] == "ok").sum()),
                "n_failed": int((model_records["status"] != "ok").sum()),
                "n_common_origins": len(common),
                "n_dropped_origins": len(dropped),
                "dropped_origins": ";".join(dropped),
                "mean_loss": float(np.mean(losses)) if losses.size else float("nan"),
                "median_loss": (
                    float(np.median(losses)) if losses.size else float("nan")
                ),
                "p90_loss": (
                    float(np.quantile(losses, 0.90)) if losses.size else float("nan")
                ),
                "common_origin_status": status,
            }
        )

    accuracy = pd.DataFrame(rows)
    benchmark_mean = float(
        accuracy.loc[accuracy["model"] == emp_config.BENCHMARK_MODEL, "mean_loss"].iloc[0]
    )
    relative = pd.DataFrame(
        {
            "model": accuracy["model"],
            "model_label": accuracy["model_label"],
            "mean_loss": accuracy["mean_loss"],
            "benchmark_mean_loss": benchmark_mean,
            "relative_msfe": accuracy["mean_loss"] / benchmark_mean,
            "n_common_origins": accuracy["n_common_origins"],
        }
    )
    return accuracy, relative


def period_loss(records: pd.DataFrame, common: list[str]) -> pd.DataFrame:
    """Build the monthly loss path on the logarithmic scale.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records.
    common : list of str
        The common origins.

    Returns
    -------
    pandas.DataFrame
        One row per month and model.
    """
    on_common = records[
        (records["period"].isin(common)) & (records["status"] == "ok")
    ]
    frame = on_common[["period", "model", "loss"]].copy()
    frame["model_label"] = frame["model"].map(emp_config.MODEL_LABELS)
    frame["log_loss"] = np.log(frame["loss"].to_numpy(dtype=float))
    return frame.sort_values(["model", "period"]).reset_index(drop=True)


def cellwise_losses(records: pd.DataFrame, common: list[str]) -> tuple:
    """Aggregate the squared error by indicator and by country.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records.
    common : list of str
        The common origins.

    Returns
    -------
    tuple of pandas.DataFrame
        The indicator table and the country table.
    """
    on_common = records[
        (records["period"].isin(common)) & (records["status"] == "ok")
    ]

    indicator_rows, country_rows = [], []
    for model in emp_config.MODELS:
        block = on_common[on_common["model"] == model]
        for indicator in emp_config.INDICATORS:
            columns = [f"se_{indicator}_{country}" for country in emp_config.COUNTRIES]
            indicator_rows.append(
                {
                    "model": model,
                    "model_label": emp_config.MODEL_LABELS[model],
                    "indicator": indicator,
                    "indicator_label": emp_config.INDICATOR_LABELS[indicator],
                    "mean_squared_error": float(
                        block[columns].to_numpy(dtype=float).mean()
                    ),
                    "n_common_origins": len(common),
                }
            )
        for country in emp_config.COUNTRIES:
            columns = [f"se_{indicator}_{country}" for indicator in emp_config.INDICATORS]
            country_rows.append(
                {
                    "model": model,
                    "model_label": emp_config.MODEL_LABELS[model],
                    "country": country,
                    "country_label": emp_config.COUNTRY_LABELS[country],
                    "mean_squared_error": float(
                        block[columns].to_numpy(dtype=float).mean()
                    ),
                    "n_common_origins": len(common),
                }
            )

    return pd.DataFrame(indicator_rows), pd.DataFrame(country_rows)


def predicted_regime_split(
    records: pd.DataFrame, common: list[str], p_high: pd.Series
) -> pd.DataFrame:
    """Split the losses by the ex-ante predicted high-variance probability.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records.
    common : list of str
        The common origins.
    p_high : pandas.Series
        Ex-ante predicted high-variance probability, indexed by month, taken
        from the full-rank MS-MAR.

    Returns
    -------
    pandas.DataFrame
        One row per model and probability group.
    """
    on_common = records[
        (records["period"].isin(common)) & (records["status"] == "ok")
    ].copy()
    on_common["p_high"] = on_common["period"].map(p_high)
    on_common["predicted_regime_group"] = np.where(
        on_common["p_high"].to_numpy(dtype=float) > 0.5, "p_high>0.5", "p_high<=0.5"
    )

    rows = []
    for model in emp_config.MODELS:
        block = on_common[on_common["model"] == model]
        for group in ("p_high<=0.5", "p_high>0.5"):
            losses = block.loc[
                block["predicted_regime_group"] == group, "loss"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model,
                    "model_label": emp_config.MODEL_LABELS[model],
                    "predicted_regime_group": group,
                    "n_origins": int(losses.size),
                    "mean_loss": (
                        float(np.mean(losses)) if losses.size else float("nan")
                    ),
                    "median_loss": (
                        float(np.median(losses)) if losses.size else float("nan")
                    ),
                    "descriptive_not_a_test": (
                        "descriptive split by the ex-ante predicted high-variance "
                        "probability of the full-rank MS-MAR; not a formal test"
                    ),
                }
            )
    return pd.DataFrame(rows)


# =========================================================
# HAC machinery shared by the two comparisons
# =========================================================
def _newey_west(residuals: np.ndarray, regressors: np.ndarray, lag: int) -> np.ndarray:
    """Return the Newey-West HAC covariance of an OLS coefficient vector.

    Parameters
    ----------
    residuals : numpy.ndarray
        OLS residuals of length ``n``.
    regressors : numpy.ndarray
        Design matrix of shape ``(n, p)``.
    lag : int
        Maximum Bartlett lag.

    Returns
    -------
    numpy.ndarray
        The ``(p, p)`` sandwich covariance.
    """
    n = regressors.shape[0]
    scores = regressors * residuals[:, None]
    meat = scores.T @ scores
    for shift in range(1, lag + 1):
        weight = 1.0 - shift / (lag + 1.0)
        cross = scores[shift:].T @ scores[:-shift]
        meat += weight * (cross + cross.T)
    bread = np.linalg.inv(regressors.T @ regressors)
    return bread @ meat @ bread


def _loss_differentials(
    records: pd.DataFrame, common: list[str]
) -> tuple[dict, list[str]]:
    """Build the per-challenger loss differential against the benchmark.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records.
    common : list of str
        The common origins.

    Returns
    -------
    tuple
        Mapping from challenger to its differential series, and the ordered
        common origins actually used.
    """
    on_common = records[
        (records["period"].isin(common)) & (records["status"] == "ok")
    ]
    wide = on_common.pivot_table(index="period", columns="model", values="loss")
    wide = wide.loc[[p for p in common if p in wide.index]]
    benchmark = wide[emp_config.BENCHMARK_MODEL].to_numpy(dtype=float)
    return (
        {
            model: wide[model].to_numpy(dtype=float) - benchmark
            for model in emp_config.CHALLENGERS
        },
        list(wide.index),
    )


def diebold_mariano(records: pd.DataFrame, common: list[str]) -> pd.DataFrame:
    """Run the Diebold-Mariano comparison.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records.
    common : list of str
        The common origins.

    Returns
    -------
    pandas.DataFrame
        One row per challenger.
    """
    differentials, used = _loss_differentials(records, common)
    rows, pvalues = [], []

    for model in emp_config.CHALLENGERS:
        d = differentials[model]
        n = d.size
        design = np.ones((n, 1))
        mean = float(d.mean())
        residuals = d - mean
        covariance = _newey_west(residuals, design, emp_config.HAC_LAG)
        standard_error = float(np.sqrt(covariance[0, 0]))
        statistic = mean / standard_error if standard_error > 0 else np.nan

        # Harvey-Leybourne-Newbold correction at horizon one.
        horizon = 1
        factor = np.sqrt(
            (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        )
        corrected = statistic * factor
        pvalue = float(2.0 * stats.t.sf(abs(corrected), df=n - 1))
        pvalues.append(pvalue)

        rows.append(
            {
                "model": model,
                "model_label": emp_config.MODEL_LABELS[model],
                "mean_loss_differential": mean,
                "favours": "challenger" if mean < 0 else "benchmark",
                "hac_lag": emp_config.HAC_LAG,
                "dm_statistic": float(statistic),
                "hln_statistic": float(corrected),
                "hln_correction_factor": float(factor),
                "p_value": pvalue,
                "n_common_origins": len(used),
                "dm_interpretation": (
                    "two-sided equal-predictive-accuracy null; a negative mean "
                    "differential favours the challenger descriptively; "
                    "non-rejection is not equivalence and a lower loss without "
                    "rejection is not certified dominance"
                ),
            }
        )

    holm = structural_tests.holm_adjusted_pvalues(pvalues)
    for row, adjusted in zip(rows, holm):
        row["holm_p_value"] = float(adjusted)
        row["holm_decision_5pct"] = (
            "reject" if adjusted < emp_config.PRIMARY_LEVEL else "do not reject"
        )
    return pd.DataFrame(rows)


def giacomini_white(
    records: pd.DataFrame, common: list[str], p_high: pd.Series
) -> pd.DataFrame:
    """Run the conditional comparison.

    Parameters
    ----------
    records : pandas.DataFrame
        Test-phase unit records.
    common : list of str
        The common origins.
    p_high : pandas.Series
        The continuous ex-ante predicted high-variance probability of the
        full-rank MS-MAR, indexed by month.

    Returns
    -------
    pandas.DataFrame
        One row per challenger.
    """
    differentials, used = _loss_differentials(records, common)
    state = np.asarray([float(p_high[period]) for period in used])
    design = np.column_stack([np.ones(state.size), state])

    rows, joint_pvalues = [], []
    for model in emp_config.CHALLENGERS:
        d = differentials[model]
        coefficients, *_ = np.linalg.lstsq(design, d, rcond=None)
        residuals = d - design @ coefficients
        covariance = _newey_west(residuals, design, emp_config.HAC_LAG)

        wald = float(coefficients @ np.linalg.solve(covariance, coefficients))
        joint_p = float(stats.chi2.sf(wald, df=design.shape[1]))
        joint_pvalues.append(joint_p)

        state_se = float(np.sqrt(covariance[1, 1]))
        state_z = float(coefficients[1] / state_se) if state_se > 0 else np.nan
        rows.append(
            {
                "model": model,
                "model_label": emp_config.MODEL_LABELS[model],
                "intercept": float(coefficients[0]),
                "state_coefficient": float(coefficients[1]),
                "state_standard_error": state_se,
                "state_p_value": float(2.0 * stats.norm.sf(abs(state_z))),
                "joint_wald_statistic": wald,
                "joint_p_value": joint_p,
                "hac_lag": emp_config.HAC_LAG,
                "n_common_origins": len(used),
                "generated_state_caveats": (
                    "the conditioning variable is the ex-ante one-step predicted "
                    "high-variance probability of the full-rank MS-MAR and is "
                    "the same series in all five comparisons; in the full-rank "
                    "MS-MAR comparison the instrument is generated by the "
                    "challenger itself; the HAC calculation treats the estimated "
                    "probability as observed and does not propagate "
                    "generated-state uncertainty"
                ),
            }
        )

    holm = structural_tests.holm_adjusted_pvalues(joint_pvalues)
    for row, adjusted in zip(rows, holm):
        row["holm_joint_p_value"] = float(adjusted)
        row["holm_decision_5pct"] = (
            "reject" if adjusted < emp_config.PRIMARY_LEVEL else "do not reject"
        )
    return pd.DataFrame(rows)


# =========================================================
# The raw forecast artifact
# =========================================================
def raw_forecast_records(
    records: pd.DataFrame, common: list[str] | None = None
) -> pd.DataFrame:
    """Flatten wide per-unit records into the long-format raw forecast artifact.

    One record per model, forecast origin, target date, indicator and
    country, carrying the fitting and evaluation preprocessor identifiers,
    the prediction on both scales, the evaluation-scale actual, the
    evaluation-scale error and its squared contribution, common-origin
    eligibility and fit status. Every aggregated loss, DM statistic and GW
    statistic in this module is reproducible from this artifact alone.

    Parameters
    ----------
    records : pandas.DataFrame
        Wide per-unit records from
        :func:`mardynamics.application.forecasting.run_units`.
    common : list of str or None, optional
        The common test origins; omitted (``None``) for the validation phase,
        which has no common-origin rule.

    Returns
    -------
    pandas.DataFrame
        The long-format artifact.
    """
    common_set = set(common) if common is not None else None
    rows: list[dict] = []

    for record in records.to_dict("records"):
        ok = record["status"] == "ok"
        base = {
            "model": record["model"],
            "candidate": record["candidate"],
            "rank": record["rank"],
            "window": record["window"],
            "origin": record["period"],
            "target": record["period"],
            "fitting_preprocessor": record.get("fitting_preprocessor"),
            "evaluation_preprocessor": record.get("evaluation_preprocessor"),
            "common_origin_eligible": (
                pd.NA if common_set is None else record["period"] in common_set
            ),
            "fit_status": record["status"],
            "failure_reason": record["failure_reason"],
            "p_high_predicted": record.get("p_high_predicted", np.nan),
        }
        for indicator in emp_config.INDICATORS:
            for country in emp_config.COUNTRIES:
                rows.append(
                    {
                        **base,
                        "indicator": indicator,
                        "country": country,
                        "prediction_fit_scale": (
                            record[f"pred_fit_{indicator}_{country}"]
                            if ok
                            else np.nan
                        ),
                        "prediction_eval_scale": (
                            record[f"pred_eval_{indicator}_{country}"]
                            if ok
                            else np.nan
                        ),
                        "actual_eval_scale": (
                            record[f"actual_eval_{indicator}_{country}"]
                            if ok
                            else np.nan
                        ),
                        "evaluation_error": (
                            record[f"actual_eval_{indicator}_{country}"]
                            - record[f"pred_eval_{indicator}_{country}"]
                            if ok
                            else np.nan
                        ),
                        "squared_error": (
                            record[f"se_{indicator}_{country}"] if ok else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


# =========================================================
# Validation tuning
# =========================================================
def select_tuning(validation_records: pd.DataFrame, targets: list[str]) -> tuple:
    """Score every candidate on validation MSFE and select the best per model.

    A candidate with a hard-failed required origin is infeasible. Ties
    within ``1e-12`` prefer the longer window and then the lower rank.

    Parameters
    ----------
    validation_records : pandas.DataFrame
        Validation-phase unit records.
    targets : list of str
        The validation target months of this run mode.

    Returns
    -------
    tuple
        The complete tuning table and the selection mapping.
    """
    rows = []
    for candidate, block in validation_records.groupby("candidate", sort=False):
        successful = block[block["status"] == "ok"]
        n_failed = int((block["status"] != "ok").sum())
        feasible = n_failed == 0 and len(successful) == len(targets)
        rows.append(
            {
                "model": block["model"].iloc[0],
                "model_label": emp_config.MODEL_LABELS[block["model"].iloc[0]],
                "candidate": candidate,
                "rank": block["rank"].iloc[0],
                "window": block["window"].iloc[0],
                "n_origins": int(len(block)),
                "n_successful": int(len(successful)),
                "n_failed": n_failed,
                "feasible": feasible,
                "validation_msfe": (
                    float(successful["loss"].mean()) if feasible else float("nan")
                ),
            }
        )

    table = pd.DataFrame(rows)
    table["selected"] = False
    table["selection_rule"] = (
        "lowest recursive one-step validation MSFE; ties within 1e-12 prefer "
        "the longer window and then the lower rank"
    )

    selections = {}
    for model in emp_config.MODELS:
        block = table[(table["model"] == model) & table["feasible"]]
        if block.empty:
            continue
        best = float(block["validation_msfe"].min())
        tied = block[block["validation_msfe"] <= best + emp_config.TIE_TOLERANCE]
        # Longer window first, then lower rank.
        tied = tied.sort_values(
            by=["window", "rank"],
            ascending=[False, True],
            na_position="last",
        )
        winner = tied.iloc[0]
        table.loc[table["candidate"] == winner["candidate"], "selected"] = True
        selections[model] = {
            "candidate": str(winner["candidate"]),
            "rank": (None if pd.isna(winner["rank"]) else int(winner["rank"])),
            "window": (None if pd.isna(winner["window"]) else int(winner["window"])),
            "validation_msfe": float(winner["validation_msfe"]),
            "n_tied": int(len(tied)),
        }

    return table, selections

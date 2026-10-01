"""Finite-sample calibration and numerical reliability.

Thirteen null cells, one paired wild-bootstrap comparison and one
multiple-start check. Every null cell simulates under a constant
canonical parameter, fits the static MAR(1) once and reads the global
statistic, its three influence blocks, the canonical recovery losses and
the numerical-reliability record from that single fit, so nothing in a
replication is computed on different data.

Because the null cells are the stable-DGP replications, the canonical
recovery output is derived from them. No separate recovery experiment
exists: one would resimulate a design the study already runs.

Two further quantities are stored here that a later branch consumes
instead of this one: the reference-dimension cell stores the
unrestricted statistic that the matrix-restriction branch compares
against, and the GARCH cell stores the realised strength of its
conditional-variance factor. Storing them here lets those comparisons
reuse the original null records instead of simulating a second null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mardynamics.experiments.design as design
import mardynamics.experiments.dgp as dgp
import mardynamics.experiments.estimators as estimators
import mardynamics.experiments.innovations as innovations
import mardynamics.experiments.metrics as metrics
import mardynamics.experiments.sim_config as simcfg
import mardynamics.core.structural_tests as structural_tests
from mardynamics.core.mar_canonical import canonical_triple
from mardynamics.experiments.designs import common


def build_context(cell) -> dict:
    """Build the per-cell objects of a calibration cell.

    Parameters
    ----------
    cell : Cell
        The cell.

    Returns
    -------
    dict
        The population and the innovation generator.
    """
    return {
        "truth": design.build_truth(
            cell.m, cell.n, cell.persistence, cell.configuration
        ),
        "innovation": innovations.build_innovation(cell.innovation, simcfg.SIGMA),
    }


def simulate(cell, context: dict, replication_id: int) -> np.ndarray:
    """Simulate one replication's dataset.

    Parameters
    ----------
    cell : Cell
        The cell.
    context : dict
        The cell context.
    replication_id : int
        Replication index.

    Returns
    -------
    numpy.ndarray
        The simulated process, shape ``(T, m, n)``.
    """
    return dgp.simulate_static_tensor(
        context["truth"],
        cell.T,
        context["innovation"],
        common.dataset_rng(cell, replication_id),
    )


def execute(cell, context: dict, replication_id: int) -> list[dict]:
    """Run one null calibration replication.

    Parameters
    ----------
    cell : Cell
        The cell.
    context : dict
        The cell context.
    replication_id : int
        Replication index.

    Returns
    -------
    list of dict
        One record.
    """
    tensor = simulate(cell, context, replication_id)
    fit = common.fit_and_test(tensor)

    row = common.battery_row(fit["battery"])
    row.update(fit["telemetry"])
    row.update(fit["canonical"])
    row.update(common.recovery_row(fit, context["truth"], cell.T))

    if "nyblom_var" in cell.statistics:
        K = structural_tests.var_dimension(cell.m, cell.n)
        statistic = structural_tests.var_nyblom_statistic(tensor)
        row["nyblom_var"] = float(statistic)
        row["K"] = int(K)
        row["p_var"] = common.pvalue(K, statistic)
        row.update(common.rejection_flags("var", row["p_var"]))

    if "garch_strength" in cell.statistics:
        row.update(
            innovations.garch_strength(
                context["innovation"].draw(
                    simcfg.BURN_IN + cell.T,
                    cell.m,
                    cell.n,
                    common.dataset_rng(cell, replication_id),
                ),
                simcfg.SIGMA,
                cell.m,
                cell.n,
                simcfg.GARCH_ALPHA,
                simcfg.GARCH_BETA,
            )
        )

    return [row]


def execute_bootstrap(cell, context: dict, replication_id: int) -> list[dict]:
    """Run one fixed-regressor wild-bootstrap replication.

    The dataset is the reference null cell's dataset at the same replication
    index, because the two cells share a dataset key, so the asymptotic and
    bootstrap p-values compared here belong to the same realised statistic on
    the same data. The bootstrap draws come from a separate sub-stream, so they
    cannot disturb the dataset.

    This is a finite-sample robustness comparison of the Nyblom-Hansen
    statistic. It does not redefine the primary test.

    Parameters
    ----------
    cell : Cell
        The bootstrap cell.
    context : dict
        The cell context.
    replication_id : int
        Replication index.

    Returns
    -------
    list of dict
        One record.
    """
    tensor = simulate(cell, context, replication_id)
    fit = common.fit_and_test(tensor)
    battery = fit["battery"]

    p_asymptotic = common.pvalue(battery.k, battery.nyblom)
    result = structural_tests.wild_bootstrap_pvalue(
        tensor,
        battery.nyblom,
        (fit["A"], fit["B"]),
        common.dataset_rng(cell, replication_id, role="wild_bootstrap"),
        n_boot=cell.n_resample,
        enforce_stability=simcfg.ENFORCE_STABILITY,
        max_radius=simcfg.MAX_RADIUS,
    )

    return [
        {
            "nyblom_global": float(battery.nyblom),
            "k": int(battery.k),
            "p_asymptotic": p_asymptotic,
            "p_bootstrap": float(result["p_bootstrap"]),
            "n_boot": int(result["n_boot"]),
            **common.rejection_flags("asymptotic", p_asymptotic),
            **common.rejection_flags("bootstrap", float(result["p_bootstrap"])),
            **fit["telemetry"],
        }
    ]


def execute_multiple_start(cell, context: dict, replication_id: int) -> list[dict]:
    """Refit one reference null dataset from the ordinary and perturbed starts.

    Start zero is the ordinary fit: the shared alternating recursion from the
    projection estimator. Starts one to four route the same recursion from
    deterministic perturbations drawn from replication-specific seeds, each of
    which satisfies the canonical norm, sign and stability restrictions by
    construction.

    This is a numerical diagnostic. It is not a separate part of the study and
    it is not a proof of global convergence.

    Parameters
    ----------
    cell : Cell
        The multiple-start cell.
    context : dict
        The cell context.
    replication_id : int
        Replication index.

    Returns
    -------
    list of dict
        One record per start.
    """
    tensor = simulate(cell, context, replication_id)
    reference: dict | None = None
    rows = []

    for start_id in range(1 + int(cell.extra["n_perturbed_starts"])):
        if start_id == 0:
            start = None
            A, B, telemetry = estimators.static_fit(tensor, return_telemetry=True)
        else:
            start = estimators.perturbed_start(
                tensor,
                common.dataset_rng(cell, replication_id, role=f"start_{start_id}"),
                cell.extra["perturbation"],
            )
            A, B, telemetry = estimators.static_fit(
                tensor,
                return_telemetry=True,
                A_init=start["A"],
                B_init=start["B"],
            )

        A_bar, B_breve, gamma = canonical_triple(A, B)
        if reference is None:
            reference = {"A_bar": A_bar, "B_breve": B_breve, "kappa": np.log(gamma)}

        A_aligned, B_aligned = design.align_joint_sign(
            A_bar, B_breve, reference["A_bar"]
        )
        d_A = float(np.linalg.norm(A_aligned - reference["A_bar"], ord="fro"))
        d_B = float(np.linalg.norm(B_aligned - reference["B_breve"], ord="fro"))
        d_kappa = float(np.log(gamma) - reference["kappa"])

        # The objective is the shared estimator's own least-squares
        # criterion: "equal within the existing objective tolerance" is
        # read in the units the recursion actually stops on.
        final = float(telemetry["criterion_after"])
        rows.append(
            {
                "start_id": start_id,
                "is_ordinary_start": start_id == 0,
                "start_rho": np.nan if start is None else start["rho"],
                "start_geodesic_displacement": (
                    0.0 if start is None else start["geodesic_displacement"]
                ),
                "converged": bool(telemetry.get("converged", False)),
                "n_iterations": int(telemetry.get("als_iterations", -1)),
                "initial_objective": float(telemetry["criterion_before"]),
                "final_objective": float(final),
                "aligned_d_row_shape": d_A,
                "aligned_d_column_shape": d_B,
                "aligned_d_log_magnitude": d_kappa,
                "aligned_canonical_distance": float(
                    np.sqrt(d_A**2 + d_B**2 + d_kappa**2)
                ),
                # The solution each start reached, so any comparison
                # across starts is a later calculation.
                **common.canonical_row(A, B),
            }
        )

    # Every gap is measured against the ordinary start, which is the fit the
    # rest of the study uses; a perturbed start is only ever a comparison.
    baseline = rows[0]["final_objective"]
    for row in rows:
        row["objective_gap"] = float(row["final_objective"] - baseline)
        row["relative_objective_gap"] = float(
            abs(row["objective_gap"]) / (1.0 + abs(baseline))
        )
    return rows


# =========================================================
# Tables
# =========================================================
def _successful(records: pd.DataFrame) -> pd.DataFrame:
    """Return the successful replications of a stored record frame.

    Parameters
    ----------
    records : pandas.DataFrame
        The stored records.

    Returns
    -------
    pandas.DataFrame
        The rows whose replication completed.
    """
    return records[records["status"] == "ok"] if "status" in records else records


def null_calibration_table(cell, records: pd.DataFrame) -> pd.DataFrame:
    """Rejection table of one null cell: one row per statistic and level.

    Reports the global rate, the three raw block rates, the Holm
    familywise rate over the three blocks, Monte Carlo standard errors
    and the completed, failed and excluded replication counts.

    Parameters
    ----------
    cell : Cell
        The cell.
    records : pandas.DataFrame
        Its stored replication records.

    Returns
    -------
    pandas.DataFrame
        The rejection rates with their uncertainty and tolerance flags.
    """
    successful = _successful(records)
    attempted = int(len(records[records["record_index"] == 0])) if not records.empty else 0
    rows = []

    statistics = [
        ("global", "global"),
        ("raw_row_shape", "raw_A"),
        ("raw_column_shape", "raw_B"),
        ("raw_magnitude", "raw_gamma"),
        ("holm_familywise_blocks", "holm_any"),
    ]
    if "p_var" in records:
        statistics.append(("unrestricted_var", "var"))

    for label, column_stem in statistics:
        for level in simcfg.NOMINAL_LEVELS:
            column = f"{column_stem}_reject_{common.level_tag(level)}"
            if column not in successful:
                continue
            summary = metrics.rejection_summary(successful[column].to_numpy(), level)
            band = simcfg.CALIBRATION_TOLERANCE_BANDS[level]
            outside = (
                summary["ci_high"] < band[0] or summary["ci_low"] > band[1]
                if np.isfinite(summary["rejection_rate"])
                else False
            )
            rows.append(
                {
                    "part": cell.part,
                    "cell_id": cell.cell_id,
                    "m": cell.m,
                    "n": cell.n,
                    "T": cell.T,
                    "persistence": cell.persistence,
                    "innovation": cell.innovation,
                    "statistic": label,
                    "k_over_T": (cell.m**2 + cell.n**2 - 1) / cell.T,
                    **summary,
                    "n_attempted": attempted,
                    "n_completed": int(len(successful)),
                    "n_failed": attempted - int(len(successful)),
                    "n_excluded": 0,
                    "tolerance_low": band[0],
                    "tolerance_high": band[1],
                    "flag_substantially_poor": bool(outside),
                }
            )

    return pd.DataFrame(rows)


def wild_bootstrap_table(cell, records: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison of asymptotic and bootstrap p-values.

    A finite-sample robustness comparison of the Nyblom-Hansen statistic. It is
    not a bootstrap-validity result and it does not define the primary test.

    Parameters
    ----------
    cell : Cell
        The bootstrap cell.
    records : pandas.DataFrame
        Its stored replication records.

    Returns
    -------
    pandas.DataFrame
        One row per nominal level and p-value source, with the paired
        correlation and the mean absolute difference.
    """
    successful = _successful(records)
    if successful.empty:
        return pd.DataFrame()

    correlation = (
        float(np.corrcoef(successful["p_asymptotic"], successful["p_bootstrap"])[0, 1])
        if len(successful) > 1
        else np.nan
    )
    mean_gap = float(
        np.mean(np.abs(successful["p_asymptotic"] - successful["p_bootstrap"]))
    )
    attempted = int(len(records[records["record_index"] == 0]))

    rows = []
    for level in simcfg.NOMINAL_LEVELS:
        tag = common.level_tag(level)
        for source in ("asymptotic", "bootstrap"):
            rows.append(
                {
                    "part": cell.part,
                    "cell_id": cell.cell_id,
                    "m": cell.m,
                    "n": cell.n,
                    "T": cell.T,
                    "statistic": f"nyblom_global_{source}",
                    "n_bootstrap_draws": int(successful["n_boot"].iloc[0]),
                    **metrics.rejection_summary(
                        successful[f"{source}_reject_{tag}"].to_numpy(), level
                    ),
                    "paired_pvalue_correlation": correlation,
                    "mean_absolute_pvalue_difference": mean_gap,
                    "n_attempted": attempted,
                    "n_completed": int(len(successful)),
                    "n_failed": attempted - int(len(successful)),
                    "n_excluded": 0,
                    "interpretation": "finite-sample robustness comparison",
                }
            )

    return pd.DataFrame(rows)


def garch_strength_table(cell, records: pd.DataFrame) -> pd.DataFrame:
    """Realised strength of the GARCH cell's conditional-variance factor.

    Measured, not assumed, because averaging the squared shock over the
    panel entries damps the factor.

    Parameters
    ----------
    cell : Cell
        The GARCH null cell.
    records : pandas.DataFrame
        Its stored replication records.

    Returns
    -------
    pandas.DataFrame
        One row with the averaged realised strength measures.
    """
    successful = _successful(records)
    if successful.empty or "h_std" not in successful:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "part": cell.part,
                "cell_id": cell.cell_id,
                "alpha": simcfg.GARCH_ALPHA,
                "beta": simcfg.GARCH_BETA,
                "n_replications": int(len(successful)),
                "mean_h_std": float(successful["h_std"].mean()),
                "mean_h_q05": float(successful["h_q05"].mean()),
                "mean_h_q95": float(successful["h_q95"].mean()),
                "mean_standardised_excess_kurtosis": float(
                    successful["standardised_excess_kurtosis"].mean()
                ),
            }
        ]
    )


# The recovery losses, raw and sample-scaled.
RECOVERY_LOSSES = ("L_Phi", "L_A", "L_B", "L_kappa")


def recovery_table(cell, records: pd.DataFrame) -> pd.DataFrame:
    """Canonical recovery of one stable null cell.

    One row per loss, with its mean, Monte Carlo standard error and scaled
    mean, alongside the convergence, stability-contraction and inversion-failure
    rates and the median condition numbers of the two matrices the statistic
    inverts.

    Each row also states the basis it rests on. The pre-run check
    measured the log-log slope of each mean loss on the number of score
    observations at 100 replications. Three of the four sat within about
    two standard errors of the expected minus one, and the log magnitude
    did not. That reading is carried on the row as provisional, beside
    the count this row is actually estimated on, so the production
    estimate and the pre-run check cannot be confused.

    Parameters
    ----------
    cell : Cell
        The null cell.
    records : pandas.DataFrame
        Its stored replication records.

    Returns
    -------
    pandas.DataFrame
        The recovery rows.
    """
    successful = _successful(records)
    if successful.empty or "L_Phi" not in successful:
        return pd.DataFrame()

    attempted = int(len(records[records["record_index"] == 0]))
    shared = {
        "part": cell.part,
        "cell_id": cell.cell_id,
        "m": cell.m,
        "n": cell.n,
        "T": cell.T,
        "N": simcfg.score_observations(cell.T),
        "k": cell.m**2 + cell.n**2 - 1,
        "persistence": cell.persistence,
        "innovation": cell.innovation,
        "convergence_rate": float(successful["converged"].mean()),
        "stability_contraction_rate": float(
            successful["stability_contracted"].mean()
        ),
        "hessian_inversion_failure_rate": float(
            successful["hessian_inversion_failed"].mean()
        ),
        "score_covariance_inversion_failure_rate": float(
            successful["score_covariance_inversion_failed"].mean()
        ),
        "median_cond_H": float(successful["cond_H"].median()),
        "median_cond_Omega": float(successful["cond_Omega"].median()),
        "median_foc_norm_canonical": float(
            successful["foc_norm_canonical"].median()
        ),
        "n_attempted": attempted,
        "n_completed": int(len(successful)),
        "n_failed": attempted - int(len(successful)),
        "n_excluded": 0,
    }

    rows = []
    for loss in RECOVERY_LOSSES:
        raw = metrics.monte_carlo_summary(successful[loss].to_numpy())
        scaled = metrics.monte_carlo_summary(successful[f"N_{loss}"].to_numpy())
        rows.append(
            {
                **shared,
                "loss": loss,
                # Which basis this row rests on, and what the pre-run
                # check of the convergence rate found. The two are
                # distinct measurements.
                "estimated_on_replications": int(
                    raw["n_replications"]
                ),
                "pre_run_check_loglog_slope": simcfg.PRE_RUN_LOGLOG_SLOPE[loss],
                "pre_run_check_replications": simcfg.PRE_RUN_SLOPE_REPLICATIONS,
                "pre_run_check_provisional": bool(
                    loss in simcfg.PRE_RUN_SLOPE_PROVISIONAL
                ),
                "mean_loss": raw["mean"],
                "loss_mcse": raw["mcse"],
                "median_loss": raw["median"],
                "scaled_mean_loss": scaled["mean"],
                "scaled_loss_mcse": scaled["mcse"],
                "scaled_median_loss": scaled["median"],
            }
        )

    return pd.DataFrame(rows)


def multiple_start_table(cell, records: pd.DataFrame) -> pd.DataFrame:
    """Summary of the numerical multiple-start check.

    Reports the proportion of replications on which every converged start
    reaches the same objective within the estimator's own tolerance, the median
    and maximum final-objective gap, the median and maximum aligned canonical
    distance, and the convergence rate of each start separately.

    Parameters
    ----------
    cell : Cell
        The multiple-start cell.
    records : pandas.DataFrame
        Its stored records, one row per start.

    Returns
    -------
    pandas.DataFrame
        One overall row plus one row per start.
    """
    successful = _successful(records)
    if successful.empty or "start_id" not in successful:
        return pd.DataFrame()

    tolerance = simcfg.MULTIPLE_START_OBJECTIVE_TOLERANCE
    converged = successful[successful["converged"]]

    agreed = (
        converged.groupby("replication_id")["relative_objective_gap"].max() <= tolerance
    )
    perturbed = successful[successful["start_id"] > 0]

    rows = [
        {
            "part": cell.part,
            "cell_id": cell.cell_id,
            "scope": "all_starts",
            "start_id": pd.NA,
            "n_replications": int(successful["replication_id"].nunique()),
            "n_starts": 1 + int(cell.extra["n_perturbed_starts"]),
            "objective_tolerance": tolerance,
            "proportion_all_converged_starts_agree": float(agreed.mean())
            if len(agreed)
            else np.nan,
            "median_objective_gap": float(perturbed["objective_gap"].abs().median()),
            "max_objective_gap": float(perturbed["objective_gap"].abs().max()),
            "median_aligned_canonical_distance": float(
                perturbed["aligned_canonical_distance"].median()
            ),
            "max_aligned_canonical_distance": float(
                perturbed["aligned_canonical_distance"].max()
            ),
            "convergence_rate": float(successful["converged"].mean()),
            "diagnostic_only": True,
        }
    ]

    for start_id, group in successful.groupby("start_id"):
        rows.append(
            {
                "part": cell.part,
                "cell_id": cell.cell_id,
                "scope": "by_start",
                "start_id": int(start_id),
                "n_replications": int(len(group)),
                "n_starts": 1 + int(cell.extra["n_perturbed_starts"]),
                "objective_tolerance": tolerance,
                "proportion_all_converged_starts_agree": np.nan,
                "median_objective_gap": float(group["objective_gap"].abs().median()),
                "max_objective_gap": float(group["objective_gap"].abs().max()),
                "median_aligned_canonical_distance": float(
                    group["aligned_canonical_distance"].median()
                ),
                "max_aligned_canonical_distance": float(
                    group["aligned_canonical_distance"].max()
                ),
                "convergence_rate": float(group["converged"].mean()),
                "diagnostic_only": True,
            }
        )

    return pd.DataFrame(rows)


def unrestricted_null_dimension_table(records: pd.DataFrame, m: int, n: int):
    """Mean null statistics against the two theoretical dimension means.

    Built on the reused null draws of the calibration cell that stores
    the unrestricted statistic. Nothing is simulated a second time for
    it. Under the null both statistics converge to the integrated
    squared norm of a Brownian bridge of their own dimension, whose mean
    is that dimension over six, so the extra central mass the
    unrestricted statistic carries is the dimension difference.

    Parameters
    ----------
    records : pandas.DataFrame
        The stored records of the null cell that also holds the unrestricted
        statistic.
    m, n : int
        Panel dimensions of that cell.

    Returns
    -------
    pandas.DataFrame
        One row per statistic with its mean, interval and theoretical mean.
    """
    successful = _successful(records)
    if successful.empty or "nyblom_var" not in successful:
        return pd.DataFrame()

    k = m * m + n * n - 1
    K = structural_tests.var_dimension(m, n)

    rows = []
    for statistic, column, dimension in (
        ("mar_global", "nyblom_global", k),
        ("var_global", "nyblom_var", K),
    ):
        summary = metrics.monte_carlo_summary(successful[column].to_numpy())
        rows.append(
            {
                "statistic": statistic,
                "dimension": dimension,
                "theoretical_mean": dimension / 6.0,
                **summary,
                "covers_theoretical_mean": bool(
                    summary["ci_low"] <= dimension / 6.0 <= summary["ci_high"]
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame["dimension_difference"] = K - k
    frame["theoretical_mean_difference"] = (K - k) / 6.0
    frame["observed_mean_difference"] = float(
        successful["nyblom_var"].mean() - successful["nyblom_global"].mean()
    )
    return frame

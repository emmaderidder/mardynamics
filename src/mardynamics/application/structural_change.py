"""Static reference, variance and constancy.

One formal constancy statistic is computed: the global canonical
Nyblom-Hansen statistic of the full-sample static fit. The wild bootstrap
recalibrates that same statistic and is not a second test family. The
three influence blocks are follow-up attribution with a Holm adjustment
over three p-values. The unrestricted VAR statistic is refused
analytically.

Every statistic, limit law, influence transformation, bootstrap and Holm
adjustment is the shared implementation. This module supplies the
empirical inputs, the interpretation-status rules and the output
columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mardynamics.application.emp_config as emp_config
import mardynamics.application.run_infrastructure as runinfra
import mardynamics.core.structural_tests as structural_tests
from mardynamics.core.mar_canonical import canonical_triple

# The three canonical blocks, in the order the shared battery reports
# them, with the block name it uses for each.
BLOCKS = (("row", "A"), ("column", "B"), ("amplitude", "gamma"))

# Where the 50,000-draw limit laws are cached, kept complete in preview.
# The directory is shared across runs because the law depends only on
# (k, draws, grid points, seed).
LIMIT_LAW_CACHE = "outputs/application/_limit_laws"


def _limit_law(k: int) -> np.ndarray:
    """Load or simulate the Nyblom-Hansen limit law of one dimension.

    Parameters
    ----------
    k : int
        Tested dimension.

    Returns
    -------
    numpy.ndarray
        The sorted draw vector.
    """
    return structural_tests.load_limit_law(
        k,
        LIMIT_LAW_CACHE,
        emp_config.LIMITLAW_N_DRAWS,
        emp_config.LIMITLAW_N_GRID,
        emp_config.LIMITLAW_SEED,
    )


def _critical_values(draws: np.ndarray) -> dict:
    """Return the 90, 95 and 99 per-cent critical values of a limit law.

    Parameters
    ----------
    draws : numpy.ndarray
        The simulated law.

    Returns
    -------
    dict
        The three critical values.
    """
    return {
        "critical_90": float(np.quantile(draws, 0.90)),
        "critical_95": float(np.quantile(draws, 0.95)),
        "critical_99": float(np.quantile(draws, 0.99)),
    }


def _decisions(pvalue: float) -> dict:
    """Return the nominal decisions at 1, 5 and 10 per cent.

    Parameters
    ----------
    pvalue : float
        The p-value.

    Returns
    -------
    dict
        ``"reject"`` or ``"do not reject"`` at each level.
    """
    return {
        f"decision_{int(level * 100)}pct": (
            "reject" if pvalue < level else "do not reject"
        )
        for level in emp_config.NOMINAL_LEVELS
    }


# =========================================================
# Static reference fit
# =========================================================
def static_reference(
    tensor: np.ndarray, fitted_pair, estimator_telemetry: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Report the full-sample static canonical reference.

    Reports the complete canonical row shape, column shape, amplitude,
    Kronecker transition operator and joint spectral radius.

    Parameters
    ----------
    tensor : numpy.ndarray
        The standardised full-sample tensor.
    fitted_pair : tuple of numpy.ndarray
        The static ``(A, B)`` fit.
    estimator_telemetry : dict or None, optional
        The NumPy estimator's telemetry for this fit (estimator identity,
        convergence status, iteration count, stopping reason, spectral
        radius and stability-contraction flag), appended as extra rows on
        the same ``quantity``/``row``/``column``/``value`` schema.

    Returns
    -------
    tuple
        The long-format table and a summary dictionary.
    """
    A, B = fitted_pair
    A_bar, B_breve, gamma = canonical_triple(A, B)
    phi = np.kron(B, A)
    jsr = float(np.max(np.abs(np.linalg.eigvals(phi))))

    rows = []
    for name, matrix, row_labels, column_labels in (
        ("row_shape_A_bar", A_bar, emp_config.INDICATORS, emp_config.INDICATORS),
        ("column_shape_B_breve", B_breve, emp_config.COUNTRIES, emp_config.COUNTRIES),
    ):
        for i, row_label in enumerate(row_labels):
            for j, column_label in enumerate(column_labels):
                rows.append(
                    {
                        "quantity": name,
                        "row": row_label,
                        "column": column_label,
                        "value": float(matrix[i, j]),
                    }
                )

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            rows.append(
                {
                    "quantity": "kronecker_transition_operator",
                    "row": f"vec{i}",
                    "column": f"vec{j}",
                    "value": float(phi[i, j]),
                }
            )

    for name, value in (
        ("amplitude_gamma", gamma),
        ("joint_spectral_radius", jsr),
        ("row_spectral_radius", float(np.max(np.abs(np.linalg.eigvals(A))))),
        ("column_spectral_radius", float(np.max(np.abs(np.linalg.eigvals(B))))),
    ):
        rows.append(
            {"quantity": name, "row": "", "column": "", "value": float(value)}
        )

    summary = {
        "amplitude_gamma": float(gamma),
        "joint_spectral_radius": jsr,
        "n_matrices": int(tensor.shape[0]),
        "n_transitions": int(tensor.shape[0] - 1),
    }

    if estimator_telemetry is not None:
        for key, value in estimator_telemetry.items():
            rows.append(
                {
                    "quantity": f"estimator_{key}",
                    "row": "",
                    "column": "",
                    "value": value if isinstance(value, str) else float(value),
                }
            )
        summary.update(
            {f"estimator_{key}": value for key, value in estimator_telemetry.items()}
        )

    return pd.DataFrame(rows), summary


# =========================================================
# Innovation-variance diagnostic
# =========================================================
def innovation_variance(tensor: np.ndarray, fitted_pair) -> dict:
    """Apply the auxiliary fluctuation statistic to the squared innovation norm.

    Uses the Bartlett long-run variance with the bandwidth rule fixed in
    the shared numerical configuration. The statistic tests one feature
    the constancy test's reference law needs: stability of the
    unconditional innovation variance. It does not test the lag order or
    the Kronecker restriction.

    Parameters
    ----------
    tensor : numpy.ndarray
        The standardised full-sample tensor.
    fitted_pair : tuple of numpy.ndarray
        The static fit.

    Returns
    -------
    dict
        Statistic, p-value, bandwidth and nominal decisions.
    """
    result = structural_tests.innovation_variance_statistic(tensor, fitted_pair)
    draws = _limit_law(1)
    pvalue = structural_tests.limit_law_pvalue(draws, result["statistic"])

    return {
        "statistic": float(result["statistic"]),
        "p_value": float(pvalue),
        "bandwidth": int(result["bandwidth"]),
        "n_score_observations": int(result["T"]),
        **_critical_values(draws),
        **_decisions(pvalue),
    }


def variance_constancy_status(variance_pvalue: float, constancy_pvalue: float) -> str:
    """Return the joint interpretation status of the two diagnostics.

    Three readings are possible. A non-rejection of the variance
    diagnostic proves nothing about the other innovation assumptions. A
    variance rejection with a constancy non-rejection does not support
    stable operators. A variance rejection with a constancy rejection
    leaves the numerical rejection standing, but it cannot be separated
    from innovation-variance instability.

    Parameters
    ----------
    variance_pvalue : float
        p-value of the innovation-variance diagnostic.
    constancy_pvalue : float
        p-value of the global constancy test.

    Returns
    -------
    str
        The status string stored in every relevant output.
    """
    level = emp_config.PRIMARY_LEVEL
    variance_rejects = variance_pvalue < level
    constancy_rejects = constancy_pvalue < level

    if not variance_rejects:
        return (
            "no_variance_rejection: the unconditional innovation variance is "
            "not rejected at 5%, which does not prove the remaining innovation "
            "assumptions"
        )
    if not constancy_rejects:
        return (
            "variance_rejection_without_constancy_rejection: a 5% variance "
            "rejection with a 5% constancy non-rejection does not support "
            "stable operators"
        )
    return (
        "variance_rejection_with_constancy_rejection: the numerical constancy "
        "rejection stands, but the evidence cannot be cleanly separated from "
        "innovation-variance instability and does not isolate operator "
        "movement alone"
    )


# =========================================================
# Global constancy, bootstrap, VAR row, blocks, paths
# =========================================================
def global_constancy(
    tensor: np.ndarray, fitted_pair, quantities: dict, mode: str, failures
) -> dict:
    """Compute the global Nyblom statistic and its wild-bootstrap calibration.

    Parameters
    ----------
    tensor : numpy.ndarray
        The standardised full-sample tensor.
    fitted_pair : tuple of numpy.ndarray
        The static fit.
    quantities : dict
        The shared chart-score quantities of the same fit.
    mode : str
        Run mode; fixes the number of bootstrap draws.
    failures : FailureLog
        Failure accounting.

    Returns
    -------
    dict
        Every field of ``constancy_global.csv``.
    """
    battery = structural_tests.nyblom_battery(tensor, *fitted_pair, quantities=quantities)
    draws = _limit_law(battery.k)
    pvalue = structural_tests.limit_law_pvalue(draws, battery.nyblom)

    n_boot = emp_config.wild_bootstrap_draws(mode)
    bootstrap_p = float("nan")
    try:
        bootstrap = structural_tests.wild_bootstrap_pvalue(
            tensor,
            float(battery.nyblom),
            fitted_pair,
            # WILD_BOOT_SEED is used directly, the same way the limit law
            # uses its own seed through load_limit_law, so the seed the
            # table reports is the seed the generator was built from.
            np.random.default_rng(emp_config.WILD_BOOT_SEED),
            n_boot=n_boot,
            enforce_stability=emp_config.ENFORCE_STABILITY,
            max_radius=emp_config.MAX_RADIUS,
        )
        # The shared bootstrap already applies the plus-one formula
        # (1 + #{stat_b >= stat_obs}) / (n_boot + 1) and names it p_bootstrap.
        bootstrap_p = float(bootstrap["p_bootstrap"])
    except Exception as error:  # pragma: no cover - recorded, not hidden
        failures.record(
            unit="wild_bootstrap",
            identity={"sample": "full", "draws": n_boot},
            reason="exception",
            message=str(error),
        )

    asymptotic_rejects = pvalue < emp_config.PRIMARY_LEVEL
    bootstrap_rejects = bootstrap_p < emp_config.PRIMARY_LEVEL
    agreement = (
        "not_available"
        if not np.isfinite(bootstrap_p)
        else (
            "agree" if asymptotic_rejects == bootstrap_rejects else "disagree"
        )
    )

    benchmark = emp_config.SIMULATION_BENCHMARK
    return {
        "statistic_name": "global_canonical_nyblom_hansen",
        "statistic": float(battery.nyblom),
        "k": int(battery.k),
        # chart_score_quantities already sets T = L - 1, so battery.T is
        # the number of usable score contributions. Subtracting one again
        # would double-count the reduction.
        "n_score_observations": int(battery.T),
        **_critical_values(draws),
        "p_value": float(pvalue),
        **_decisions(pvalue),
        "limit_law_draws": int(emp_config.LIMITLAW_N_DRAWS),
        "limit_law_grid_points": int(emp_config.LIMITLAW_N_GRID),
        "bootstrap_draws": int(n_boot),
        "wild_bootstrap_seed": int(emp_config.WILD_BOOT_SEED),
        "bootstrap_p_value": bootstrap_p,
        "bootstrap_agreement_status": agreement,
        # Reported once, not computed.
        "var_parameter_directions": int(emp_config.K_VAR),
        "var_score_rank_bound": int(tensor.shape[0] - 2),
        "var_feasibility_status": (
            f"analytically_infeasible: K={emp_config.K_VAR} parameter directions "
            f"exceed the score-covariance rank bound T-2={tensor.shape[0] - 2}; "
            "the statistic is not computed and no pseudo-inverse or ridge is used"
        ),
        # One concise comparison, recorded not narrated.
        "k_over_n": float(battery.k / battery.T),
        "benchmark_cell": benchmark["benchmark_cell"],
        "benchmark_k_over_n": float(benchmark["benchmark_k_over_N"]),
        "benchmark_size_at_5pct": float(benchmark["benchmark_size_at_5pct"]),
        "simulation_benchmark_statement": benchmark["statement"],
        "constancy_decision_5pct": (
            "reject" if asymptotic_rejects else "do not reject"
        ),
        "cond_H": float(battery.cond_H),
        "cond_Omega": float(battery.cond_Omega),
    }


def block_attribution(
    tensor: np.ndarray, fitted_pair, quantities: dict, global_pvalue: float
) -> pd.DataFrame:
    """Compute the three influence-based blocks with a Holm adjustment.

    Every block uses ``R_j H_hat^(-1) g_t`` with covariance ``R_j V_hat
    R_j'``, which is what the shared battery computes, and the Holm
    adjustment runs over three p-values.

    Parameters
    ----------
    tensor : numpy.ndarray
        The standardised full-sample tensor.
    fitted_pair : tuple of numpy.ndarray
        The static fit.
    quantities : dict
        The shared chart-score quantities.
    global_pvalue : float
        p-value of the global test, which decides whether the blocks may be
        interpreted substantively.

    Returns
    -------
    pandas.DataFrame
        One row per block.
    """
    battery = structural_tests.nyblom_battery(tensor, *fitted_pair, quantities=quantities)
    by_name = dict(zip(battery.block_names, battery.block_nyblom))
    dims = dict(zip(battery.block_names, battery.block_k))

    statistics, pvalues, laws = [], [], {}
    for _, shared_name in BLOCKS:
        k_block = int(dims[shared_name])
        draws = laws.setdefault(k_block, _limit_law(k_block))
        statistic = float(by_name[shared_name])
        statistics.append(statistic)
        pvalues.append(structural_tests.limit_law_pvalue(draws, statistic))

    holm = structural_tests.holm_adjusted_pvalues(pvalues)

    interpretable = global_pvalue < emp_config.PRIMARY_LEVEL
    status = (
        "substantive_first_order_attribution: the global null is rejected at "
        "5%; blocks are attribution, not three separate break tests, do not "
        "sum to the global statistic and are not comparable across dimensions"
        if interpretable
        else "descriptive_only: the global null is not rejected at 5%, so no "
        "conclusion about where change resides is drawn"
    )

    rows = []
    for index, (block, shared_name) in enumerate(BLOCKS):
        k_block = int(dims[shared_name])
        rows.append(
            {
                "block": block,
                "shared_block_name": shared_name,
                "block_dimension": k_block,
                "statistic": statistics[index],
                "p_value": float(pvalues[index]),
                "holm_p_value": float(holm[index]),
                **_critical_values(laws[k_block]),
                "decision_5pct": (
                    "reject"
                    if pvalues[index] < emp_config.PRIMARY_LEVEL
                    else "do not reject"
                ),
                "holm_decision_5pct": (
                    "reject" if holm[index] < emp_config.PRIMARY_LEVEL else "do not reject"
                ),
                "n_holm_hypotheses": len(BLOCKS),
                "attribution_status": status,
            }
        )
    return pd.DataFrame(rows)


def influence_paths(quantities: dict, periods: list[str]) -> pd.DataFrame:
    """Build the four coordinate-invariant influence paths.

    The global path standardises the cumulative score by the score
    covariance. Each block path standardises the cumulative influence
    process ``R_j H_hat^(-1) S_t`` by ``R_j V_hat R_j'``. Both are
    quadratic forms and are invariant to a smooth change of chart within
    a block.

    Parameters
    ----------
    quantities : dict
        The shared chart-score quantities.
    periods : list of str
        The ordered months of the tensor the scores came from.

    Returns
    -------
    pandas.DataFrame
        One row per period and path, in long format.
    """
    S = np.asarray(quantities["S"], dtype=float)
    n_obs = S.shape[0]
    influence = np.linalg.solve(np.asarray(quantities["H_hat"], dtype=float), S.T).T

    # The score contributions run from the second matrix onwards.
    score_periods = list(periods[1 : n_obs + 1])

    def _standardised(values: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        solved = np.linalg.solve(covariance, values.T).T
        return np.sqrt(np.maximum(np.einsum("ti,ti->t", values, solved), 0.0) / n_obs)

    paths = {
        "global": _standardised(S, np.asarray(quantities["Omega_hat"], dtype=float))
    }
    V_hat = np.asarray(quantities["V_hat"], dtype=float)
    for block, shared_name in BLOCKS:
        sl = quantities[f"mode_{shared_name}"]
        paths[block] = _standardised(influence[:, sl], V_hat[sl, sl])

    rows = []
    for name, values in paths.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "period": score_periods[index],
                    "observation_index": index + 1,
                    "path": name,
                    "value": float(value),
                }
            )
    return pd.DataFrame(rows)

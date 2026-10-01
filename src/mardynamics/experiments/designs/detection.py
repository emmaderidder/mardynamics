"""Detection, conditional mode-specific attribution and influence paths.

The main grid is the full factorial of three changing blocks, three
scalar paths and three departure sizes at the reference population.
Because every planted direction carries the same first-order global
signal under the population metric, a row, a column and an amplitude
change driven by the same path are comparable on that scale by
construction. They are not comparable on the blockwise signal, which
differs between them and is reported separately, letting a difference
in blockwise rejection rates be discussed as blockwise signal and as
block dimension instead of being attributed to either alone.

The global Nyblom-Hansen test is the formal stability decision. The
Holm-adjusted block results are follow-up attribution measures and
count only after a global rejection: every conditional quantity in this
module is conditioned on the observed global rejections, carries that
denominator beside it, and is missing, not zero, when there were none.

The same routines run the sensitivity branch over the two additional
population configurations, because that experiment differs from the
main one only in which population and which directions it plants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mardynamics.experiments.design as design
import mardynamics.experiments.dgp as dgp
import mardynamics.experiments.innovations as innovations
import mardynamics.experiments.metrics as metrics
import mardynamics.experiments.sim_config as simcfg
import mardynamics.core.structural_tests as structural_tests
from mardynamics.experiments.designs import common

# Chart block name the shared statistics use for each canonical block.
BLOCK_TO_STATISTIC = simcfg.BLOCK_TO_STATISTIC

# The eight quantities per data-generating process, in this order.
# ``joint`` rows are unconditional and carry a Monte Carlo standard
# error. ``conditional`` rows are conditioned on the observed global
# rejections and carry a descriptive binomial one.
ATTRIBUTION_MEASURES = (
    ("global_rejection", "global_reject", "unconditional"),
    ("correct_joint", "global_and_correct", "unconditional"),
    ("correct_given_global", "global_and_correct", "conditional"),
    ("correct_only_given_global", "global_and_correct_only", "conditional"),
    ("false_given_global", "global_and_false", "conditional"),
    ("correct_and_false_given_global", "global_and_correct_and_false", "conditional"),
    ("false_only_given_global", "global_and_false_only", "conditional"),
    ("none_flagged_given_global", "global_and_none_flagged", "conditional"),
)

# The four mutually exclusive conditional classifications, which must
# sum to one within tolerance whenever there was at least one global
# rejection.
CLASSIFICATION_MEASURES = (
    "correct_only_given_global",
    "correct_and_false_given_global",
    "false_only_given_global",
    "none_flagged_given_global",
)


def _inverse_square_root(matrix: np.ndarray) -> np.ndarray:
    """Symmetric inverse square root of a positive definite matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Symmetric positive definite matrix.

    Returns
    -------
    numpy.ndarray
        Its symmetric inverse square root.

    Raises
    ------
    mardynamics.core.structural_tests.SingularCovarianceError
        If the matrix is not positive definite.
    """
    values, vectors = np.linalg.eigh(np.asarray(matrix, dtype=float))
    if np.any(values <= 0.0):
        raise structural_tests.SingularCovarianceError(
            "the block covariance is not positive definite; its standardisation "
            "is undefined."
        )
    return vectors @ np.diag(values**-0.5) @ vectors.T


def build_context(cell) -> dict:
    """Build the population, direction, path and generator of one cell.

    The designed path is verified against the maintained simulation design here,
    before any replication runs, so a violating design fails at construction
    rather than producing records.

    Parameters
    ----------
    cell : Cell
        The cell.

    Returns
    -------
    dict
        The per-cell design objects and the path verification record.
    """
    truth = design.build_truth(cell.m, cell.n, cell.persistence, cell.configuration)
    G = design.signal_metric(truth, simcfg.SIGMA)
    V = design.parameter_covariance(truth, simcfg.SIGMA)
    direction = design.planted_direction(truth, cell.block, G, V)
    built = design.build_parameter_path(
        truth, direction, cell.path, cell.magnitude, cell.T
    )
    verification = design.verify_parameter_path(truth, direction, built, strict=True)

    return {
        "truth": truth,
        "G": G,
        "V": V,
        "direction": direction,
        "path": built,
        "verification": verification,
        "innovation": innovations.build_innovation(cell.innovation, simcfg.SIGMA),
        "store_influence_path": "influence_path" in cell.statistics,
    }


def projected_influence_path(
    quantities: dict, direction: design.PlantedDirection, block: str
) -> np.ndarray:
    """Project the standardised block influence path on the planted direction.

    The block influence process is standardised by its own estimated
    covariance, the same way as in the methodology's path result, and
    then projected on the planted block direction standardised the same
    way. The result is one scalar series per replication whose
    theoretical mean is the cumulative demeaned signature with a minus
    sign.

    Parameters
    ----------
    quantities : dict
        The chart score quantities of the fitted replication.
    direction : PlantedDirection
        The planted direction; its block coordinates are what is projected on.
    block : str
        Which canonical block moved.

    Returns
    -------
    numpy.ndarray
        The projected standardised path, length ``N``.
    """
    name = BLOCK_TO_STATISTIC[block]
    sl = quantities[f"mode_{name}"]
    N = quantities["T"]

    influence = np.cumsum(quantities["g_influence"][:, sl], axis=0)
    root = _inverse_square_root(quantities["V_hat"][sl, sl])

    standardised = root @ direction.block_coordinates
    norm = float(np.linalg.norm(standardised))
    unit = standardised / norm if norm > 0 else standardised

    return (influence @ root.T @ unit) / np.sqrt(N)


def population_block_covariance(context: dict, cell) -> np.ndarray:
    """Population covariance of the changing block.

    Parameters
    ----------
    context : dict
        The cell context.
    cell : Cell
        The cell.

    Returns
    -------
    numpy.ndarray
        The population block covariance.
    """
    R = design.selection_matrix(context["truth"], cell.block)
    return R @ context["V"] @ R.T


def theoretical_projection(
    cell, context: dict, block_covariance: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Theoretical mean of the projected standardised influence path.

    From the methodology's path result the standardised block path
    converges to a Brownian bridge minus the standardised cumulative
    demeaned signature. Projecting on the standardised planted direction
    leaves a scalar multiple of that signature.

    The coefficient is the local one, not the scalar that multiplies the
    direction in the path itself: the standardisation of the influence
    path already divides by the square root of the sample, which is
    what makes a local sequence produce a bounded signature.

    The two standardisations are asymmetric and must not be described as
    though they were the same. This function is handed the
    population block covariance, so the theoretical curve is the
    population one. :func:`projected_influence_path` standardises each
    replication by that replication's own estimated ``V_hat`` block,
    because that is the only covariance a replication has. Estimating
    the standardiser is one of the two reasons the Monte Carlo mean sits
    slightly below the population curve at a finite sample (the other
    being that a local departure is not infinitesimal there), and both
    shrink with the sample, which is what
    :func:`mardynamics.experiments.figures.attenuation_at_longer_sample`
    measures.

    Parameters
    ----------
    cell : Cell
        The cell.
    context : dict
        The cell context.
    block_covariance : numpy.ndarray
        The block covariance the path was standardised by.
    r : numpy.ndarray
        Fractions in the unit interval.

    Returns
    -------
    numpy.ndarray
        The theoretical mean at each fraction.

    Raises
    ------
    ValueError
        For the fixed magnitude, whose departure is not local and whose
        standardised path has no bounded signature.
    """
    local_scalar = context["path"].local_scalar
    if local_scalar is None:
        raise ValueError(
            "the theoretical signature is defined for a LOCAL departure only; "
            f"cell {cell.cell_id} uses the fixed departure {cell.magnitude!r}."
        )

    root = _inverse_square_root(block_covariance)
    scale = float(np.linalg.norm(root @ context["direction"].block_coordinates))
    return -local_scalar * scale * design.theoretical_signature(cell.path, r)


def simulate(cell, context: dict, replication_id: int) -> np.ndarray:
    """Simulate one replication dataset.

    Kept separate from :func:`execute`, letting the same dataset be
    rebuilt later from its stored seed without refitting anything.

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
    return dgp.simulate_path_tensor(
        context["path"],
        context["innovation"],
        common.dataset_rng(cell, replication_id),
    )


def execute(cell, context: dict, replication_id: int) -> list[dict]:
    """Run one detection replication.

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
    row.update(common.gated_attribution_row(row, cell.block))

    row["population_configuration"] = cell.configuration
    row["changing_block"] = cell.block
    row["path"] = cell.path
    row["magnitude"] = cell.magnitude
    row["noncentrality"] = context["path"].noncentrality
    row["blockwise_signal"] = context["direction"].blockwise_signal
    row["global_signal_error"] = context["direction"].global_signal_deviation

    if context["store_influence_path"]:
        row["influence_projection"] = projected_influence_path(
            fit["quantities"], context["direction"], cell.block
        )

    return [row]


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


def attribution_rows(
    cell, records: pd.DataFrame, n_rep: int | None = None
) -> list[dict]:
    """The eight conditional attribution quantities of one cell, at every level.

    Parameters
    ----------
    cell : Cell
        The cell.
    records : pandas.DataFrame
        Its stored replication records.
    n_rep : int or None, optional
        Use only the first this many replications. This is how the
        sensitivity branch reuses the leading replications of a main
        cell instead of drawing new ones.

    Returns
    -------
    list of dict
        One row per measure and level.
    """
    successful = _successful(records)
    if n_rep is not None:
        successful = successful[successful["replication_id"] < int(n_rep)]
    if successful.empty:
        return []

    attempted = int(len(records[records["record_index"] == 0]))
    if n_rep is not None:
        attempted = min(attempted, int(n_rep))

    shared = {
        "part": cell.part,
        "cell_id": cell.cell_id,
        "population_configuration": cell.configuration,
        "m": cell.m,
        "n": cell.n,
        "T": cell.T,
        "changing_block": cell.block,
        "path": cell.path,
        "magnitude": cell.magnitude,
        "block_dimension": simcfg.block_dimension(cell.block, cell.m, cell.n),
        "blockwise_signal": float(successful["blockwise_signal"].iloc[0]),
        "global_signal_error": float(successful["global_signal_error"].iloc[0]),
        "noncentrality": float(successful["noncentrality"].iloc[0]),
        "n_attempted": attempted,
        "n_completed": int(len(successful)),
        "n_failed": attempted - int(len(successful)),
        "n_excluded": 0,
    }

    rows = []
    for level in simcfg.NOMINAL_LEVELS:
        tag = common.level_tag(level)
        gate = successful[f"global_reject_{tag}"].to_numpy(dtype=bool)

        for label, column, scope in ATTRIBUTION_MEASURES:
            values = successful[f"{column}_{tag}"].to_numpy(dtype=bool)
            if scope == "unconditional":
                summary = metrics.rejection_summary(values, level)
                rows.append(
                    {
                        **shared,
                        "level": level,
                        "measure": label,
                        "scope": scope,
                        "rate": summary["rejection_rate"],
                        "mcse": summary["mcse"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "conditional_denominator": pd.NA,
                        "descriptive_mc_se": np.nan,
                        "uncertainty_kind": "monte_carlo_standard_error",
                    }
                )
            else:
                summary = metrics.conditional_summary(values, gate)
                rows.append(
                    {
                        **shared,
                        "level": level,
                        "measure": label,
                        "scope": scope,
                        "rate": summary["conditional_rate"],
                        "mcse": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "conditional_denominator": summary["conditional_denominator"],
                        "descriptive_mc_se": summary["descriptive_mc_se"],
                        # This is descriptive Monte Carlo uncertainty at
                        # the observed denominator, not conditional error
                        # control.
                        "uncertainty_kind": "descriptive_binomial_at_observed_denominator",
                    }
                )

    return rows


def classification_check(rows: list[dict]) -> list[dict]:
    """Verify that the four conditional classifications sum to one.

    Parameters
    ----------
    rows : list of dict
        The attribution rows of one or more cells.

    Returns
    -------
    list of dict
        One row per cell and level with the realised sum and its verdict. A
        cell with no global rejection has no classification to check and is
        reported as missing, not as a failure.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []

    checks = []
    for (cell_id, level), group in frame.groupby(["cell_id", "level"]):
        selected = group[group["measure"].isin(CLASSIFICATION_MEASURES)]
        total = float(selected["rate"].sum(skipna=False))
        denominator = selected["conditional_denominator"].iloc[0]
        checks.append(
            {
                "cell_id": cell_id,
                "level": level,
                "conditional_denominator": denominator,
                "classification_sum": total,
                "tolerance": simcfg.CLASSIFICATION_SUM_TOLERANCE,
                "sums_to_one": (
                    bool(abs(total - 1.0) <= simcfg.CLASSIFICATION_SUM_TOLERANCE)
                    if np.isfinite(total)
                    else pd.NA
                ),
            }
        )
    return checks


def unconditional_block_rows(cell, records: pd.DataFrame) -> list[dict]:
    """Unconditional Holm rejection rates of the three individual blocks.

    These are the block rates before conditioning on the global rejection,
    reported beside the block dimension and the blockwise signal so
    neither is read as the sole explanation of the other.

    Parameters
    ----------
    cell : Cell
        The cell.
    records : pandas.DataFrame
        Its stored replication records.

    Returns
    -------
    list of dict
        One row per block and level.
    """
    successful = _successful(records)
    if successful.empty:
        return []

    rows = []
    for level in simcfg.NOMINAL_LEVELS:
        tag = common.level_tag(level)
        for block, name in BLOCK_TO_STATISTIC.items():
            summary = metrics.rejection_summary(
                successful[f"holm_{name}_reject_{tag}"].to_numpy(dtype=bool), level
            )
            rows.append(
                {
                    "part": cell.part,
                    "cell_id": cell.cell_id,
                    "population_configuration": cell.configuration,
                    "m": cell.m,
                    "n": cell.n,
                    "T": cell.T,
                    "changing_block": cell.block,
                    "path": cell.path,
                    "magnitude": cell.magnitude,
                    "rejecting_block": block,
                    "is_intended_block": block == cell.block,
                    "block_dimension": simcfg.block_dimension(block, cell.m, cell.n),
                    "adjustment": "holm_over_three_blocks",
                    **summary,
                    "raw_block_rate": float(
                        successful[f"raw_{name}_reject_{tag}"].mean()
                    ),
                }
            )
    return rows

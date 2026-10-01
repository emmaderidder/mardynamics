"""The value and the limits of the matrix restriction.

Three conditions at the one dimension where the unrestricted statistic
exists, each at three departure sizes and all on the abrupt-break path:

1. a separable row-shape break under isotropic Gaussian innovations,
   where the restricted and unrestricted statistics differ by a purely
   central component.
2. the same separable structural path under contemporaneously
   correlated homoskedastic innovations. Only the innovation covariance
   changes, and the structural departure is not renormalised under it.
3. a non-separable break along a transition direction that satisfies
   the tangent-orthogonality condition, which the least-squares
   Kronecker projection cannot see at first order.

Each replication computes both statistics from the same dataset and the
same fit, and each is read against its own directly simulated limit
law: the restricted dimension for the matrix test and the unrestricted
one for the vectorised test.

The third condition is why no uniform power ordering is claimed
anywhere. Under it the unrestricted transition matrix is unstable while
the pseudo-true restricted projection does not move at first order, so
the restricted rejection rate there is not ordinary power against a
false restricted null, and the output labels it as what it is.
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

# How the restricted rejection rate of each condition may be read. The
# non-separable condition is the reason this column exists: its
# restricted rate is not power against a false restricted null.
CONDITION_INTERPRETATION = {
    "separable_isotropic": "restricted_projection_moves",
    "separable_correlated": "restricted_projection_moves",
    "unrestricted": "restricted_projection_static_at_first_order",
}

# What is actually unstable under each condition.
CONDITION_INSTABILITY = {
    "separable_isotropic": "separable_canonical_parameter",
    "separable_correlated": "separable_canonical_parameter",
    "unrestricted": "unrestricted_transition_matrix_only",
}


def build_context(cell) -> dict:
    """Build the population and the designed departure of one condition.

    Both separable conditions plant the same direction, normalised
    under the isotropic design. Only the innovation covariance differs
    between them, and the structural departure is not renormalised
    under the correlated one.

    Parameters
    ----------
    cell : Cell
        The cell.

    Returns
    -------
    dict
        The per-cell design objects. A separable condition carries a canonical
        parameter path; the non-separable condition carries a transition-matrix
        path instead.
    """
    truth = design.build_truth(cell.m, cell.n, cell.persistence, cell.configuration)
    condition = cell.extra["condition"]
    context = {
        "truth": truth,
        "condition": condition,
        "innovation": innovations.build_innovation(cell.innovation, simcfg.SIGMA),
    }

    if condition == "unrestricted":
        departure = design.orthogonal_departure(truth, simcfg.SIGMA)
        context["departure"] = departure
        context["Phi_path"] = design.unrestricted_transition_path(
            truth, departure, cell.path, cell.magnitude, cell.T
        )
        context["unrestricted_signal"] = departure.unrestricted_signal
        context["tangent_orthogonality"] = departure.orthogonality
        context["rank_feasibility"] = departure.rank_feasibility
    else:
        G = design.signal_metric(truth, simcfg.SIGMA)
        V = design.parameter_covariance(truth, simcfg.SIGMA)
        direction = design.planted_direction(truth, cell.block, G, V)
        built = design.build_parameter_path(
            truth, direction, cell.path, cell.magnitude, cell.T
        )
        induced = design.induced_separable_direction(truth, direction, simcfg.SIGMA)

        context["direction"] = direction
        context["path"] = built
        context["induced"] = induced
        context["verification"] = design.verify_parameter_path(
            truth, direction, built, strict=True
        )
        context["unrestricted_signal"] = induced.unrestricted_signal
        # A separable direction lies in the Kronecker tangent space, so
        # its orthogonality residual is not zero and is not supposed to
        # be. The condition belongs to the non-separable direction
        # alone.
        context["tangent_orthogonality"] = np.nan
        context["rank_feasibility"] = {
            "k": int(truth.k),
            "K": int((cell.m * cell.n) ** 2),
            "jacobian_rank": int(
                np.linalg.matrix_rank(design.kronecker_jacobian(truth))
            ),
            "complement_dimension": int((cell.m * cell.n) ** 2 - truth.k),
        }

    return context


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
    rng = common.dataset_rng(cell, replication_id)
    if context["condition"] == "unrestricted":
        return dgp.simulate_unrestricted_tensor(
            context["Phi_path"], cell.m, cell.n, context["innovation"], rng
        )
    return dgp.simulate_path_tensor(context["path"], context["innovation"], rng)


def execute(cell, context: dict, replication_id: int) -> list[dict]:
    """Run one matrix-restriction replication.

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
    row["condition"] = context["condition"]
    row["magnitude"] = cell.magnitude
    row["path"] = cell.path

    K = structural_tests.var_dimension(cell.m, cell.n)
    var_statistic = structural_tests.var_nyblom_statistic(tensor)
    p_var = common.pvalue(K, var_statistic)

    row["nyblom_var"] = float(var_statistic)
    row["K"] = int(K)
    row["p_var"] = p_var
    row.update(common.rejection_flags("var", p_var))

    return [row]


# =========================================================
# Tables
# =========================================================
def matrix_restriction_table(cell, context: dict, records: pd.DataFrame):
    """Restricted and unrestricted rejection rates of one condition.

    Parameters
    ----------
    cell : Cell
        The cell.
    context : dict
        Its context, for the signal norms and rank quantities.
    records : pandas.DataFrame
        Its stored replication records.

    Returns
    -------
    pandas.DataFrame
        One row per level, carrying both rates, their standard errors, their
        ratio and the design quantities of the condition.
    """
    successful = records[records["status"] == "ok"] if "status" in records else records
    if successful.empty:
        return pd.DataFrame()

    condition = cell.extra["condition"]
    attempted = int(len(records[records["record_index"] == 0]))
    k = cell.m**2 + cell.n**2 - 1
    K = structural_tests.var_dimension(cell.m, cell.n)
    rank = context["rank_feasibility"]

    rows = []
    for level in simcfg.NOMINAL_LEVELS:
        tag = common.level_tag(level)
        mar = metrics.rejection_summary(
            successful[f"global_reject_{tag}"].to_numpy(dtype=bool), level
        )
        var = metrics.rejection_summary(
            successful[f"var_reject_{tag}"].to_numpy(dtype=bool), level
        )
        contrast = metrics.rate_contrast(mar, var)

        rows.append(
            {
                "part": cell.part,
                "cell_id": cell.cell_id,
                "condition": condition,
                "innovation": cell.innovation,
                "path": cell.path,
                "magnitude": cell.magnitude,
                "level": level,
                "k": k,
                "K": K,
                "mar_rejection_rate": mar["rejection_rate"],
                "mar_mcse": mar["mcse"],
                "var_rejection_rate": var["rejection_rate"],
                "var_mcse": var["mcse"],
                "mar_to_var_ratio": contrast["rate_ratio"],
                "rate_difference": contrast["rate_difference"],
                "difference_se": contrast["difference_se"],
                "separable_unrestricted_signal": (
                    context["unrestricted_signal"]
                    if condition != "unrestricted"
                    else np.nan
                ),
                "non_separable_unrestricted_signal": (
                    context["unrestricted_signal"]
                    if condition == "unrestricted"
                    else np.nan
                ),
                "tangent_orthogonality_error": context["tangent_orthogonality"],
                "jacobian_rank": rank["jacobian_rank"],
                "complement_dimension": rank["complement_dimension"],
                "what_is_unstable": CONDITION_INSTABILITY[condition],
                "mar_rate_reading": CONDITION_INTERPRETATION[condition],
                "n_attempted": attempted,
                "n_completed": int(len(successful)),
                "n_failed": attempted - int(len(successful)),
                "n_excluded": 0,
            }
        )

    return pd.DataFrame(rows)


def mean_statistic_table(cell, records: pd.DataFrame) -> pd.DataFrame:
    """Finite-sample mean statistics against the two dimension means.

    Under the null both statistics converge to the integrated squared
    norm of a Brownian bridge of their own dimension, whose mean is that
    dimension over six. Under an alternative the comparison is
    descriptive only.

    Parameters
    ----------
    cell : Cell
        The cell.
    records : pandas.DataFrame
        Its stored replication records.

    Returns
    -------
    pandas.DataFrame
        One row per statistic.
    """
    successful = records[records["status"] == "ok"] if "status" in records else records
    if successful.empty:
        return pd.DataFrame()

    k = cell.m**2 + cell.n**2 - 1
    K = structural_tests.var_dimension(cell.m, cell.n)

    rows = []
    for statistic, column, dimension in (
        ("mar_global", "nyblom_global", k),
        ("var_global", "nyblom_var", K),
    ):
        rows.append(
            {
                "part": cell.part,
                "cell_id": cell.cell_id,
                "condition": cell.extra["condition"],
                "innovation": cell.innovation,
                "magnitude": cell.magnitude,
                "statistic": statistic,
                "dimension": dimension,
                "dimension_over_six": dimension / 6.0,
                **metrics.monte_carlo_summary(successful[column].to_numpy()),
            }
        )

    return pd.DataFrame(rows)

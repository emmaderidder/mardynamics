"""Helpers shared by the experiment sections.

Nothing here decides a design. These are the small routines every
section needs in the same form: the memoised limit-law cache, the
p-value and rejection conventions, the one fit-and-test call that
guarantees competing statistics see the same data and the same fit, the
numerical-reliability record, the conditional attribution events, and
the seeding rule that ties a replication to its dataset stream instead
of to the cell that happens to consume it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import mardynamics.config as paths_cfg
import mardynamics.experiments.estimators as estimators
import mardynamics.experiments.metrics as metrics
import mardynamics.experiments.seeds as seeds
import mardynamics.experiments.sim_config as simcfg
import mardynamics.core.structural_tests as structural_tests
from mardynamics.core.mar_canonical import canonical_triple

# Where this study caches its simulated limit laws. It lives beside the
# other outputs instead of inside a run directory, so a law is
# simulated once and every run of the study reads the same file.
#
# The empirical pipeline keeps its own cache, at
# ``outputs/application/_limit_laws``. The two directories are separate,
# but the values cannot differ: ``load_limit_law`` keys a law on
# ``(k, n_draws, n_grid_points, seed)`` and simulates it from
# ``default_rng(seed + k)``, and both callers use the same
# ``numerics_config`` constants, so the same dimension yields
# bit-identical draws on either side. What the split costs is the
# simulation being run twice for a dimension both studies use, not a
# divergence in critical values.
#
# This path is resolved from ``paths_cfg.OUTPUT_DIR`` at import time and
# is relative to the working directory. It is not derived from the
# ``output_dir`` argument threaded through ``run_study``: the cache is
# shared across runs instead of owned by one, so directing a run at
# another output root still reads and writes the laws here.
LIMIT_LAW_CACHE = Path(paths_cfg.OUTPUT_DIR) / "limit_laws"

# Per-process memo of the loaded laws. Each worker loads a law once.
_LAW_MEMO: dict[int, np.ndarray] = {}


def limit_law(k: int) -> np.ndarray:
    """Return the directly simulated Nyblom limit law of one dimension.

    This is a direct simulation of the Brownian-bridge limit distribution, used
    for critical values and p-values. It is not a bootstrap and is not a
    resampling scheme.

    Parameters
    ----------
    k : int
        Tested dimension.

    Returns
    -------
    numpy.ndarray
        The sorted draw vector.
    """
    k = int(k)
    if k not in _LAW_MEMO:
        _LAW_MEMO[k] = structural_tests.load_limit_law(k, LIMIT_LAW_CACHE)
    return _LAW_MEMO[k]


def pvalue(k: int, statistic: float) -> float:
    """Asymptotic p-value of a statistic against its simulated limit law.

    Parameters
    ----------
    k : int
        Tested dimension.
    statistic : float
        The observed statistic.

    Returns
    -------
    float
        The tail proportion of the simulated draws.
    """
    return structural_tests.limit_law_pvalue(limit_law(k), statistic)


def level_tag(level: float) -> str:
    """Return the two-digit column suffix of a nominal level.

    Parameters
    ----------
    level : float
        The nominal level.

    Returns
    -------
    str
        The suffix used in every rejection column name.
    """
    return f"{int(level * 100):02d}"


def rejection_flags(prefix: str, p_value: float) -> dict:
    """Rejection indicators of one p-value at the three nominal levels.

    Parameters
    ----------
    prefix : str
        Column-name prefix.
    p_value : float
        The p-value.

    Returns
    -------
    dict
        One boolean per nominal level.
    """
    return {
        f"{prefix}_reject_{level_tag(level)}": bool(p_value < level)
        for level in simcfg.NOMINAL_LEVELS
    }


def dataset_rng(cell, replication_id: int, role: str | None = None):
    """Return the generator of one replication's dataset or of a sub-stream.

    The key is the cell's dataset identity, not the cell identity, so
    two paired cells draw the same data at the same replication index. A
    sub-stream role separates draws that must be independent of the
    data, such as the bootstrap draws or a perturbed starting value, so
    adding one does not disturb the dataset.

    Parameters
    ----------
    cell : Cell
        The cell.
    replication_id : int
        Zero-based replication index.
    role : str or None, optional
        Sub-stream name.

    Returns
    -------
    numpy.random.Generator
        The generator.
    """
    parts = list(cell.dataset_key) + [replication_id]
    if role is not None:
        parts.append(role)
    return seeds.cell_rng(*parts)


def fit_and_test(tensor: np.ndarray) -> dict:
    """Fit the static MAR once and compute every quantity from that one fit.

    Competing statistics within a replication must consume the same dataset and
    the same fit; routing them all through this function is what enforces it,
    and passing the chart quantities on means the score loop runs once.

    Parameters
    ----------
    tensor : numpy.ndarray
        The simulated process, shape ``(T, m, n)``.

    Returns
    -------
    dict
        Keys ``A``, ``B``, ``quantities``, ``battery``, ``telemetry`` and
        ``canonical``.
    """
    A, B, fit_telemetry = estimators.static_fit(tensor, return_telemetry=True)
    quantities = structural_tests.chart_score_quantities(tensor, A, B)
    battery = structural_tests.nyblom_battery(tensor, A, B, quantities=quantities)
    telemetry = estimators.numerical_telemetry(
        tensor, A, B, fit_telemetry, quantities=quantities
    )

    return {
        "A": A,
        "B": B,
        "quantities": quantities,
        "battery": battery,
        "telemetry": telemetry,
        "canonical": canonical_row(A, B),
    }


def canonical_row(A: np.ndarray, B: np.ndarray, prefix: str = "") -> dict:
    """The fitted canonical triple, stored on the record.

    The records already carry every scalar needed, but only scalars: a
    quantity nobody thought of at design time (a different loss, a
    different normalisation, a canonical distance not already named)
    would otherwise cost a full rerun to obtain. The triple is the
    sufficient summary of the fit, so storing it turns any function of
    the estimate into a later calculation instead of a later run.

    It does not make functions of the data or of the score recomputable.
    Those require rebuilding the dataset from its stored spawn key
    instead.

    Parameters
    ----------
    A, B : numpy.ndarray
        The fitted operator pair, on any bilinear scale.
    prefix : str, default ""
        Column-name prefix, for a record that stores more than one fit.

    Returns
    -------
    dict
        The two shapes flattened in column-major order and the amplitude.
    """
    A_bar, B_breve, gamma = canonical_triple(A, B)
    return {
        f"{prefix}fit_A_bar_vec": ";".join(
            f"{v:.17g}" for v in A_bar.flatten(order="F")
        ),
        f"{prefix}fit_B_breve_vec": ";".join(
            f"{v:.17g}" for v in B_breve.flatten(order="F")
        ),
        f"{prefix}fit_gamma": float(gamma),
    }


def battery_row(battery, prefix: str = "") -> dict:
    """Flatten a Nyblom battery into statistic, p-value and rejection columns.

    The three block p-values are Holm adjusted over the three tests. The
    global test is the formal stability decision and is a separate
    decision, not a member of that family.

    Parameters
    ----------
    battery : NyblomBattery
        The computed statistics.
    prefix : str, default ""
        Optional column prefix.

    Returns
    -------
    dict
        The flattened row.
    """
    row: dict = {}
    p_global = pvalue(battery.k, battery.nyblom)

    row[f"{prefix}nyblom_global"] = float(battery.nyblom)
    row[f"{prefix}p_global"] = p_global
    row[f"{prefix}k"] = int(battery.k)
    row.update(rejection_flags(f"{prefix}global", p_global))

    block_p = []
    for name, k_block, value in zip(
        battery.block_names, battery.block_k, battery.block_nyblom
    ):
        p_block = pvalue(k_block, value)
        block_p.append(p_block)
        row[f"{prefix}nyblom_{name}"] = float(value)
        row[f"{prefix}p_{name}"] = p_block
        row[f"{prefix}k_{name}"] = int(k_block)
        row.update(rejection_flags(f"{prefix}raw_{name}", p_block))

    holm = structural_tests.holm_adjusted_pvalues(block_p)
    for name, adjusted in zip(battery.block_names, holm):
        row[f"{prefix}p_holm_{name}"] = float(adjusted)
        row.update(rejection_flags(f"{prefix}holm_{name}", float(adjusted)))

    for level in simcfg.NOMINAL_LEVELS:
        row[f"{prefix}holm_any_reject_{level_tag(level)}"] = bool(
            np.any(holm < level)
        )

    row[f"{prefix}T"] = int(battery.T)
    return row


def gated_attribution_row(row: dict, intended_block: str | None) -> dict:
    """Add the conditional attribution events at every nominal level.

    The global rejection is the formal decision. The Holm block results
    are follow-up attribution measures and count only after it. The
    binary global event is called ``global_reject``, not ``G``, because
    the methodology already uses that letter for the population signal
    matrix.

    With no intended block (a stable-parameter cell), the correct and
    false attribution events are not applicable and are stored as
    missing instead of as False.

    Parameters
    ----------
    row : dict
        A row already carrying the rejection columns of :func:`battery_row`.
    intended_block : str or None
        Which canonical block was planted, if any.

    Returns
    -------
    dict
        The event columns, one set per nominal level.
    """
    events: dict = {}
    for level in simcfg.NOMINAL_LEVELS:
        tag = level_tag(level)
        holm = {
            block: bool(row[f"holm_{name}_reject_{tag}"])
            for block, name in simcfg.BLOCK_TO_STATISTIC.items()
        }
        computed = metrics.attribution_events(
            row[f"global_reject_{tag}"], holm, intended_block
        )
        events.update({f"{key}_{tag}": value for key, value in computed.items()})
    return events


def recovery_row(fit: dict, truth, T: int) -> dict:
    """Canonical recovery losses of one stable-DGP replication.

    Parameters
    ----------
    fit : dict
        The :func:`fit_and_test` result.
    truth : CanonicalTruth
        The population the sample was simulated at.
    T : int
        Retained sample length.

    Returns
    -------
    dict
        The four losses and their sample-scaled counterparts.
    """
    return metrics.recovery_losses(
        fit["A"],
        fit["B"],
        truth.A_bar,
        truth.B_breve,
        truth.gamma,
        simcfg.score_observations(T),
    )

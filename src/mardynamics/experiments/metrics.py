"""Recovery losses, conditional attribution events and Monte Carlo summaries.

Recovery is measured in the canonical coordinates of the shared
methodology and nowhere else. Both the estimate and the population pass
through :func:`mardynamics.core.mar_canonical.canonical_triple`, and
the estimate is then aligned with the population on the joint sign, so
the anchor is the population's, not whichever entry of the estimate
happens to be largest. That distinction is not cosmetic: an anchor that
depends on the data can flip between two accurate estimates of the same
population and report an error of order two where there is none.

The conditional attribution events live here too, so the definition of
a correct, a false and an unflagged rejection exists in one place and
every table reads the same one.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from mardynamics.core.mar_canonical import canonical_triple
from mardynamics.experiments.design import align_joint_sign

_EPS = 1e-12


# =========================================================
# (a) Canonical recovery under stability
# =========================================================
def recovery_losses(
    A_hat: np.ndarray,
    B_hat: np.ndarray,
    A_bar_population: np.ndarray,
    B_breve_population: np.ndarray,
    gamma_population: float,
    N: int,
) -> dict:
    """Canonical recovery losses of one estimate, raw and sample-scaled.

    The four losses: the relative squared transition error, the two
    squared shape errors and the squared log-magnitude error. Each is reported
    both raw and multiplied by the number of score observations, which is the
    descriptive form in which the expected rate can be read: a loss whose mean
    falls at the inverse of the sample size has a scaled mean that is flat in
    the sample size.

    Parameters
    ----------
    A_hat, B_hat : numpy.ndarray
        The estimated operator pair, on any bilinear scale.
    A_bar_population, B_breve_population : numpy.ndarray
        Canonical shapes of the population.
    gamma_population : float
        Amplitude of the population.
    N : int
        Number of score observations behind the estimate.

    Returns
    -------
    dict
        The four losses, the four scaled losses and the realised sign alignment.
    """
    A_bar, B_breve, gamma = canonical_triple(A_hat, B_hat)
    A_bar_aligned, B_breve_aligned = align_joint_sign(A_bar, B_breve, A_bar_population)
    flipped = bool(not np.array_equal(A_bar, A_bar_aligned))

    Phi_hat = np.kron(gamma * B_breve_aligned, A_bar_aligned)
    Phi_population = np.kron(
        gamma_population * B_breve_population, A_bar_population
    )

    L_Phi = float(
        np.linalg.norm(Phi_hat - Phi_population, ord="fro") ** 2
        / max(np.linalg.norm(Phi_population, ord="fro") ** 2, _EPS)
    )
    L_A = float(np.linalg.norm(A_bar_aligned - A_bar_population, ord="fro") ** 2)
    L_B = float(
        np.linalg.norm(B_breve_aligned - B_breve_population, ord="fro") ** 2
    )
    L_kappa = float((np.log(gamma) - np.log(gamma_population)) ** 2)

    N = int(N)
    return {
        "L_Phi": L_Phi,
        "L_A": L_A,
        "L_B": L_B,
        "L_kappa": L_kappa,
        "N_L_Phi": N * L_Phi,
        "N_L_A": N * L_A,
        "N_L_B": N * L_B,
        "N_L_kappa": N * L_kappa,
        "sign_flipped_to_align": flipped,
    }


# =========================================================
# (b) Conditional attribution events
# =========================================================
def attribution_events(
    global_reject: bool, holm_reject: dict, intended_block: str | None
) -> dict:
    """The conditional attribution events of one replication.

    Three indicators are defined. ``global_reject`` is the formal
    stability decision. ``correct_block`` is whether the intended block
    rejects after Holm adjustment. ``false_block`` is whether at least
    one unchanged block rejects after Holm adjustment. The block results
    are follow-up attribution measures and count only after a global
    rejection, which is what the joint events below encode.

    The binary global event is named ``global_reject``, not ``G``: the
    methodology already uses that letter for the population signal matrix,
    and the two are unrelated.

    Parameters
    ----------
    global_reject : bool
        Whether the global statistic rejects.
    holm_reject : dict
        Holm-adjusted block rejection indicator per canonical block.
    intended_block : str or None
        Which block was planted, or None when no block moved. With no
        intended block the correct-attribution events are undefined and
        are returned as missing, not as False.

    Returns
    -------
    dict
        The events and their joint combinations. Undefined events are ``None``.
    """
    global_reject = bool(global_reject)

    if intended_block is None:
        return {
            "global_reject": global_reject,
            "correct_block": None,
            "false_block": None,
            "global_and_correct": None,
            "global_and_false": None,
            "global_and_correct_only": None,
            "global_and_correct_and_false": None,
            "global_and_false_only": None,
            "global_and_none_flagged": None,
        }

    correct = bool(holm_reject[intended_block])
    false = bool(
        any(value for block, value in holm_reject.items() if block != intended_block)
    )

    return {
        "global_reject": global_reject,
        "correct_block": correct,
        "false_block": false,
        "global_and_correct": global_reject and correct,
        "global_and_false": global_reject and false,
        "global_and_correct_only": global_reject and correct and not false,
        "global_and_correct_and_false": global_reject and correct and false,
        "global_and_false_only": global_reject and (not correct) and false,
        "global_and_none_flagged": global_reject and (not correct) and (not false),
    }


# =========================================================
# (c) Monte Carlo summaries
# =========================================================
def rejection_summary(indicators: np.ndarray, level: float) -> dict:
    """Unconditional rate with its Monte Carlo standard error and interval.

    Parameters
    ----------
    indicators : numpy.ndarray
        Boolean indicators of the successful replications.
    level : float
        The nominal level the indicators were formed at.

    Returns
    -------
    dict
        Keys ``level``, ``n_replications``, ``rejection_rate``, ``mcse``,
        ``ci_low`` and ``ci_high`` (a Clopper-Pearson 95% interval). The count
        is not called ``n``: a table that reports both a rate and a panel
        dimension would otherwise collide on that name.
    """
    indicators = np.asarray(indicators, dtype=bool)
    n = int(indicators.size)
    successes = int(indicators.sum())

    if n == 0:
        return {
            "level": level,
            "n_replications": 0,
            "rejection_rate": np.nan,
            "mcse": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    rate = successes / n
    low = (
        0.0
        if successes == 0
        else float(stats.beta.ppf(0.025, successes, n - successes + 1))
    )
    high = (
        1.0
        if successes == n
        else float(stats.beta.ppf(0.975, successes + 1, n - successes))
    )

    return {
        "level": level,
        "n_replications": n,
        "rejection_rate": float(rate),
        "mcse": float(np.sqrt(rate * (1.0 - rate) / n)),
        "ci_low": low,
        "ci_high": high,
    }


def conditional_summary(joint: np.ndarray, gate: np.ndarray) -> dict:
    """A rate conditional on the global rejection, with its observed denominator.

    The denominator is the number of global rejections actually
    observed, and it is stored beside the rate. With no global rejection
    the rate is missing, not zero: nothing was conditioned on, so there
    is nothing to report.

    The standard error is the ordinary binomial one at the observed
    denominator. It is labelled as descriptive Monte Carlo uncertainty,
    not conditional error control, which the column name below states.

    Parameters
    ----------
    joint : numpy.ndarray
        Boolean indicators of the joint event (the numerator).
    gate : numpy.ndarray
        Boolean indicators of the global rejection (the denominator).

    Returns
    -------
    dict
        Keys ``conditional_rate``, ``conditional_denominator`` and
        ``descriptive_mc_se``.
    """
    joint = np.asarray(joint, dtype=bool)
    gate = np.asarray(gate, dtype=bool)
    denominator = int(gate.sum())

    if denominator == 0:
        return {
            "conditional_rate": np.nan,
            "conditional_denominator": 0,
            "descriptive_mc_se": np.nan,
        }

    rate = float(np.sum(joint & gate) / denominator)
    return {
        "conditional_rate": rate,
        "conditional_denominator": denominator,
        "descriptive_mc_se": float(np.sqrt(rate * (1.0 - rate) / denominator)),
    }


def monte_carlo_summary(values: np.ndarray, confidence: float = 0.95) -> dict:
    """Summarise a replication sample with its Monte Carlo uncertainty.

    Parameters
    ----------
    values : numpy.ndarray
        The replication values.
    confidence : float, default 0.95
        Interval level for the mean.

    Returns
    -------
    dict
        Keys ``n_replications``, ``mean``, ``median``, ``sd``, ``mcse``,
        ``ci_low``, ``ci_high``, ``q05`` and ``q95``.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)

    if n == 0:
        return {
            key: np.nan
            for key in (
                "mean",
                "median",
                "sd",
                "mcse",
                "ci_low",
                "ci_high",
                "q05",
                "q95",
            )
        } | {"n_replications": 0}

    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    mcse = sd / np.sqrt(n) if n > 1 else 0.0
    half = float(stats.norm.ppf(0.5 + confidence / 2.0)) * mcse

    return {
        "n_replications": n,
        "mean": mean,
        "median": float(np.median(values)),
        "sd": sd,
        "mcse": float(mcse),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def rate_contrast(first: dict, second: dict) -> dict:
    """Difference and ratio of two unconditional rates, with a difference error.

    The two rates come from independent cells, so the standard error of the
    difference is the root of the sum of their squared Monte Carlo
    standard errors. A ratio whose denominator is zero is missing, not
    infinite.

    Parameters
    ----------
    first, second : dict
        Two :func:`rejection_summary` results; the contrast is first minus
        second.

    Returns
    -------
    dict
        Keys ``rate_difference``, ``rate_ratio`` and ``difference_se``.
    """
    a, b = first.get("rejection_rate", np.nan), second.get("rejection_rate", np.nan)
    se_a, se_b = first.get("mcse", np.nan), second.get("mcse", np.nan)

    ratio = np.nan
    if np.isfinite(a) and np.isfinite(b) and b > 0.0:
        ratio = float(a / b)

    return {
        "rate_difference": float(a - b) if np.isfinite(a) and np.isfinite(b) else np.nan,
        "rate_ratio": ratio,
        "difference_se": (
            float(np.sqrt(se_a**2 + se_b**2))
            if np.isfinite(se_a) and np.isfinite(se_b)
            else np.nan
        ),
    }

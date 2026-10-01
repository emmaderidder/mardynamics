"""Forward simulators for the designed data-generating processes.

Every simulator here rolls one recursion forward and returns the
retained sample. They share three rules, enforced in one place instead
of per experiment:

* the burn-in is 200 observations and runs entirely under the initial
  parameter and the initial innovation scale, so a designed change (of
  the parameter or of the innovation variance) starts only inside the
  retained sample.
* the recursion starts from the zero matrix, the unconditional mean of
  the process, whose influence the burn-in exists to remove.
* the whole innovation stream, burn-in included, is drawn from the
  replication's one generator before the recursion begins, so the data
  of a replication depends on its seed and not on the order in which a
  caller consumes it.

Nothing here constructs a population, a direction or a path. Those come
from :mod:`mardynamics.experiments.design`. Nothing here fits or tests
anything.
"""

from __future__ import annotations

import numpy as np

import mardynamics.experiments.sim_config as simcfg
from mardynamics.experiments.design import CanonicalTruth, ParameterPath


def _roll_forward(
    operators: list[tuple[np.ndarray, np.ndarray]],
    innovations: np.ndarray,
    burn_in: int,
    m: int,
    n: int,
) -> np.ndarray:
    """Iterate the bilinear recursion and return the retained sample.

    Parameters
    ----------
    operators : list of tuple of numpy.ndarray
        One operator pair per retained observation. The burn-in always
        uses the first pair.
    innovations : numpy.ndarray
        Innovation stream of shape ``(burn_in + T, m, n)``.
    burn_in : int
        Number of leading observations to discard.
    m, n : int
        Panel dimensions.

    Returns
    -------
    numpy.ndarray
        The retained process, shape ``(T, m, n)``.
    """
    T = len(operators)
    A_0, B_0 = operators[0]

    X = np.zeros((m, n))
    for i in range(burn_in):
        X = A_0 @ X @ B_0.T + innovations[i]

    retained = np.empty((T, m, n))
    for j in range(T):
        A_j, B_j = operators[j]
        X = A_j @ X @ B_j.T + innovations[burn_in + j]
        retained[j] = X

    return retained


def _full_scale(sigma_scale: np.ndarray | None, burn_in: int) -> np.ndarray | None:
    """Prepend the burn-in, which always runs at scale one.

    Parameters
    ----------
    sigma_scale : numpy.ndarray or None
        Per-retained-observation innovation scale, or None.
    burn_in : int
        Burn-in length.

    Returns
    -------
    numpy.ndarray or None
        The scale over the whole stream, or None when no scale is applied.
    """
    if sigma_scale is None:
        return None
    return np.concatenate([np.ones(burn_in), np.asarray(sigma_scale, dtype=float)])


def simulate_static_tensor(
    truth: CanonicalTruth,
    T: int,
    innovation,
    rng: np.random.Generator,
    burn_in: int = simcfg.BURN_IN,
    sigma_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate a constant-parameter MAR(1) at the population.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    T : int
        Number of retained observations.
    innovation : object
        An innovation generator from :mod:`mardynamics.experiments.innovations`.
    rng : numpy.random.Generator
        Source of the innovation stream.
    burn_in : int, optional
        Burn-in length.
    sigma_scale : numpy.ndarray or None, optional
        Per-retained-observation innovation scale, used by the variance-break
        designs.

    Returns
    -------
    numpy.ndarray
        The retained process, shape ``(T, m, n)``.
    """
    m, n = truth.m, truth.n
    innovations = innovation.draw(
        burn_in + T, m, n, rng, sigma_scale=_full_scale(sigma_scale, burn_in)
    )
    return _roll_forward([truth.operators()] * T, innovations, burn_in, m, n)


def simulate_path_tensor(
    path: ParameterPath,
    innovation,
    rng: np.random.Generator,
    burn_in: int = simcfg.BURN_IN,
    sigma_scale: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate a MAR(1) whose canonical parameter follows a designed path.

    Parameters
    ----------
    path : ParameterPath
        The designed canonical path; its first entry is the initial parameter
        and governs the whole burn-in.
    innovation : object
        An innovation generator.
    rng : numpy.random.Generator
        Source of the innovation stream.
    burn_in : int, optional
        Burn-in length.
    sigma_scale : numpy.ndarray or None, optional
        Per-retained-observation innovation scale. Passing one is how the
        innovation-variance sensitivity combines a parameter change and a
        variance change on the same sample.

    Returns
    -------
    numpy.ndarray
        The retained process, shape ``(T, m, n)``.
    """
    T = path.A_bar.shape[0]
    m = path.A_bar.shape[1]
    n = path.B_breve.shape[1]

    innovations = innovation.draw(
        burn_in + T, m, n, rng, sigma_scale=_full_scale(sigma_scale, burn_in)
    )
    operators = [path.operators_at(j) for j in range(T)]
    return _roll_forward(operators, innovations, burn_in, m, n)


def simulate_unrestricted_tensor(
    Phi_path: np.ndarray,
    m: int,
    n: int,
    innovation,
    rng: np.random.Generator,
    burn_in: int = simcfg.BURN_IN,
) -> np.ndarray:
    """Simulate a vectorised VAR(1) whose transition matrix follows a path.

    The non-separable departure of the matrix-restriction branch leaves
    the Kronecker model, so the recursion runs on the vectorised process
    directly.

    Parameters
    ----------
    Phi_path : numpy.ndarray
        Transition matrices of shape ``(T, mn, mn)``; the first governs the
        burn-in.
    m, n : int
        Panel dimensions.
    innovation : object
        An innovation generator.
    rng : numpy.random.Generator
        Source of the innovation stream.
    burn_in : int, optional
        Burn-in length.

    Returns
    -------
    numpy.ndarray
        The retained process, shape ``(T, m, n)``.
    """
    T = Phi_path.shape[0]
    innovations = innovation.draw(burn_in + T, m, n, rng)

    x = np.zeros(m * n)
    for i in range(burn_in):
        x = Phi_path[0] @ x + innovations[i].flatten(order="F")

    retained = np.empty((T, m, n))
    for j in range(T):
        x = Phi_path[j] @ x + innovations[burn_in + j].flatten(order="F")
        retained[j] = x.reshape(m, n, order="F")

    return retained

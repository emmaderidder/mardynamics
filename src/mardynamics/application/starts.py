"""The eight fresh MS-MAR starting values.

The construction is fixed:

- start 0 is the deterministic projection and residual-split construction.
- starts 1 to 7 are independent tangent perturbations of Frobenius norm
  ``0.10`` for each canonical shape through the sphere exponential map, an
  ``N(0, 0.10^2)`` perturbation of ``kappa`` and an iid Bernoulli(0.5)
  initial regime split.

All eight are rebuilt at every origin with no warm start from the
previous one. Each start is seeded from the full logical identity of its
fit through :func:`mardynamics.application.run_infrastructure.rng`.

The starts are only starting values. The canonical map, the rank
projection, the stability rule, the EM recursion and the GEM acceptance
all remain in the shared estimation code; this module composes their
inputs. The one piece of mathematics written here is the geodesic
exponential map of a unit sphere.
"""

from __future__ import annotations

import numpy as np

import mardynamics.application.emp_config as emp_config
import mardynamics.application.run_infrastructure as runinfra
import mardynamics.core.mar_tensor as mar_tensor
import mardynamics.estimation.regime_em as regime_em
from mardynamics.core.mar_canonical import canonical_triple


def sphere_exponential(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Move along a great circle from a unit vector in a tangent direction.

    Parameters
    ----------
    base : numpy.ndarray
        Unit-norm base point, flattened.
    tangent : numpy.ndarray
        Tangent vector at ``base``, orthogonal to it. Its norm is the geodesic
        distance travelled.

    Returns
    -------
    numpy.ndarray
        The unit-norm image of the exponential map.
    """
    norm = float(np.linalg.norm(tangent))
    if norm <= 0.0:
        return base.copy()
    return np.cos(norm) * base + np.sin(norm) * (tangent / norm)


def _perturb_shape(base: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Move one canonical shape 0.10 along a random tangent direction.

    Parameters
    ----------
    base : numpy.ndarray
        The unit-norm shape matrix.
    generator : numpy.random.Generator
        Source of the tangent direction.

    Returns
    -------
    numpy.ndarray
        The perturbed shape, of the same size and unit Frobenius norm.
    """
    flat = base.flatten(order="F")
    tangent = generator.standard_normal(flat.size)
    tangent -= (tangent @ flat) * flat
    tangent *= emp_config.START_PERTURBATION / np.linalg.norm(tangent)
    return sphere_exponential(flat, tangent).reshape(base.shape, order="F")


def _prepared_initialisation(tensor: np.ndarray, static_pair) -> dict:
    """Assemble the input the shared residual-split initialiser expects.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(T, m, n)``.
    static_pair : tuple of numpy.ndarray
        The full-sample static fit of the same data.

    Returns
    -------
    dict
        The ``prepared_init`` mapping of
        :func:`mardynamics.estimation.regime_em.initialise_regimes_from_static_mar`.
    """
    X_prev, Y = mar_tensor.split_mar_lagged_tensor(tensor)
    A_static, B_static = static_pair
    return {
        "mar_tensor": tensor,
        "X_prev": X_prev,
        "Y": Y,
        "A_static": A_static,
        "B_static": B_static,
        "resid_norm": regime_em.static_residual_norms(X_prev, Y, A_static, B_static),
    }


def _deterministic_start(tensor: np.ndarray, static_pair) -> dict:
    """Build start 0, the deterministic projection and residual split.

    ``tau_init`` is zero: it is the scale of the random perturbation the
    shared initialiser adds to the residual split, and a non-zero value
    would make this start stochastic.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process.
    static_pair : tuple of numpy.ndarray
        The static fit of the same data.

    Returns
    -------
    dict
        The start, in the shared estimator's format.
    """
    prepared = _prepared_initialisation(tensor, static_pair)
    _, A_list, B_list, sigma2_list, _ = regime_em.initialise_regimes_from_static_mar(
        prepared_init=prepared,
        n_regimes=emp_config.N_REGIMES,
        n_als_iter=emp_config.EM_ALS_ITER_PER_MSTEP,
        tau_init=0.0,
        random_state=0,
        init_id=0,
        enforce_stability=emp_config.ENFORCE_STABILITY,
        max_radius=emp_config.MAX_RADIUS,
    )
    transition = regime_em.initialise_transition_matrix(emp_config.N_REGIMES, stay_prob=0.90)
    return {
        "A_list": [np.asarray(a, dtype=float) for a in A_list],
        "B_list": [np.asarray(b, dtype=float) for b in B_list],
        "sigma2_list": [float(s) for s in sigma2_list],
        "transition_matrix": np.asarray(transition, dtype=float),
        "initial_probs": np.full(emp_config.N_REGIMES, 1.0 / emp_config.N_REGIMES),
        "init_info": {
            "method": "deterministic_projection_residual_split",
            "init_id": 0,
        },
    }


def _perturbed_start(
    tensor: np.ndarray, static_pair, fit_label: str, init_id: int, rank: int | None
) -> dict | None:
    """Build one of starts 1 to 7.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process.
    static_pair : tuple of numpy.ndarray
        The static fit of the same data.
    fit_label : str
        The full logical identity of the fit.
    init_id : int
        Start index, between one and seven.
    rank : int or None
        Common rank imposed on the operators, or None.

    Returns
    -------
    dict or None
        The start, or None when the Bernoulli split leaves a regime empty.
    """
    X_prev, Y = mar_tensor.split_mar_lagged_tensor(tensor)
    n_obs = len(X_prev)

    label_rng = runinfra.rng("ms_start", fit_label, init_id, "labels")
    labels = (
        label_rng.random(n_obs) < emp_config.START_LABEL_PROBABILITY
    ).astype(int)
    if len(np.unique(labels)) < emp_config.N_REGIMES:
        return None

    A_bar, B_breve, gamma = canonical_triple(*static_pair)
    kappa = float(np.log(gamma))

    A_list, B_list, sigma2_list = [], [], []
    for regime in range(emp_config.N_REGIMES):
        row_rng = runinfra.rng("ms_start", fit_label, init_id, regime, "row")
        column_rng = runinfra.rng("ms_start", fit_label, init_id, regime, "column")
        kappa_rng = runinfra.rng("ms_start", fit_label, init_id, regime, "amplitude")

        row = _perturb_shape(A_bar, row_rng)
        column = _perturb_shape(B_breve, column_rng)
        kappa_j = kappa + kappa_rng.normal(0.0, emp_config.START_PERTURBATION)

        A_j, B_breve_j, gamma_j = canonical_triple(row, np.exp(kappa_j) * column)
        B_j = gamma_j * B_breve_j
        if rank is not None:
            from mardynamics.core.mar_canonical import project_low_rank

            A_j = project_low_rank(A_j, rank)
            B_j = project_low_rank(B_j, rank)

        A_list.append(A_j)
        B_list.append(B_j)

        mask = labels == regime
        residual = Y[mask] - np.array(
            [A_j @ X_prev[t] @ B_j.T for t in np.flatnonzero(mask)]
        )
        sigma2_list.append(float(max((residual**2).mean(), emp_config.PROBABILITY_FLOOR)))

    counts = np.zeros((emp_config.N_REGIMES, emp_config.N_REGIMES))
    for first, second in zip(labels[:-1], labels[1:]):
        counts[first, second] += 1.0
    if np.any(counts.sum(axis=1) == 0):
        return None
    row_stochastic = counts / counts.sum(axis=1, keepdims=True)

    return {
        "A_list": A_list,
        "B_list": B_list,
        "sigma2_list": sigma2_list,
        # The shared estimator's convention is column stochastic.
        "transition_matrix": row_stochastic.T,
        "initial_probs": np.full(emp_config.N_REGIMES, 1.0 / emp_config.N_REGIMES),
        "init_info": {
            "method": "tangent_perturbation_with_bernoulli_split",
            "init_id": init_id,
            "perturbation": emp_config.START_PERTURBATION,
        },
    }


def build_starts(
    tensor: np.ndarray, fit_label: str, rank: int | None = None
) -> list[dict]:
    """Build the eight fresh starts of one MS-MAR fit.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(T, m, n)``.
    fit_label : str
        The full logical identity of the fit. Two different origins do
        not share a start.
    rank : int or None, optional
        Common rank imposed on the operators.

    Returns
    -------
    list of dict
        The starts, in index order. A start whose Bernoulli split leaves a
        regime empty is dropped, not replaced, so the list can be shorter
        than eight. A fit with no admissible start is a hard failure.
    """
    import mardynamics.application.estimators as estimators

    static_pair = (
        estimators.als_fit(tensor, rank=rank)
        if rank is not None
        else estimators.static_fit(tensor)
    )

    starts = [_deterministic_start(tensor, static_pair)]
    for init_id in range(1, emp_config.N_STARTS):
        start = _perturbed_start(tensor, static_pair, fit_label, init_id, rank)
        if start is not None:
            starts.append(start)

    # All eight starts are required at every fit, and a fit with no
    # admissible start is a hard failure. A Bernoulli split that empties a
    # regime is not silently tolerated: the fit fails the same way instead
    # of continuing on seven. The split itself is untouched: no rejection
    # sampling and no reseeding.
    if len(starts) < emp_config.N_STARTS:
        from mardynamics.application.forecasting import HardFailure

        raise HardFailure(
            "insufficient_admissible_starts",
            f"{len(starts)} of {emp_config.N_STARTS} starts admissible for {fit_label}",
        )
    return starts

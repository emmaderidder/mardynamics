"""Canonical coordinates and rank projection for the MAR coefficient pair.

The bilinear form ``A X B'`` is only identified up to a scale swap between ``A``
and ``B``, and up to a joint sign flip. This module fixes that indeterminacy: it
normalises the pair, maps it to the canonical triple ``(A_bar, B_breve, gamma)``
where ``gamma`` carries the common amplitude, aligns a whole path of pairs to one
convention, and reports the identified per-period increments. It also holds the
machinery of the transition matrix ``Phi``: the unrestricted least-squares fit
:func:`fit_var1_lse`, the Van Loan-Pitsianis rearrangement, the Kronecker
projection estimator, and the truncated-SVD projection used by the low-rank
estimators. Pure NumPy/pandas; nothing here reads configuration.
"""

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass

import warnings

import numpy as np
import pandas as pd

# Coordinate of ``vec(A_bar)`` (column-major) whose sign fixes the
# canonical representative. Assumption A4 designates one fixed index.
# This is it. Picking "the first entry that happens to be non-zero"
# instead would make the index data-dependent, which is not the rule
# the paper states.
SIGN_COORDINATE = 0

# Below this magnitude the sign coordinate is too close to zero for
# Assumption A4 to be credible and :func:`canonical_triple` warns: the
# sign of the canonical representative would then be decided by
# sampling noise.
SIGN_COORDINATE_WARN = 1e-3


def normalise_ab(
    A: np.ndarray, B: np.ndarray, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Rescale ``(A, B)`` so that ``A`` has unit Frobenius norm.

    Resolves the bilinear scale indeterminacy ``A X Bᵀ = (A/s) X (B·s)ᵀ`` by moving
    the scale from ``A`` into ``B``; the implied forecast is unchanged. If ``A`` is
    numerically zero (norm below ``eps``) the inputs are returned unchanged.

    Parameters
    ----------
    A, B : numpy.ndarray
        Left (``m x m``) and right (``n x n``) coefficient matrices.
    eps : float, default 1e-12
        Threshold below which ``A`` is treated as zero.

    Returns
    -------
    tuple of numpy.ndarray
        The rescaled ``(A, B)``.
    """
    scale = np.linalg.norm(A, ord="fro")

    if scale < eps:
        return A, B

    return A / scale, B * scale


def canonical_triple(
    A: np.ndarray,
    B: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map ``(A, B)`` to the canonical triple ``(A_bar, B_breve, gamma)``.

    Resolves the bilinear scale and sign indeterminacies in three steps: (i) the
    Frobenius scale is moved from ``A`` into ``B`` (via :func:`normalise_ab`, so
    ``||A_bar||_F = 1``); (ii) the joint sign is fixed by requiring coordinate
    :data:`SIGN_COORDINATE` of ``vec(A_bar)`` (column-major) to be positive,
    flipping *both* matrices when it is not (the invariant object is
    ``B_bar (x) A_bar``); (iii) the remaining amplitude ``gamma = ||B_bar||_F``
    is split off, leaving the unit-norm shape ``B_breve = B_bar / gamma``.

    Parameters
    ----------
    A, B : numpy.ndarray
        Left (``m x m``) and right (``n x n``) coefficient matrices.
    eps : float, default 1e-12
        Threshold below which a norm is treated as numerically zero.

    Returns
    -------
    A_bar : numpy.ndarray
        Unit-Frobenius, sign-fixed left operator.
    B_breve : numpy.ndarray
        Unit-Frobenius right operator shape.
    gamma : float
        The common amplitude ``||B_bar||_F``.

    Raises
    ------
    ValueError
        If ``||A||_F < eps`` or the implied ``gamma < eps``.
    """
    if np.linalg.norm(A, ord="fro") < eps:
        raise ValueError("canonical_triple: ||A||_F is numerically zero.")

    A_bar, B_bar = normalise_ab(A, B, eps=eps)

    sign_entry = float(A_bar.flatten(order="F")[SIGN_COORDINATE])
    if abs(sign_entry) < SIGN_COORDINATE_WARN:
        warnings.warn(
            f"canonical_triple: |vec(A_bar)[{SIGN_COORDINATE}]| = "
            f"{abs(sign_entry):.2e} is below {SIGN_COORDINATE_WARN:g}; "
            "Assumption A4 requires this coordinate to be bounded away from "
            "zero, and the sign of the canonical representative is otherwise "
            "decided by sampling noise."
        )
    if sign_entry < 0.0:
        A_bar = -A_bar
        B_bar = -B_bar

    gamma = float(np.linalg.norm(B_bar, ord="fro"))
    if gamma < eps:
        raise ValueError("canonical_triple: gamma = ||B_bar||_F is numerically zero.")

    return A_bar, B_bar / gamma, gamma


def align_canonical_path(
    A_bar_list: list[np.ndarray],
    B_bar_list: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sequentially sign-align a time-ordered path of canonical pairs.

    For ``t >= 1`` the sign is the minimiser of
    ``||s A_t - A_{t-1}||_F^2 + ||s B_t - B_{t-1}||_F^2`` over ``s`` in
    ``{-1, +1}``, jointly over both shapes. For unit-norm shapes that
    objective equals ``const - 2 s (<A_t, A_{t-1}> + <B_t, B_{t-1}>)``, so
    the minimiser is the sign of the sum of the two inner products.
    Deciding on the ``A`` term alone is a different rule, and the two
    disagree precisely when the two inner products have opposite signs.
    Flipping the sign of ``A_bar`` requires flipping ``B_bar`` too,
    because the invariant object is ``B_bar (x) A_bar``. The per-period
    canonical sign convention can jump when the sign coordinate crosses
    zero, so a smoothness reading of the path needs this sequential
    alignment.

    Parameters
    ----------
    A_bar_list, B_bar_list : list of numpy.ndarray
        Time-ordered canonical pairs (e.g. from :func:`canonical_triple`).

    Returns
    -------
    tuple of list of numpy.ndarray
        The aligned ``(A_bar_list, B_bar_list)``.
    """
    if len(A_bar_list) != len(B_bar_list):
        raise ValueError("align_canonical_path: A and B paths differ in length.")

    A_aligned: list[np.ndarray] = []
    B_aligned: list[np.ndarray] = []

    for t, (A_t, B_t) in enumerate(zip(A_bar_list, B_bar_list)):
        A_t = A_t.copy()
        B_t = B_t.copy()

        if t >= 1:
            inner = float(
                np.sum(A_t.flatten(order="F") * A_aligned[-1].flatten(order="F"))
                + np.sum(B_t.flatten(order="F") * B_aligned[-1].flatten(order="F"))
            )
            if inner < 0:
                A_t = -A_t
                B_t = -B_t

        A_aligned.append(A_t)
        B_aligned.append(B_t)

    return A_aligned, B_aligned


def align_pair_signs(
    A_list: Sequence[np.ndarray],
    B_list: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Put a sequence of operator pairs on one common sign convention.

    Each pair ``(A, B)`` is identified only up to the joint flip
    ``(A, B) -> (-A, -B)``, because the invariant object is ``B (x) A``.
    Any quantity that combines several pairs is undefined until they are
    put on one sign: a probability-weighted average of regime operators,
    for instance, cancels instead of interpolating wherever two regimes
    carry opposite signs, and anything computed from that average
    measures the convention, not the dynamics.

    The rule is the pairwise one of the paper: element ``i`` is flipped when it
    lies further from element ``i - 1`` than its negative does, measured jointly
    on the two shapes. The inner products are taken on the unit-norm shapes, so
    the decision does not depend on how the common scale happens to be split
    between the two operators, while the flip is applied to the inputs as given,
    so the caller keeps the operators on their original scale.

    Parameters
    ----------
    A_list, B_list : sequence of numpy.ndarray
        Operator pairs, in the order they should be chained (time order for a
        path, regime order for a regime-switching fit).

    Returns
    -------
    tuple of list of numpy.ndarray
        The sign-aligned ``(A_list, B_list)``.

    Raises
    ------
    ValueError
        If the two sequences differ in length.
    """
    if len(A_list) != len(B_list):
        raise ValueError("align_pair_signs: A and B sequences differ in length.")

    def _shape(M: np.ndarray) -> np.ndarray:
        """Unit-Frobenius direction of ``M`` as a flat column-major vector."""
        v = np.asarray(M, dtype=float).flatten(order="F")
        norm = float(np.linalg.norm(v))
        return v if norm == 0.0 else v / norm

    A_aligned: list[np.ndarray] = []
    B_aligned: list[np.ndarray] = []

    for i, (A_i, B_i) in enumerate(zip(A_list, B_list)):
        A_i = np.asarray(A_i, dtype=float).copy()
        B_i = np.asarray(B_i, dtype=float).copy()

        if i >= 1:
            inner = float(
                np.dot(_shape(A_i), _shape(A_aligned[-1]))
                + np.dot(_shape(B_i), _shape(B_aligned[-1]))
            )
            if inner < 0:
                A_i = -A_i
                B_i = -B_i

        A_aligned.append(A_i)
        B_aligned.append(B_i)

    return A_aligned, B_aligned


def canonical_path_increments(
    A_bar_list: list[np.ndarray],
    B_breve_list: list[np.ndarray],
    gamma_list: list[float],
    index: Sequence | None = None,
) -> pd.DataFrame:
    """Per-period increments of an aligned canonical coefficient path.

    The pairs are first sequentially sign-aligned with
    :func:`align_canonical_path` (the ``B_breve`` sign flips jointly with the
    ``A_bar`` sign, since the pair flips jointly), after which the identified
    revision sizes are ``d_A = ||A_bar_t - A_bar_{t-1}||_F``,
    ``d_B = ||B_breve_t - B_breve_{t-1}||_F`` and the log-amplitude change
    ``d_gamma = |log(gamma_t) - log(gamma_{t-1})|``.

    Parameters
    ----------
    A_bar_list, B_breve_list : list of numpy.ndarray
        Time-ordered canonical shapes (from :func:`canonical_triple`).
    gamma_list : list of float
        The matching amplitude path.
    index : sequence or None, optional
        Index for the returned DataFrame (e.g. the path's time index); a
        ``RangeIndex`` is used if omitted.

    Returns
    -------
    pandas.DataFrame
        Columns ``d_A``, ``d_B`` and ``d_gamma``; the first row is ``NaN``.
    """
    A_aligned, B_aligned = align_canonical_path(A_bar_list, B_breve_list)
    n_obs = len(A_aligned)

    d_A = [np.nan]
    d_B = [np.nan]
    d_gamma = [np.nan]

    for t in range(1, n_obs):
        d_A.append(float(np.linalg.norm(A_aligned[t] - A_aligned[t - 1], ord="fro")))
        d_B.append(float(np.linalg.norm(B_aligned[t] - B_aligned[t - 1], ord="fro")))
        d_gamma.append(float(abs(np.log(gamma_list[t]) - np.log(gamma_list[t - 1]))))

    return pd.DataFrame(
        {"d_A": d_A, "d_B": d_B, "d_gamma": d_gamma},
        index=index if index is not None else pd.RangeIndex(n_obs),
    )


def project_low_rank(M: np.ndarray, rank: int, eps: float = 1e-12) -> np.ndarray:
    """Project a matrix onto its best rank-``rank`` approximation via truncated SVD.

    Parameters
    ----------
    M : numpy.ndarray
        Matrix to truncate.
    rank : int
        Target rank; clipped to ``[1, min(M.shape)]``.
    eps : float, default 1e-12
        If the largest singular value is below this, ``M`` is returned unchanged.

    Returns
    -------
    numpy.ndarray
        The rank-``rank`` approximation of ``M``.
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    rank = max(1, min(int(rank), len(s)))

    if s[0] < eps:
        return M.copy()

    U_r = U[:, :rank]
    s_r = s[:rank]
    Vt_r = Vt[:rank, :]

    return U_r @ np.diag(s_r) @ Vt_r


def rearrange_kronecker(Phi: np.ndarray, m: int, n: int) -> np.ndarray:
    """Van Loan-Pitsianis rearrangement of an ``mn x mn`` matrix.

    Maps ``Phi`` to the ``m^2 x n^2`` matrix ``R(Phi)`` characterised by
    ``R(B (x) A) = vec(A) vec(B)^T`` for column-major ``vec``. The Kronecker
    restriction therefore holds precisely when ``R(Phi)`` has rank one, which is
    what the specification check of
    :func:`mardynamics.core.structural_tests.kronecker_rank_one_check` examines, and the
    leading singular triplet of ``R(Phi_hat)`` is the projection estimator.

    Parameters
    ----------
    Phi : numpy.ndarray
        Transition matrix of shape ``(m * n, m * n)``.
    m, n : int
        Row and column operator dimensions.

    Returns
    -------
    numpy.ndarray
        The rearranged matrix, shape ``(m * m, n * n)``.
    """
    Phi = np.asarray(Phi, dtype=float)
    return Phi.reshape(n, m, n, m).transpose(3, 1, 2, 0).reshape(m * m, n * n)


def _var1_design(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-major regressor and response matrices of the vectorised VAR(1).

    Parameters
    ----------
    tensor : numpy.ndarray
        Process of shape ``(L, m, n)`` with at least two time points.

    Returns
    -------
    Z, Y : numpy.ndarray
        The ``(mn) x (L - 1)`` lagged and contemporaneous matrices, whose columns
        are ``vec(X_{t-1})`` and ``vec(X_t)``.
    """
    arr = np.asarray(tensor, dtype=float)
    V = np.column_stack([arr[t].flatten(order="F") for t in range(arr.shape[0])])
    return V[:, :-1], V[:, 1:]


def fit_var1_lse(tensor: np.ndarray) -> np.ndarray:
    """Least-squares fit of the unrestricted vectorised VAR(1) transition matrix.

    Stacks each ``m x n`` observation into a length-``mn`` column-major vector and
    fits ``vec(X_t) = Phi vec(X_{t-1}) + e_t`` by ordinary least squares. This is
    the single unrestricted estimate of ``Phi`` in the project: the Kronecker
    projection estimator below starts from it, the Kronecker rank-one check
    rearranges it, and the vectorised Nyblom statistic of Proposition 2 is built
    on its scores.

    The solve is the minimum-norm least-squares solution, so a sample
    with fewer than ``mn`` transitions returns a value instead of
    raising. This is an estimator, not a statistic: the covariance
    matrices that enter a test are not regularised this way.

    Parameters
    ----------
    tensor : numpy.ndarray
        Process of shape ``(L, m, n)`` with at least two time points.

    Returns
    -------
    numpy.ndarray
        The ``(mn) x (mn)`` transition matrix ``Phi_hat``.
    """
    Z, Y = _var1_design(tensor)
    phi_t, *_ = np.linalg.lstsq(Z.T, Y.T, rcond=None)
    return phi_t.T


@dataclass(frozen=True)
class ProjectionEstimate:
    """Result of the Kronecker projection estimator.

    Attributes
    ----------
    A, B : numpy.ndarray
        The projected pair, rescaled so that ``||A||_F = 1``.
    regressor_rank : int
        Numerical rank of the lagged design ``[vec(X_1), ..., vec(X_{L-1})]``.
    minimum_norm : bool
        ``True`` when that rank is below ``mn``, so ``Phi_hat`` is the
        minimum-norm solution of an underdetermined system, not the
        unique least-squares solution.
    """

    A: np.ndarray
    B: np.ndarray
    regressor_rank: int
    minimum_norm: bool


def projection_estimator(tensor: np.ndarray, eps: float = 1e-12) -> ProjectionEstimate:
    """Kronecker projection estimator of the MAR(1) operators.

    Fits the unrestricted ``vec(X_t) = Phi vec(X_{t-1}) + e_t`` by least
    squares (:func:`fit_var1_lse`), rearranges the estimate and reads
    off its leading singular triplet, which is the closest Kronecker
    pair to ``Phi_hat`` in the Frobenius norm. It is used as the
    starting value of the alternating scheme. Because the population
    coefficient matrix is of Kronecker form under the model, the
    projection is itself consistent, so the alternating scheme starts
    in a neighbourhood of the truth instead of at an arbitrary point.

    A segment with fewer than ``mn`` transitions leaves ``Phi`` underdetermined
    and the fit returns the minimum-norm solution. That is admissible for a
    starting value, but it is not the same object as the least-squares estimate,
    so the rank of the design is checked and the branch that was taken is
    recorded on the returned :class:`ProjectionEstimate`.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``, at least two time points.
    eps : float, default 1e-12
        Floor on the leading singular value before taking its square root.

    Returns
    -------
    ProjectionEstimate
        The pair together with the design rank and the branch indicator.
    """
    arr = np.asarray(tensor, dtype=float)
    _, m, n = arr.shape

    Z, _ = _var1_design(arr)
    rank = int(np.linalg.matrix_rank(Z))
    Phi = fit_var1_lse(arr)

    U, s, Vt = np.linalg.svd(rearrange_kronecker(Phi, m, n), full_matrices=False)
    scale = np.sqrt(max(float(s[0]), eps))

    A = (scale * U[:, 0]).reshape(m, m, order="F")
    B = (scale * Vt[0, :]).reshape(n, n, order="F")
    A, B = normalise_ab(A, B)

    return ProjectionEstimate(A=A, B=B, regressor_rank=rank, minimum_norm=rank < m * n)

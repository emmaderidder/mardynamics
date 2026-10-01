"""Constancy test for the MAR(1) operators on the canonical chart.

The methodology has one test of parameter constancy: the global Nyblom-Hansen
fluctuation statistic

    N_T = (T - 1)^{-2} sum_t S_t' Omega_hat^{-1} S_t,

computed from a single full-sample fit, together with its three influence-block
statistics for the row shape ``A_bar``, the column shape ``B_breve`` and the
amplitude ``gamma``. There is no second test, no candidate break date and no
trimming grid: the statistic integrates over the whole path, and its limit law
is pivotal in the tested dimension ``k`` alone.

This module holds that machinery: the regressor Jacobians of the bilinear map
``vec(A X B^T)`` with the commutation-matrix convention, the orthonormal chart
basis of the canonical parameterisation ``(A_bar, B_breve, gamma)``, the chart
scores with their Gauss-Newton Hessian ``H_hat``, outer-product score
covariance ``Omega_hat`` and sandwich ``V_hat = H_hat^{-1} Omega_hat H_hat^{-1}``,
the global and block statistics, the Holm adjustment of the block p-values, the
simulated Brownian-bridge limit law, the fixed-regressor wild bootstrap, and the
three specification checks.

Every function here is pure NumPy: it takes arrays and explicit arguments,
reads only :mod:`mardynamics.numerics_config`, knows nothing about pandas,
labels, dates or paths, and takes a ``numpy.random.Generator`` wherever it
draws.

No covariance that enters a statistic is ever pseudo-inverted or ridged. A
singular ``H_hat``, ``Omega_hat``, ``V_hat`` block or VAR score covariance
raises :class:`SingularCovarianceError`: it is the feasibility boundary that
Proposition 2 describes, and a number returned there would hide the
effect the paper reports.

All ``vec`` operations in this module are column-major (``order="F"``), without
exception.
"""

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np

import mardynamics.numerics_config as ncfg

# Condition-number threshold above which a solve logs a warning.
_COND_WARN = 1e10


class SingularCovarianceError(np.linalg.LinAlgError):
    """A covariance matrix inside a statistic is singular or not positive definite.

    Raised instead of falling back on a pseudo-inverse or a ridge. Assumption A5
    makes ``H`` and ``Omega`` positive definite, so a statistic is defined only
    where these systems are solvable; where they are not, the failure is the
    result. The outer product of ``T - 1`` scores that sum to zero has rank at
    most ``T - 2``, so the vectorised statistic of Proposition 2 ceases to exist
    below ``T - 2 >= K`` and this exception is what reports that.
    """


# =========================================================
# (a) Regressor Jacobian
# =========================================================
def commutation_matrix(n: int) -> np.ndarray:
    """Build the ``n^2 x n^2`` commutation matrix ``K_nn``.

    Defined by ``K_nn @ vec(B) = vec(B.T)`` for any ``n x n`` matrix ``B``
    (column-major ``vec``): ``K[i + j*n, j + i*n] = 1`` for all ``i, j``.

    Parameters
    ----------
    n : int
        Matrix dimension.

    Returns
    -------
    numpy.ndarray
        The commutation matrix ``K_nn``.
    """
    K = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            K[i + j * n, j + i * n] = 1.0
    return K


def regressor_jacobian(
    A: np.ndarray, B: np.ndarray, X_prev_t: np.ndarray, K_nn: np.ndarray | None = None
) -> np.ndarray:
    """Jacobian ``Z_t`` of ``vec(A X_{t-1} B^T)`` w.r.t. ``[vec(A); vec(B)]``.

    Public because it is the ambient regressor Jacobian of the methodology's
    score, and the population information matrix ``H = 2 J'(Gamma_0 (x) I) J``
    is assembled from it: with the chart basis ``Q`` and the chart Jacobian
    ``J = D Q``, the identity ``E[Z_t' Z_t] = J'(Gamma_0 (x) I) J`` holds, so a
    caller that knows ``Gamma_0`` can build ``H`` without simulating.

    With ``M_t = X_{t-1} B^T`` the ``A`` block is ``J_A = kron(M_t.T, I_m)``
    (``mn x m^2``); with ``N_t = A X_{t-1}`` the ``B`` block is
    ``J_B = kron(I_n, N_t) @ K_nn`` (``mn x n^2``), where ``K_nn`` is the
    commutation matrix. All ``vec`` operations are column-major. The two blocks
    have different widths whenever ``m != n``; nothing here assumes a square
    panel.

    Parameters
    ----------
    A, B : numpy.ndarray
        Current operators (``m x m`` and ``n x n``).
    X_prev_t : numpy.ndarray
        The lagged observation ``X_{t-1}`` (``m x n``).
    K_nn : numpy.ndarray or None, optional
        Precomputed commutation matrix; built once per call if omitted.

    Returns
    -------
    numpy.ndarray
        ``Z_t = hstack([J_A, J_B])`` of shape ``(mn, m^2 + n^2)``.
    """
    m = A.shape[0]
    n = B.shape[0]

    if K_nn is None:
        K_nn = commutation_matrix(n)

    M_t = X_prev_t @ B.T
    J_A = np.kron(M_t.T, np.eye(m))

    N_t = A @ X_prev_t
    J_B = np.kron(np.eye(n), N_t) @ K_nn

    return np.hstack([J_A, J_B])


# =========================================================
# (b) Chart basis
# =========================================================
def mode_slices(m: int, n: int) -> tuple[slice, slice, slice]:
    """Chart-coordinate slices of the ``A``-shape, ``B``-shape and amplitude modes.

    The first ``m^2 - 1`` chart coordinates belong to the shape of mode ``A``,
    the next ``n^2 - 1`` to the shape of mode ``B``, and the final coordinate to
    the amplitude ``gamma``. These are the selection matrices ``R_A``, ``R_B``
    and ``R_gamma`` of the methodology, with dimensions ``k_A = m^2 - 1``,
    ``k_B = n^2 - 1`` and ``k_gamma = 1``.

    Parameters
    ----------
    m, n : int
        Row and column operator dimensions.

    Returns
    -------
    tuple of slice
        ``(MODE_A_SLICE, MODE_B_SLICE, MODE_GAMMA_SLICE)``.
    """
    k = m * m + n * n - 1
    return slice(0, m * m - 1), slice(m * m - 1, k - 1), slice(k - 1, k)


def _unit_complement(v: np.ndarray) -> np.ndarray:
    """Orthonormal basis of the orthogonal complement of a unit vector.

    Built from the left singular vectors ``2..p`` of ``v`` as a column, the
    same construction the chart uses for both operator shapes.

    Parameters
    ----------
    v : numpy.ndarray
        Unit vector of length ``p``.

    Returns
    -------
    numpy.ndarray
        A ``(p, p - 1)`` matrix with orthonormal columns spanning ``v``'s
        orthogonal complement.
    """
    U, _, _ = np.linalg.svd(v.reshape(-1, 1), full_matrices=True)
    return U[:, 1:]


def chart_basis(A_bar: np.ndarray, B_breve: np.ndarray) -> np.ndarray:
    """Orthonormal chart basis at the canonical point ``(A_bar, B_breve)``.

    Block structured ``(m^2 + n^2, k)`` with ``k = m^2 + n^2 - 1``: the first
    ``m^2 - 1`` columns are an orthonormal basis of the orthogonal complement of
    ``vec(A_bar)`` inside ``R^{m^2}`` (the shape directions of ``A``), embedded
    with zeros in the ``B`` block; the next ``n^2 - 1`` columns are an
    orthonormal basis of the orthogonal complement of ``vec(B_breve)`` inside
    ``R^{n^2}`` (the shape directions of ``B``), embedded with zeros in the
    ``A`` block; the final column is ``vec(B_breve)`` itself in the ``B`` block,
    the amplitude direction, since ``B_bar = gamma * B_breve`` moves along it
    when only ``gamma`` moves. This is the orthonormal representation of the
    shared statistics' chart. It differs from the chart of the methodology by
    a fixed diagonal rescaling, to which the global and block quadratic forms
    are invariant.

    Parameters
    ----------
    A_bar : numpy.ndarray
        Canonical (unit-Frobenius) left operator (``m x m``).
    B_breve : numpy.ndarray
        Canonical (unit-Frobenius) right operator shape (``n x n``).

    Returns
    -------
    numpy.ndarray
        The chart basis ``Q`` of shape ``(m^2 + n^2, k)``.
    """
    m = A_bar.shape[0]
    n = B_breve.shape[0]

    k = m * m + n * n - 1
    b_dir = B_breve.flatten(order="F")

    Q = np.zeros((m * m + n * n, k))
    Q[: m * m, : m * m - 1] = _unit_complement(A_bar.flatten(order="F"))
    Q[m * m :, m * m - 1 : k - 1] = _unit_complement(b_dir)
    Q[m * m :, k - 1] = b_dir

    return Q


# =========================================================
# (c) Scores, Hessian, variance
# =========================================================
def _checked_solve(M: np.ndarray, rhs: np.ndarray, label: str = "matrix") -> np.ndarray:
    """Solve ``M x = rhs`` for a symmetric positive definite ``M``, or raise.

    No regularisation is applied anywhere. The check is the Cholesky
    factorisation: it succeeds precisely when ``M`` is numerically positive
    definite, which is what Assumption A5 asserts for every covariance that
    enters a statistic, and it fails where the statistic is not defined.

    Parameters
    ----------
    M : numpy.ndarray
        Symmetric positive definite system matrix.
    rhs : numpy.ndarray
        Right-hand side (vector or matrix).
    label : str, optional
        Name used in the warning and in the exception message.

    Returns
    -------
    numpy.ndarray
        The solution ``x``.

    Raises
    ------
    SingularCovarianceError
        If ``M`` is not finite, or not numerically positive definite.
    """
    M = np.asarray(M, dtype=float)

    if not np.all(np.isfinite(M)):
        raise SingularCovarianceError(f"structural_tests: {label} is not finite.")

    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as exc:
        raise SingularCovarianceError(
            f"structural_tests: {label} is singular or not positive definite "
            f"(shape {M.shape}); no pseudo-inverse or ridge is applied here."
        ) from exc

    cond = float(np.linalg.cond(M))
    if cond > _COND_WARN:
        warnings.warn(
            f"structural_tests: condition number of {label} is {cond:.2e} (> 1e10)."
        )

    return np.linalg.solve(L.T, np.linalg.solve(L, rhs))


def chart_score_quantities(tensor: np.ndarray, A: np.ndarray, B: np.ndarray) -> dict:
    """Canonical pair, chart scores, Hessian and variance for a fitted MAR(1).

    The input pair is first mapped to its canonical triple; the scores are
    evaluated at ``(A_bar, B_bar)`` with ``B_bar = gamma * B_breve``, so every
    returned quantity is invariant to the bilinear scale of the input pair.
    Per observation pair ``(X_{t-1}, X_t)``: residual
    ``E_t = X_t - A_bar X_{t-1} B_bar^T``, ambient score
    ``s_t = -2 Z_t^T vec(E_t)`` and chart score ``g_t = Q^T s_t``.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``; the ``T = L - 1`` one-step
        pairs are the score sample.
    A, B : numpy.ndarray
        Fitted operators (any bilinear scale).

    Returns
    -------
    dict
        Keys ``A_bar``, ``B_breve``, ``B_bar``, ``gamma``, ``Q``, ``g``
        (``T x k`` chart scores), ``g_influence`` (``H_hat^{-1} g_t``, the
        process the mode-specific statistics decompose), ``S`` (the cumulative
        score sums), ``H_hat``, ``Omega_hat``, ``V_hat``, ``T``, ``m``, ``n``,
        ``k`` and the mode slices ``mode_A``/``mode_B``/``mode_gamma``.
    """
    from mardynamics.core.mar_canonical import canonical_triple

    arr = np.asarray(tensor, dtype=float)
    L, m, n = arr.shape
    T = L - 1

    A_bar, B_breve, gamma = canonical_triple(
        np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    )
    B_bar = gamma * B_breve

    Q = chart_basis(A_bar, B_breve)
    k = Q.shape[1]
    K_nn = commutation_matrix(n)

    g = np.empty((T, k))
    ZtZ_sum = np.zeros((m * m + n * n, m * m + n * n))

    for t in range(T):
        X_prev_t = arr[t]
        E_t = arr[t + 1] - A_bar @ X_prev_t @ B_bar.T

        Z_t = regressor_jacobian(A_bar, B_bar, X_prev_t, K_nn=K_nn)
        s_t = -2.0 * Z_t.T @ E_t.flatten(order="F")

        g[t] = Q.T @ s_t
        ZtZ_sum += Z_t.T @ Z_t

    # Gauss-Newton Hessian: the omitted second-derivative term is linear in E_t
    # and has zero mean at the truth, so this curvature estimate is consistent.
    H_hat = Q.T @ ((2.0 / T) * ZtZ_sum) @ Q

    Omega_hat = (g.T @ g) / T

    H_inv_Omega = _checked_solve(H_hat, Omega_hat, label="H_hat")
    V_hat = _checked_solve(H_hat, H_inv_Omega.T, label="H_hat").T

    # Influence function H_hat^{-1} g_t. The mode-specific statistics are built
    # on this process and not on the raw score: under a local alternative the
    # raw score has drift -H (theta_t - theta_0), so block j of it moves whenever
    # H is non-zero off the diagonal for some other moving block, while the
    # influence function has drift -(theta_t - theta_0) and its jth block follows
    # parameter block j alone.
    g_influence = _checked_solve(H_hat, g.T, label="H_hat").T

    mode_A, mode_B, mode_gamma = mode_slices(m, n)

    return {
        "A_bar": A_bar,
        "B_breve": B_breve,
        "B_bar": B_bar,
        "gamma": gamma,
        "Q": Q,
        "g": g,
        "g_influence": g_influence,
        "S": np.cumsum(g, axis=0),
        "H_hat": H_hat,
        "Omega_hat": Omega_hat,
        "V_hat": V_hat,
        "T": T,
        "m": m,
        "n": n,
        "k": k,
        "mode_A": mode_A,
        "mode_B": mode_B,
        "mode_gamma": mode_gamma,
    }


# =========================================================
# (d) The Nyblom-Hansen statistic and its influence blocks
# =========================================================
def nyblom_statistic(g: np.ndarray, omega: np.ndarray | None = None) -> float:
    """Nyblom-Hansen statistic from a partial-sum process.

    With ``S_t`` the cumulative sums of ``g`` (``T x k``),
    ``N_T = T^{-2} sum_t S_t' omega^{-1} S_t``. The normalisation is by the
    squared number of transitions, which is what makes the statistic converge to
    the integrated squared Brownian bridge instead of diverging.

    This is the only constancy statistic in the project. The global test passes
    the chart score with its own outer product; a block test passes the
    influence block ``R_j H_hat^{-1} g_t`` with the matching block of ``V_hat``;
    the vectorised comparison of Proposition 2 passes the VAR scores with their
    outer product. All three go through this one function.

    Parameters
    ----------
    g : numpy.ndarray
        Per-observation process of shape ``(T, k)``.
    omega : numpy.ndarray or None, optional
        Standardising matrix (``k x k``). Defaults to the outer-product
        estimator ``g'g / T``, which is the right choice for a raw score and the
        wrong one for an influence block.

    Returns
    -------
    float
        The statistic ``N_T``.

    Raises
    ------
    SingularCovarianceError
        If the standardising matrix is not positive definite.
    """
    g = np.asarray(g, dtype=float)
    T = g.shape[0]

    S = np.cumsum(g, axis=0)
    if omega is None:
        omega = (g.T @ g) / T
    omega = np.asarray(omega, dtype=float).reshape(g.shape[1], g.shape[1])

    solved = _checked_solve(omega, S.T, label="Omega_hat")  # (k, T)
    return float(np.sum(S.T * solved) / T**2)


# Names of the three influence blocks, in chart order.
BLOCK_NAMES = ("A", "B", "gamma")


@dataclass(frozen=True)
class NyblomBattery:
    """The global Nyblom statistic and its three influence-block statistics.

    Attributes
    ----------
    T : int
        Number of one-step transitions the statistics are built on.
    m, n : int
        Panel dimensions.
    k : int
        Global chart dimension ``m^2 + n^2 - 1``.
    nyblom : float
        The global statistic of the methodology's fluctuation test.
    block_names : tuple of str
        ``("A", "B", "gamma")``.
    block_k : tuple of int
        Block dimensions ``(m^2 - 1, n^2 - 1, 1)``.
    block_nyblom : tuple of float
        The three block statistics. Their raw values are not comparable with one
        another, because their dimensions differ.
    score_cusum : numpy.ndarray
        The descriptive path ``||T^{-1/2} S_t||``, length ``T``. It carries no
        decision: the shape of a realised path is not a dating or a
        classification statement.
    cond_H, cond_Omega : float
        Condition numbers of the two matrices the statistic inverts. They are
        reported, not acted on: near the feasibility boundary of
        Proposition 2 the outer-product covariance becomes ill-conditioned long
        before it becomes singular, and a statistic computed there is worth
        less than its value suggests.
    """

    T: int
    m: int
    n: int
    k: int
    nyblom: float
    block_names: tuple[str, ...]
    block_k: tuple[int, ...]
    block_nyblom: tuple[float, ...]
    score_cusum: np.ndarray
    cond_H: float
    cond_Omega: float


def nyblom_battery(
    tensor: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    quantities: dict | None = None,
) -> NyblomBattery:
    """Global and block Nyblom statistics for one tensor and one fitted pair.

    Pure: it takes an ``(L, m, n)`` array and the fitted operators, and returns
    numbers. The fit itself is the caller's business, and so is anything to do
    with dates, labels or files.

    The block statistics decompose the influence function ``R_j H_hat^{-1} S_t``
    standardised by the matching block ``R_j V_hat R_j'``, not the raw score
    block and not a block of ``Omega_hat``. Applied to the full selection the
    construction reproduces the global statistic, because
    ``S' H^{-1} (H^{-1} Omega H^{-1})^{-1} H^{-1} S = S' Omega^{-1} S``.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``.
    A, B : numpy.ndarray
        Fitted operators (any bilinear scale).
    quantities : dict or None, optional
        A previously computed :func:`chart_score_quantities` result for the same
        tensor and pair. A caller that also needs the influence process or the
        covariance blocks passes them here so that the score loop runs once;
        omitting it recomputes them.

    Returns
    -------
    NyblomBattery
        The statistics, their dimensions and the descriptive CUSUM path.
    """
    q = chart_score_quantities(tensor, A, B) if quantities is None else quantities

    g_influence = q["g_influence"]
    V_hat = q["V_hat"]
    T = q["T"]

    block_stats = []
    block_k = []
    for name in BLOCK_NAMES:
        sl = q[f"mode_{name}"]
        block_stats.append(nyblom_statistic(g_influence[:, sl], omega=V_hat[sl, sl]))
        block_k.append(sl.stop - sl.start)

    return NyblomBattery(
        T=T,
        m=q["m"],
        n=q["n"],
        k=q["k"],
        nyblom=nyblom_statistic(q["g"]),
        block_names=BLOCK_NAMES,
        block_k=tuple(block_k),
        block_nyblom=tuple(block_stats),
        score_cusum=np.linalg.norm(q["S"] / np.sqrt(T), axis=1),
        cond_H=float(np.linalg.cond(q["H_hat"])),
        cond_Omega=float(np.linalg.cond(q["Omega_hat"])),
    )


# =========================================================
# (e) The unrestricted vectorised comparison (Proposition 2)
# =========================================================
def var_dimension(m: int, n: int) -> int:
    """Parameter count ``K = (mn)^2`` of the unrestricted VAR(1) on ``vec(X_t)``.

    Parameters
    ----------
    m, n : int
        Panel dimensions.

    Returns
    -------
    int
        The unrestricted dimension ``K``, against which the identified MAR
        dimension is ``k = m^2 + n^2 - 1``.
    """
    return int((m * n) ** 2)


def var_scores(tensor: np.ndarray) -> np.ndarray:
    """Per-observation scores of the unrestricted vectorised VAR(1) loss.

    With ``z_t = vec(X_{t-1})``, ``e_t = vec(X_t) - Phi_hat z_t`` and the loss
    ``||vec(X_t) - Phi vec(X_{t-1})||^2``, the score with respect to
    ``varphi = vec(Phi)`` is ``u_t = -2 (z_t (x) e_t)``. The transition matrix is
    the one shared estimate :func:`mardynamics.core.mar_canonical.fit_var1_lse`, the same
    object the Kronecker rank-one check rearranges.

    Parameters
    ----------
    tensor : numpy.ndarray
        Process of shape ``(L, m, n)``.

    Returns
    -------
    numpy.ndarray
        Score sample of shape ``(L - 1, (mn)^2)``.
    """
    from mardynamics.core.mar_canonical import _var1_design, fit_var1_lse

    arr = np.asarray(tensor, dtype=float)
    Z, Y = _var1_design(arr)
    resid = Y - fit_var1_lse(arr) @ Z

    return np.array(
        [
            -2.0 * np.outer(resid[:, t], Z[:, t]).flatten(order="F")
            for t in range(Z.shape[1])
        ]
    )


def var_nyblom_statistic(tensor: np.ndarray) -> float:
    """Nyblom-Hansen statistic of the unrestricted vectorised VAR(1).

    The comparison object of Proposition 2. It is the same functional as
    the matrix test (it calls :func:`nyblom_statistic`) applied to the
    unrestricted scores and their outer-product covariance, so the two
    statistics differ only in the model whose scores they read and in the
    dimension of the resulting limit law (``K`` against ``k``).

    Parameters
    ----------
    tensor : numpy.ndarray
        Process of shape ``(L, m, n)``.

    Returns
    -------
    float
        The unrestricted statistic.

    Raises
    ------
    SingularCovarianceError
        Where the outer-product covariance of ``K`` parameters from ``L - 1``
        scores is not positive definite, which is the feasibility boundary
        ``T - 2 >= K`` of the methodology.
    """
    return nyblom_statistic(var_scores(tensor))


# =========================================================
# (f) Multiplicity
# =========================================================
def holm_adjusted_pvalues(pvalues: Sequence[float] | np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values.

    The methodology adjusts the three block p-values with the Holm procedure
    when they are treated as formal decisions, which controls the unconditional
    familywise error rate at ``alpha_0``. Holm is uniformly at least as powerful
    as Bonferroni and needs no independence assumption, which matters here
    because the three block paths are projections of one Brownian bridge and are
    therefore dependent.

    Sorting the ``M`` p-values ascending, the adjusted value at rank ``i`` is
    the running maximum of ``(M - i) * p_(i)``, capped at one, restored to the
    input order. Comparing the result with ``alpha_0`` is the Holm decision.

    Parameters
    ----------
    pvalues : sequence of float
        The raw p-values of one family.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values, in the order of the input.

    Raises
    ------
    ValueError
        If the input is empty or contains a value outside ``[0, 1]``.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    M = p.size

    if M == 0:
        raise ValueError("holm_adjusted_pvalues: no p-values given.")
    if np.any(~np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("holm_adjusted_pvalues: p-values must lie in [0, 1].")

    order = np.argsort(p, kind="stable")
    scaled = (M - np.arange(M)) * p[order]

    adjusted = np.empty(M)
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


# =========================================================
# (g) Limit law of the Nyblom-Hansen statistic
# =========================================================
def simulate_nyblom_limit_law(
    k: int, n_draws: int, n_grid_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Simulate the limit law ``int_0^1 ||BB_k(r)||^2 dr`` of the statistic.

    Standard ``k``-dimensional Brownian bridges are simulated on
    ``n_grid_points`` equispaced points (Gaussian Wiener increments, bridge
    ``BB(r) = W(r) - r W(1)``) and the integral is the mean of ``||BB(r)||^2``
    over those points. Draws are processed in batches (with the ``k`` dimensions
    chunked) to bound memory.

    The law is pivotal: it depends on the tested dimension ``k`` and on nothing
    else about the data. There is no trimming region and no grid of candidate
    break dates in it, because the functional integrates over the whole path.

    Parameters
    ----------
    k : int
        Tested dimension.
    n_draws : int
        Number of independent bridges.
    n_grid_points : int
        Number of integration points per bridge.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    numpy.ndarray
        The ``n_draws`` simulated values, sorted ascending.
    """
    k = int(k)
    n_draws = int(n_draws)
    n_grid_points = int(n_grid_points)

    steps = np.arange(1, n_grid_points + 1) / n_grid_points
    draws = np.empty(n_draws)

    batch_size = 1000
    k_chunk = max(1, min(k, 8))

    start = 0
    while start < n_draws:
        stop = min(start + batch_size, n_draws)
        n_batch = stop - start

        sq_norm = np.zeros((n_batch, n_grid_points))
        for c0 in range(0, k, k_chunk):
            c1 = min(c0 + k_chunk, k)
            dW = rng.normal(
                0.0,
                np.sqrt(1.0 / n_grid_points),
                size=(n_batch, n_grid_points, c1 - c0),
            )
            W = np.cumsum(dW, axis=1)
            BB = W - steps[None, :, None] * W[:, -1:, :]
            sq_norm += np.sum(BB**2, axis=2)

        draws[start:stop] = np.mean(sq_norm, axis=1)
        start = stop

    draws.sort()
    return draws


def limit_law_path(
    cache_dir: "str | Path", k: int, n_draws: int, n_grid_points: int, seed: int
) -> Path:
    """Cache path of one simulated limit law.

    The key is ``(k, n_draws, n_grid_points, seed)`` and nothing else: those four
    numbers are the complete identity of the simulated law, because the law
    itself depends only on ``k`` and the remaining three fix the simulation that
    approximates it.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        Directory holding the cached laws.
    k : int
        Tested dimension.
    n_draws, n_grid_points, seed : int
        Simulation controls.

    Returns
    -------
    pathlib.Path
        The ``.npz`` cache path.
    """
    return Path(cache_dir) / (
        f"nyblom_limit_law_k{int(k)}_d{int(n_draws)}"
        f"_g{int(n_grid_points)}_s{int(seed)}.npz"
    )


def load_limit_law(
    k: int,
    cache_dir: "str | Path",
    n_draws: int = ncfg.LIMITLAW_N_DRAWS,
    n_grid_points: int = ncfg.LIMITLAW_N_GRID,
    seed: int = ncfg.LIMITLAW_SEED,
) -> np.ndarray:
    """Load, simulating and caching on first use, the limit-law draws for ``k``.

    Parameters
    ----------
    k : int
        Tested dimension.
    cache_dir : str or pathlib.Path
        Directory the ``.npz`` caches live in; created if missing.
    n_draws, n_grid_points, seed : int, optional
        Simulation controls (defaults from :mod:`mardynamics.numerics_config`). They are
        part of the cache key, so asking for a different simulation size reads a
        different file instead of silently reusing the old one.

    Returns
    -------
    numpy.ndarray
        The sorted draw vector.
    """
    path = limit_law_path(cache_dir, k, n_draws, n_grid_points, seed)

    if path.exists():
        with np.load(path) as data:
            return data["nyblom"]

    # The seed is offset by k so that the laws of different dimensions are not
    # driven by the same Wiener increments.
    draws = simulate_nyblom_limit_law(
        k, n_draws, n_grid_points, np.random.default_rng(int(seed) + int(k))
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, nyblom=draws)

    return draws


def limit_law_pvalue(draws: np.ndarray, value: float) -> float:
    """Tail-proportion p-value of ``value`` against sorted limit-law draws.

    Parameters
    ----------
    draws : numpy.ndarray
        Sorted draw vector from :func:`load_limit_law`.
    value : float
        Observed statistic.

    Returns
    -------
    float
        ``P(draw >= value)`` under the simulated limit law.
    """
    return float(np.mean(np.asarray(draws) >= value))


def limit_law_columns(draws: np.ndarray, value: float) -> dict:
    """Critical values and asymptotic p-value of a statistic under its limit law.

    Parameters
    ----------
    draws : numpy.ndarray
        Sorted draw vector from :func:`load_limit_law`.
    value : float
        Observed statistic.

    Returns
    -------
    dict
        Keys ``crit_90``, ``crit_95``, ``crit_99`` and ``p_asymptotic``.
    """
    return {
        "crit_90": float(np.quantile(draws, 0.90)),
        "crit_95": float(np.quantile(draws, 0.95)),
        "crit_99": float(np.quantile(draws, 0.99)),
        "p_asymptotic": limit_law_pvalue(draws, value),
    }


# =========================================================
# (h) Fixed-regressor wild bootstrap
# =========================================================
def _decoupled_chart_scores(
    X_prev: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Chart scores for a decoupled ``(X_{t-1}, Y_t)`` pair sequence.

    Same construction as :func:`chart_score_quantities`, but the response is
    not the lead of the regressor (fixed-regressor bootstrap design). Only the
    score array is returned; the bootstrapped statistic standardises it by its
    own outer product.

    Parameters
    ----------
    X_prev, Y : numpy.ndarray
        Regressor and response tensors of shape ``(T, m, n)``.
    A, B : numpy.ndarray
        Fitted operators (any bilinear scale).

    Returns
    -------
    numpy.ndarray
        The ``T x k`` chart scores.
    """
    from mardynamics.core.mar_canonical import canonical_triple

    T, m, n = Y.shape

    A_bar, B_breve, gamma = canonical_triple(
        np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    )
    B_bar = gamma * B_breve

    Q = chart_basis(A_bar, B_breve)
    K_nn = commutation_matrix(n)

    g = np.empty((T, Q.shape[1]))
    for t in range(T):
        E_t = Y[t] - A_bar @ X_prev[t] @ B_bar.T
        Z_t = regressor_jacobian(A_bar, B_bar, X_prev[t], K_nn=K_nn)
        g[t] = Q.T @ (-2.0 * Z_t.T @ E_t.flatten(order="F"))

    return g


def wild_bootstrap_pvalue(
    tensor: np.ndarray,
    nyblom_observed: float,
    fitted_pair: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    n_boot: int = ncfg.N_BOOT,
    enforce_stability: bool = ncfg.ENFORCE_STABILITY,
    max_radius: float = ncfg.STABILITY_MAX_RADIUS,
) -> dict:
    """Fixed-regressor wild bootstrap p-value for the global Nyblom statistic.

    With fitted values ``F_t`` and residuals ``E_hat_t``, each draw builds
    ``X_t^b = F_t + eta_t E_hat_t`` from iid Rademacher scalars ``eta_t`` while
    the regressor side keeps the original ``X_{t-1}``. The model is re-estimated
    on the decoupled pair sequence and the same Nyblom statistic is recomputed.

    This is a finite-sample robustness check reported next to the asymptotic
    test, not a second test and not a replacement for it: no bootstrap-validity
    proof is available for the constrained bilinear estimator. The block
    statistics are not bootstrapped.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``.
    nyblom_observed : float
        The observed global statistic the draws are compared with.
    fitted_pair : tuple of numpy.ndarray
        Full-sample ``(A, B)`` used as the bootstrap data-generating device.
    rng : numpy.random.Generator
        Source of the Rademacher draws.
    n_boot : int, optional
        Number of bootstrap draws (default from :mod:`mardynamics.numerics_config`).
    enforce_stability : bool, optional
        Whether each bootstrap refit is contracted into the stable region.
    max_radius : float, optional
        The stability boundary used when it is.

    Returns
    -------
    dict
        Keys ``p_bootstrap`` (``(1 + #{stat_b >= stat_obs}) / (n_boot + 1)``)
        and ``n_boot``, which always equals the requested number of draws.
    """
    import mardynamics.core.mar_tensor as mt
    from mardynamics.estimation.regime_em import estimate_weighted_mar_parameters

    tensor = np.asarray(tensor, dtype=float)
    X_prev, Y = mt.split_mar_lagged_tensor(tensor)
    T = len(X_prev)

    A_hat, B_hat = fitted_pair
    fitted = np.array([A_hat @ X_prev[t] @ B_hat.T for t in range(T)])
    resid = Y - fitted

    unit_weights = np.ones(T)
    n_used = int(n_boot)
    count = 0

    for _ in range(n_used):
        eta = rng.choice([-1.0, 1.0], size=T)
        Y_boot = fitted + eta[:, None, None] * resid

        A_b, B_b, _ = estimate_weighted_mar_parameters(
            X_prev=X_prev,
            Y=Y_boot,
            weights=unit_weights,
            max_iter=200,
            A_init=A_hat,
            B_init=B_hat,
            tol=1e-6,
            enforce_stability=enforce_stability,
            max_radius=max_radius,
        )

        g_boot = _decoupled_chart_scores(X_prev, Y_boot, A_b, B_b)
        if nyblom_statistic(g_boot) >= nyblom_observed:
            count += 1

    return {"p_bootstrap": (1 + count) / (n_used + 1), "n_boot": n_used}


# =========================================================
# (i) Specification checks
# =========================================================
def _mar_residuals(
    tensor: np.ndarray, fitted_pair: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """One-step residual matrices ``E_hat_t = X_t - A X_{t-1} B'``.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``.
    fitted_pair : tuple of numpy.ndarray
        The fitted operator pair.

    Returns
    -------
    numpy.ndarray
        Residuals of shape ``(L - 1, m, n)``.
    """
    arr = np.asarray(tensor, dtype=float)
    A_hat, B_hat = fitted_pair
    return arr[1:] - np.array(
        [A_hat @ arr[t] @ B_hat.T for t in range(arr.shape[0] - 1)]
    )


def kronecker_singular_share(tensor: np.ndarray) -> float:
    """Squared-singular-value share of the leading component of ``R(Phi_hat)``.

    The statistic ``zeta(Phi_hat)`` of the methodology. The unrestricted
    transition matrix is estimated by :func:`mardynamics.core.mar_canonical.fit_var1_lse` and
    rearranged. The share is one precisely when the rearranged matrix has rank one,
    that is, when the Kronecker restriction holds.

    Parameters
    ----------
    tensor : numpy.ndarray
        Process of shape ``(L, m, n)``.

    Returns
    -------
    float
        Share in ``(0, 1]``, or ``NaN`` for a degenerate panel.
    """
    from mardynamics.core.mar_canonical import fit_var1_lse, rearrange_kronecker

    arr = np.asarray(tensor, dtype=float)
    _, m, n = arr.shape

    s = np.linalg.svd(rearrange_kronecker(fit_var1_lse(arr), m, n), compute_uv=False)
    total = float((s**2).sum())
    return float(s[0] ** 2 / total) if total > 0.0 else float("nan")


def kronecker_rank_one_check(
    tensor: np.ndarray,
    fitted_pair: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
    n_boot: int = ncfg.N_SPEC_BOOT,
    burn_in: int = 50,
    block: int | None = None,
) -> dict:
    """Direct check of the Kronecker restriction on the transition matrix.

    The restriction ``Phi = B (x) A`` holds if and only if the rearranged matrix
    ``R(Phi) = vec(A) vec(B)'`` has rank one, so the singular-value share
    :func:`kronecker_singular_share` is one under the restriction and below one
    otherwise. A residual-based check cannot see this: the least-squares
    first-order conditions already make the residuals orthogonal to the
    regressors in the ``k`` directions the matrix model spans, so the restriction
    has to be examined directly.

    Sampling noise pushes the observed share below one even when the restriction
    holds, so it is compared with a simulated reference instead of with
    one. Each replication rolls the fitted bilinear law forward with resampled
    innovation matrices, so the simulated panels satisfy the restriction by
    construction while retaining the observed innovation distribution.

    The innovations are resampled in moving blocks of consecutive residual
    matrices, of length ``block``, and not one at a time. Assumption A1 leaves
    the conditional covariance free to depend on the past, and an independent
    resample destroys that dependence: it generates panels whose
    innovations are far better behaved than the data's, which inflates the
    simulated share and turns the check into one that rejects almost regardless
    of the truth. The default block length follows the ``round(T^(1/3))``
    rule.

    Estimating ``Phi_hat`` needs only ``T >= mn``, not the ``T - 2 >= K`` that
    the vectorised test requires, so this check remains available at panel sizes
    where that test is not.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``.
    fitted_pair : tuple of numpy.ndarray
        Full-sample ``(A, B)``, used as the simulation device.
    rng : numpy.random.Generator
        Source of the block draws.
    n_boot : int, optional
        Replications (default from :mod:`mardynamics.numerics_config`).
    burn_in : int, default 50
        Leading simulated periods discarded in each replication.
    block : int or None, optional
        Moving-block length; ``round(T ** (1 / 3))`` when omitted.

    Returns
    -------
    dict
        Keys ``share_observed``, ``share_boot_mean``, ``share_boot_q05``,
        ``p_lower`` (the proportion of simulated shares at or below the observed
        one, so a small value is evidence against the restriction), ``n_boot``,
        ``block``, ``T`` and ``K``.
    """
    arr = np.asarray(tensor, dtype=float)
    L, m, n = arr.shape
    A_hat, B_hat = fitted_pair

    share_obs = kronecker_singular_share(arr)
    resid = _mar_residuals(arr, fitted_pair)
    n_resid = resid.shape[0]

    if block is None:
        block = int(round(n_resid ** (1.0 / 3.0)))
    block = int(max(1, min(block, n_resid)))

    total = L + burn_in
    n_blocks = int(np.ceil(total / block))
    shares = np.empty(int(n_boot))

    for b in range(int(n_boot)):
        starts = rng.integers(0, n_resid - block + 1, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:total]
        draws = resid[idx]

        X = np.zeros((m, n))
        panel = np.empty((L, m, n))
        for t in range(total):
            X = A_hat @ X @ B_hat.T + draws[t]
            if t >= burn_in:
                panel[t - burn_in] = X
        shares[b] = kronecker_singular_share(panel)

    return {
        "share_observed": share_obs,
        "share_boot_mean": float(np.mean(shares)),
        "share_boot_q05": float(np.quantile(shares, 0.05)),
        "p_lower": float(np.mean(shares <= share_obs)),
        "n_boot": int(n_boot),
        "block": int(block),
        "T": int(n_resid),
        "K": var_dimension(m, n),
    }


def innovation_variance_statistic(
    tensor: np.ndarray,
    fitted_pair: tuple[np.ndarray, np.ndarray],
    bandwidth: int | None = None,
) -> dict:
    """Fluctuation statistic for the constancy of the total innovation variance.

    Assumption A1 requires the innovations to be strictly stationary, so a
    structural change in their covariance lies outside the theory even when the
    operators are constant: under such a change the partial sums of the scores
    no longer converge to a standard Brownian bridge and the critical values of
    the main test no longer apply. This check applies the same fluctuation
    functional to the scalar series ``||E_hat_t||_F^2``, whose auxiliary
    criterion is the mean, so its score is the demeaned series itself.

    That score is a martingale difference only under conditional homoskedasticity,
    which is not assumed here, so the outer-product estimator is
    replaced by a Bartlett kernel-smoothed long-run variance with bandwidth
    ``floor(4 (T / 100)^{2/9})`` unless one is supplied. A rejection is evidence
    on the constancy of the unconditional innovation variance and not a test of
    Assumption A1 as a whole. Its limit law is the ``k = 1`` law.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``.
    fitted_pair : tuple of numpy.ndarray
        Full-sample ``(A, B)``.
    bandwidth : int or None, optional
        Bartlett bandwidth; the rule above is used when omitted.

    Returns
    -------
    dict
        Keys ``statistic``, ``bandwidth``, ``T`` and ``k`` (always one).
    """
    resid = _mar_residuals(tensor, fitted_pair)
    v = (resid**2).sum(axis=(1, 2))
    u = v - v.mean()
    T = int(u.size)

    if bandwidth is None:
        bandwidth = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    bandwidth = max(0, min(int(bandwidth), T - 1))

    lrv = float(np.dot(u, u) / T)
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1.0)
        lrv += 2.0 * weight * float(np.dot(u[lag:], u[:-lag]) / T)

    if not np.isfinite(lrv) or lrv <= 0.0:
        raise SingularCovarianceError(
            "structural_tests: the long-run variance of ||E_t||_F^2 is not "
            f"positive ({lrv:.3e}); the fluctuation statistic is undefined."
        )

    S = np.cumsum(u)
    return {
        "statistic": float(np.sum(S**2) / (lrv * T**2)),
        "bandwidth": int(bandwidth),
        "T": T,
        "k": 1,
    }

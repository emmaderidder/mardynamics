"""Stability diagnostics for the MAR forecasting models.

A matrix autoregression ``X_t = A X_{t-1} B' + E_t`` is stationary when the
companion operator ``B (x) A`` (Kronecker product, since ``vec(A X B') =
(B (x) A) vec(X)``) has spectral radius below one. This module turns estimated
``(A, B)`` matrices (static, rolling, low-rank or regime-switching) into
that spectral radius, and holds the soft spectral projection that the
estimators use to keep a fitted pair inside the stable region.
"""

from __future__ import annotations

import numpy as np


def spectral_radius_from_matrices(A_matrix: np.ndarray, B_matrix: np.ndarray) -> float:
    """Compute the spectral radius of the MAR companion operator ``B (x) A``.

    Parameters
    ----------
    A_matrix : numpy.ndarray
        Variable-side coefficient matrix ``A``.
    B_matrix : numpy.ndarray
        Country-side coefficient matrix ``B``.

    Returns
    -------
    float
        Largest absolute eigenvalue of ``kron(B, A)``.
    """
    transition_matrix = np.kron(B_matrix, A_matrix)
    eigenvalues = np.linalg.eigvals(transition_matrix)
    return float(np.max(np.abs(eigenvalues)))


def project_to_stable(
    A: np.ndarray,
    B: np.ndarray,
    max_radius: float = 0.99,
    n_iter: int = 5,
    safety: float = 1e-3,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft spectral projection of an ``(A, B)`` pair onto a stable MAR(1).

    The MAR(1) ``X_t = A X_{t-1} Bᵀ + E_t`` is stationary/causal iff the companion
    operator ``B ⊗ A`` has spectral radius below one. Because the eigenvalues of
    ``kron(B, A)`` are the pairwise products of those of ``B`` and ``A``, we have
    ``ρ(kron(B, A)) = ρ(A)·ρ(B)``. Scaling ``A → sA`` and ``B → sB`` therefore
    multiplies the radius by ``s²`` while preserving the *ranks* of ``A`` and ``B``.
    When the system is unstable we pick ``s = sqrt(target / ρ)`` so the new radius
    equals ``target = max_radius·(1 - safety)`` (just inside the boundary), then call
    :func:`normalise_ab` to push the scale back into ``B`` and restore the
    ``‖A‖_F = 1`` identification.

    The projection is *soft*: while ``ρ < max_radius`` the inputs are returned
    untouched, so switching it on does not perturb an already-stable model.

    Parameters
    ----------
    A, B : numpy.ndarray
        MAR coefficient matrices.
    max_radius : float, default 0.99
        Stability threshold; the system is left untouched while ``ρ < max_radius``.
    n_iter : int, default 5
        Maximum rescaling passes. One closed-form pass is exact in theory; the loop
        only re-polishes floating-point residual and breaks once ``ρ < max_radius``.
    safety : float, default 1e-3
        Relative margin below ``max_radius`` to target, so the result is strictly
        (not marginally) inside the stable region.
    eps : float, default 1e-12
        Radii at or below this are treated as numerically zero and left untouched.

    Returns
    -------
    tuple of numpy.ndarray
        The (possibly rescaled) ``(A, B)``; the original objects when no projection
        is needed.
    """
    from mardynamics.core.mar_canonical import normalise_ab

    radius = spectral_radius_from_matrices(A, B)

    # Soft: only intervene on a genuinely unstable (finite, non-degenerate) system.
    if not np.isfinite(radius) or radius < max_radius or radius <= eps:
        return A, B

    target_radius = max_radius * (1.0 - safety)

    A_proj, B_proj = A, B
    for _ in range(max(int(n_iter), 1)):
        radius = spectral_radius_from_matrices(A_proj, B_proj)
        if radius < max_radius or radius <= eps:
            break
        scale = np.sqrt(target_radius / radius)
        A_proj, B_proj = normalise_ab(A_proj * scale, B_proj * scale)

    return A_proj, B_proj

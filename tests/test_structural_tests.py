"""Unit tests for the shared test machinery and the canonical coordinates.

They are deterministic and small: no Monte Carlo size or power run, no bootstrap
beyond three draws. What they check:

1.  Score correctness: the analytic chart score summed over ``t`` matches the
    central-difference numerical gradient of the total loss along each chart
    basis direction, at ``(3, 3)`` and at ``(5, 3)``.
2.  Commutation matrix: ``K_nn @ vec(B) == vec(B.T)`` exactly.
3.  Chart basis: orthonormality and the orbit-direction projection property.
4.  Canonical map: invariance to ``(A, B) -> (sA, B/s)``, at ``(3, 3)`` and
    ``(5, 3)``.
5.  Scale invariance of every statistic under the same reparameterisation.
6.  Block statistics: they use ``V_hat`` blocks of the influence process, not
    blocks of ``Omega_hat``, and the full selection reproduces the global
    statistic.
7.  Holm: reproduces a hand-computed example and dominates Bonferroni.
8.  The VAR path calls the same ``nyblom_statistic`` as the MAR test.
9.  A singular covariance inside a statistic raises instead of returning a
    number.
10. Every statistic runs at ``(5, 3)`` and not only on a square panel.
11. The rolling-window fit uses no observation at or after the forecast date.
12. Limit law: the simulated ``k = 1`` 95% quantile matches the published
    Nyblom (1989) value, and the cache is keyed by the simulation alone.

Run from the project venv with pytest::

    python -m pytest tests/test_structural_tests.py
"""

from __future__ import annotations

import numpy as np
import pytest

import mardynamics.numerics_config as ncfg
from mardynamics.core.mar_canonical import canonical_triple
import mardynamics.core.structural_tests as sts


def _total_loss(tensor: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Total squared-Frobenius one-step loss ``sum_t ||X_t - A X_{t-1} B^T||_F^2``.

    Parameters
    ----------
    tensor : numpy.ndarray
        Observed process of shape ``(L, m, n)``.
    A, B : numpy.ndarray
        Operator pair the loss is evaluated at.

    Returns
    -------
    float
        The total loss over the ``L - 1`` one-step pairs.
    """
    total = 0.0
    for t in range(1, tensor.shape[0]):
        resid = tensor[t] - A @ tensor[t - 1] @ B.T
        total += float(np.sum(resid**2))
    return total


def _stable_pair(m: int, n: int, rng, persistence: float = 0.6):
    """Build a stable operator pair with a given joint spectral radius.

    Local to this module on purpose: the shared numerical layer is tested
    without the simulation package, so a change to the study's design
    cannot silently change what these tests exercise.

    Parameters
    ----------
    m, n : int
        Row and column dimensions.
    rng : numpy.random.Generator
        Source of the dense draws.
    persistence : float, default 0.6
        Target ``rho(A) rho(B)``.

    Returns
    -------
    A, B : numpy.ndarray
        The operator pair.
    """
    A = rng.standard_normal((m, m))
    B = rng.standard_normal((n, n))
    A = A / float(np.max(np.abs(np.linalg.eigvals(A))))
    B = B / float(np.max(np.abs(np.linalg.eigvals(B))))
    return A, persistence * B


def _simulate(A, B, T, sigma, rng, burn_in=50):
    """Iterate ``X_t = A X_{t-1} B' + E_t`` with Gaussian innovations.

    Parameters
    ----------
    A, B : numpy.ndarray
        The operator pair.
    T : int
        Observations to keep after burn-in.
    sigma : float
        Per-entry innovation standard deviation.
    rng : numpy.random.Generator
        Source of the innovations.
    burn_in : int, default 50
        Leading observations to discard.

    Returns
    -------
    numpy.ndarray
        The retained process, shape ``(T, m, n)``.
    """
    m, n = A.shape[0], B.shape[0]
    X = np.zeros((m, n))
    out = np.empty((T, m, n))
    for t in range(burn_in + T):
        X = A @ X @ B.T + rng.normal(0.0, sigma, size=(m, n))
        if t >= burn_in:
            out[t - burn_in] = X
    return out


def _simulated_pair_and_tensor(m: int, n: int, T: int, seed: int, sigma: float = 0.1):
    """A stable canonical pair and a simulated MAR(1) path from it.

    Parameters
    ----------
    m, n : int
        Row and column panel dimensions. They are not required to be equal.
    T : int
        Number of observations to keep after burn-in.
    seed : int
        RNG seed for the pair construction and the innovation path.
    sigma : float, default 0.1
        Innovation standard deviation.

    Returns
    -------
    A_bar, B_bar : numpy.ndarray
        The canonical operator pair (``B_bar = gamma * B_breve``).
    tensor : numpy.ndarray
        The simulated process of shape ``(T, m, n)``.
    """
    rng = np.random.default_rng(seed)
    A, B = _stable_pair(m, n, rng)
    tensor = _simulate(A, B, T, sigma, rng)
    A_bar, B_breve, gamma = canonical_triple(A, B)
    return A_bar, gamma * B_breve, tensor


@pytest.mark.parametrize("m, n", [(3, 3), (5, 3)])
def test_score_correctness(m, n):
    """(1) Analytic chart score vs numerical chart-direction gradient.

    Parameters
    ----------
    m, n : int
        Panel dimensions; the non-square case is checked as well.
    """
    T, h = 60, 1e-6
    A_bar, B_bar, tensor = _simulated_pair_and_tensor(m, n, T + 1, seed=20260712)

    quantities = sts.chart_score_quantities(tensor, A_bar, B_bar)
    analytic = quantities["g"].sum(axis=0)
    Q = quantities["Q"]

    numerical = np.empty_like(analytic)
    for j in range(Q.shape[1]):
        dA = Q[: m * m, j].reshape(m, m, order="F")
        dB = Q[m * m :, j].reshape(n, n, order="F")
        loss_plus = _total_loss(tensor, A_bar + h * dA, B_bar + h * dB)
        loss_minus = _total_loss(tensor, A_bar - h * dA, B_bar - h * dB)
        numerical[j] = (loss_plus - loss_minus) / (2.0 * h)

    rel_error = np.linalg.norm(analytic - numerical) / np.linalg.norm(numerical)
    assert rel_error < 1e-4, f"chart-score relative error {rel_error:.2e} exceeds 1e-4"


@pytest.mark.parametrize("n", [2, 5])
def test_commutation_matrix(n):
    """(2) K_nn @ vec(B) equals vec(B.T) exactly.

    Parameters
    ----------
    n : int
        Matrix dimension (parametrised over {2, 5}).
    """
    rng = np.random.default_rng(20260713 + n)
    B = rng.standard_normal((n, n))
    K = sts.commutation_matrix(n)

    assert np.array_equal(K @ B.flatten(order="F"), B.T.flatten(order="F"))


@pytest.mark.parametrize("m, n", [(3, 3), (5, 3)])
def test_chart_basis(m, n):
    """(3) Orthonormality. The orbit direction is purely in the amplitude.

    Parameters
    ----------
    m, n : int
        Panel dimensions.
    """
    A_bar, B_bar, _ = _simulated_pair_and_tensor(m, n, 10, seed=20260714)
    gamma = float(np.linalg.norm(B_bar, ord="fro"))
    B_breve = B_bar / gamma

    Q = sts.chart_basis(A_bar, B_breve)
    k = m * m + n * n - 1

    assert Q.shape == (m * m + n * n, k)
    assert np.max(np.abs(Q.T @ Q - np.eye(k))) < 1e-10

    # Orbit direction of the scale group: [vec(A_bar); -vec(B_bar)]. Its A-part
    # is vec(A_bar), orthogonal to every A-shape column; its B-part is
    # -gamma * vec(B_breve), orthogonal to every B-shape column, so the only
    # non-zero chart coordinate is the amplitude one (-gamma), and the
    # projection Q Q^T v cannot reproduce v.
    v = np.concatenate([A_bar.flatten(order="F"), -B_bar.flatten(order="F")])
    mode_A, mode_B, mode_gamma = sts.mode_slices(m, n)

    coords = Q.T @ v
    assert np.max(np.abs(coords[mode_A])) < 1e-12
    assert np.max(np.abs(coords[mode_B])) < 1e-12
    assert np.allclose(coords[mode_gamma], [-gamma])
    assert not np.allclose(Q @ coords, v)


@pytest.mark.parametrize("m, n", [(3, 3), (5, 3)])
def test_canonical_map_is_orbit_invariant(m, n):
    """(4) The canonical triple is unchanged by ``(A, B) -> (sA, B/s)``.

    Parameters
    ----------
    m, n : int
        Panel dimensions.
    """
    A_bar, B_bar, _ = _simulated_pair_and_tensor(m, n, 10, seed=20260718)
    reference = canonical_triple(A_bar, B_bar)

    for s in (-3.5, -1.0, 0.25, 2.0, 7.0):
        A_s, B_breve_s, gamma_s = canonical_triple(s * A_bar, B_bar / s)
        assert np.max(np.abs(A_s - reference[0])) < 1e-12
        assert np.max(np.abs(B_breve_s - reference[1])) < 1e-12
        assert abs(gamma_s - reference[2]) < 1e-12

    # ... and the invariant object really is the Kronecker product.
    assert np.allclose(
        np.kron(reference[2] * reference[1], reference[0]), np.kron(B_bar, A_bar)
    )


@pytest.mark.parametrize("m, n", [(3, 3), (5, 3)])
def test_scale_invariance(m, n):
    """(5) Every statistic agrees for the pairs ``(A, B)`` and ``(2A, B/2)``.

    Parameters
    ----------
    m, n : int
        Panel dimensions.
    """
    A_bar, B_bar, tensor = _simulated_pair_and_tensor(m, n, 80, seed=20260715)

    q1 = sts.chart_score_quantities(tensor, A_bar, B_bar)
    q2 = sts.chart_score_quantities(tensor, 2.0 * A_bar, B_bar / 2.0)

    for sl in [slice(None), q1["mode_A"], q1["mode_B"], q1["mode_gamma"]]:
        stat_1 = sts.nyblom_statistic(q1["g"][:, sl])
        stat_2 = sts.nyblom_statistic(q2["g"][:, sl])
        assert abs(stat_1 - stat_2) < 1e-8

    assert np.max(np.abs(q1["V_hat"] - q2["V_hat"])) < 1e-8

    b1 = sts.nyblom_battery(tensor, A_bar, B_bar)
    b2 = sts.nyblom_battery(tensor, 2.0 * A_bar, B_bar / 2.0)
    assert abs(b1.nyblom - b2.nyblom) < 1e-8
    assert np.allclose(b1.block_nyblom, b2.block_nyblom, atol=1e-8)


def test_block_statistics_use_v_hat_not_omega():
    """(6) Blocks decompose H^{-1} S_t against V_hat. The global one is unchanged.

    The block statistic selects blocks of the influence function and standardises
    them by the matching block of ``V_hat``. Applied to the full selection that
    construction must reproduce the global statistic exactly, because
    ``S' H^{-1} (H^{-1} Omega H^{-1})^{-1} H^{-1} S = S' Omega^{-1} S``. Applied
    to a single block it must agree with neither the raw-score block nor the
    influence block standardised by a block of ``Omega_hat``.
    """
    A_bar, B_bar, tensor = _simulated_pair_and_tensor(3, 3, 300, seed=20260815)
    q = sts.chart_score_quantities(tensor, A_bar, B_bar)
    battery = sts.nyblom_battery(tensor, A_bar, B_bar)

    full = slice(0, q["k"])
    global_via_influence = sts.nyblom_statistic(
        q["g_influence"][:, full], omega=q["V_hat"][full, full]
    )
    assert abs(battery.nyblom - global_via_influence) < 1e-8 * max(battery.nyblom, 1.0)

    for name, reported in zip(battery.block_names, battery.block_nyblom):
        sl = q[f"mode_{name}"]
        expected = sts.nyblom_statistic(
            q["g_influence"][:, sl], omega=q["V_hat"][sl, sl]
        )
        assert abs(reported - expected) < 1e-12

        wrong_omega = sts.nyblom_statistic(
            q["g_influence"][:, sl], omega=q["Omega_hat"][sl, sl]
        )
        raw_block = sts.nyblom_statistic(q["g"][:, sl])
        assert reported != wrong_omega
        assert reported != raw_block


def test_holm_matches_hand_computed_example():
    """(7) Holm reproduces a hand-computed family and dominates Bonferroni."""
    # p = (0.01, 0.04, 0.03), M = 3. Sorted: 0.01, 0.03, 0.04.
    #   rank 1: 3 * 0.01 = 0.03
    #   rank 2: max(0.03, 2 * 0.03) = 0.06
    #   rank 3: max(0.06, 1 * 0.04) = 0.06
    # Restored to input order: (0.03, 0.06, 0.06).
    adjusted = sts.holm_adjusted_pvalues([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])

    # Capped at one.
    assert np.allclose(sts.holm_adjusted_pvalues([0.5, 0.6, 0.9]), [1.0, 1.0, 1.0])

    # Never larger than Bonferroni, which is what makes Holm the better rule.
    p = np.array([0.001, 0.02, 0.049])
    assert np.all(sts.holm_adjusted_pvalues(p) <= np.minimum(p.size * p, 1.0) + 1e-15)

    with pytest.raises(ValueError):
        sts.holm_adjusted_pvalues([0.5, 1.5])


def test_var_path_calls_shared_nyblom_statistic(monkeypatch):
    """(8) The vectorised statistic goes through the MAR test's own functional.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to count the calls into the shared statistic.
    """
    _, _, tensor = _simulated_pair_and_tensor(2, 2, 60, seed=20260719)

    calls = []
    original = sts.nyblom_statistic

    def counting(g, omega=None):
        """Record the call and defer to the shared statistic."""
        calls.append(np.asarray(g).shape)
        return original(g, omega)

    monkeypatch.setattr(sts, "nyblom_statistic", counting)
    value = sts.var_nyblom_statistic(tensor)

    assert len(calls) == 1
    assert calls[0] == (tensor.shape[0] - 1, sts.var_dimension(2, 2))

    monkeypatch.undo()
    assert value == sts.nyblom_statistic(sts.var_scores(tensor))


def test_singular_covariance_raises():
    """(9) A singular covariance inside a statistic raises, and returns nothing."""
    rng = np.random.default_rng(20260720)
    g = rng.standard_normal((50, 3))
    g[:, 2] = g[:, 0]  # exactly collinear: g'g / T is singular

    with pytest.raises(sts.SingularCovarianceError):
        sts.nyblom_statistic(g)

    # The unrestricted VAR statistic hits the same wall below T - 2 >= K, which
    # is the feasibility boundary of the methodology and not a nuisance.
    _, _, short = _simulated_pair_and_tensor(3, 3, 20, seed=20260721)
    with pytest.raises(sts.SingularCovarianceError):
        sts.var_nyblom_statistic(short)


def test_statistics_run_on_a_non_square_panel():
    """(10) The whole battery is defined at (m, n) = (5, 3)."""
    m, n = 5, 3
    A_bar, B_bar, tensor = _simulated_pair_and_tensor(m, n, 200, seed=20260722)

    battery = sts.nyblom_battery(tensor, A_bar, B_bar)

    assert battery.m == m and battery.n == n
    assert battery.k == m * m + n * n - 1
    assert battery.block_k == (m * m - 1, n * n - 1, 1)
    assert np.isfinite(battery.nyblom) and battery.nyblom > 0.0
    assert all(np.isfinite(v) and v > 0.0 for v in battery.block_nyblom)
    assert battery.score_cusum.shape == (battery.T,)

    # The two specification checks and the bootstrap are defined here as well.
    kron = sts.kronecker_rank_one_check(
        tensor, (A_bar, B_bar), np.random.default_rng(1), n_boot=3
    )
    assert kron["K"] == sts.var_dimension(m, n) == (m * n) ** 2
    assert 0.0 < kron["share_observed"] <= 1.0

    variance = sts.innovation_variance_statistic(tensor, (A_bar, B_bar))
    assert variance["k"] == 1 and np.isfinite(variance["statistic"])

    boot = sts.wild_bootstrap_pvalue(
        tensor, battery.nyblom, (A_bar, B_bar), np.random.default_rng(2), n_boot=3
    )
    assert boot["n_boot"] == 3 and 0.0 < boot["p_bootstrap"] <= 1.0


def test_rolling_fit_uses_no_future_data():
    """(11) A rolling fit never uses information at or after the forecast date.

    Checks the tensor-based loop
    (``mardynamics.application.forecasting.fit_and_forecast``): its estimation
    slice is ``tensor[:index]``, taken before the rolling window is applied,
    so corrupting every entry at or after the forecast index must leave the
    fitted operators and the forecast unchanged.
    """
    import mardynamics.application.emp_config as ecfg
    import mardynamics.application.forecasting as forecasting

    # forecasting._validate_operator_pair enforces the fixed 5x5 empirical
    # panel shape (ecfg.M_DIM x ecfg.N_DIM), so the tensor must be built at
    # that shape.
    m, n, T, window = ecfg.M_DIM, ecfg.N_DIM, 40, 12
    _, _, tensor = _simulated_pair_and_tensor(m, n, T, seed=20260723)
    index = 25

    spec = forecasting.ModelSpec(model="rolling_full", window=window)
    reference = forecasting.fit_and_forecast(spec, tensor, index)

    corrupted = tensor.copy()
    corrupted[index:] = np.nan
    outcome = forecasting.fit_and_forecast(spec, corrupted, index)

    assert np.allclose(outcome["A"], reference["A"])
    assert np.allclose(outcome["B"], reference["B"])
    assert np.allclose(outcome["forecast"], reference["forecast"])


def test_limit_law_matches_the_published_quantile(tmp_path):
    """(12) The simulated k=1 95% quantile matches Nyblom (1989) within 10%.

    The published value 0.470 refers to the integrated squared Brownian
    bridge, precisely the functional simulated here. There is no trimming
    region in it. Fewer draws than production are used so the check stays
    fast, and a temporary directory keeps the production cache untouched.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary cache directory.
    """
    draws = sts.load_limit_law(
        1, tmp_path, n_draws=20000, n_grid_points=ncfg.LIMITLAW_N_GRID, seed=7
    )

    q95 = float(np.quantile(draws, 0.95))
    assert abs(q95 - 0.470) / 0.470 < 0.10, f"k=1 95% quantile {q95:.3f} is off"

    # The cache is keyed by (k, n_draws, n_grid_points, seed) alone: a second
    # call with the same key reads the file and returns the identical draws.
    again = sts.load_limit_law(
        1, tmp_path, n_draws=20000, n_grid_points=ncfg.LIMITLAW_N_GRID, seed=7
    )
    assert np.array_equal(draws, again)
    assert sts.limit_law_path(tmp_path, 1, 20000, ncfg.LIMITLAW_N_GRID, 7).exists()

    # The mean of the k-dimensional law is k/6; k = 1 gives 1/6.
    assert abs(float(np.mean(draws)) - 1.0 / 6.0) < 0.01



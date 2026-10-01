"""The design layer: populations, directions, metrics, paths and verification.

Everything that defines what is simulated lives here, and nothing that
defines how a statistic is computed. The experiment sections orchestrate
cells; they call this module for the population parameter, the planted
change directions, the metrics that make those directions comparable,
the parameter paths, their verification and the theoretical signatures,
and they call the shared numerical layer for every estimator, score and
statistic.

Three things this module is built around.

**Block coordinates and embedded directions are different objects.** A
block direction ``v_tilde_j`` lives in ``R^{k_j}``; the embedded
direction ``v_j = R_j' v_tilde_j`` lives in ``R^k``. They are carried in
separate fields of :class:`PlantedDirection`, each in its own array, and
their dimensions are asserted.

**Shapes move on the sphere by the exact exponential map.** A shape
departure is ``cos(a |w|) x_0 + sin(a |w|) w / |w|`` for a signed
displacement ``a`` and a non-zero ambient tangent ``w``. There is no
ambient addition followed by Frobenius renormalisation anywhere, a
negative displacement keeps its sign instead of being replaced by its
absolute value, and a zero direction always raises an error.

**Equal global signal is not equal blockwise signal.** Every embedded
direction is normalised by ``v_j' G v_j = 1`` under ``G = H' Omega^{-1}
H``, and the blockwise measure ``lambda_j = v_tilde_j' V_j^{-1}
v_tilde_j`` with ``V_j = R_j V R_j'`` is computed and stored beside it.
The two are reported separately.

Two charts appear and must not be confused. The shared statistics use
the orthonormal chart basis ``Q``, whose amplitude coordinate is the
ambient movement along ``vec(B_breve)`` and
therefore measures ``d gamma``. The methodology chart measures ``d kappa
= d log gamma``, and its column block parameterises the unit-norm shape
``B_breve`` instead of the scale-carrying ``B_bar``. The two differ by
the fixed diagonal rescaling ``S = diag(1_{k_A}, gamma_0 1_{k_B},
gamma_0)``: the factor attaches to the whole second block, because
``dvec(B_bar) = e^kappa (dkappa vec(B_breve) + dvec(B_breve))``. Planted
directions and every metric here are expressed in the methodology chart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

import mardynamics.experiments.seeds as seeds
import mardynamics.experiments.sim_config as simcfg
import mardynamics.core.structural_tests as structural_tests
from mardynamics.core.mar_canonical import SIGN_COORDINATE, SIGN_COORDINATE_WARN, canonical_triple

# The margin the designated sign coordinate must keep along a designed
# path. It is the shared methodology-anchored margin below which the
# sign of the canonical representative would be decided by noise instead
# of the population.
PATH_MIN_SIGN_COORDINATE = SIGN_COORDINATE_WARN


class DesignConstructionError(RuntimeError):
    """A designed object could not be built, or failed a verification condition.

    Raised when a population, a planted direction or a designed path
    fails a stated admissibility or verification condition, before any
    simulation runs.
    """


class PathVerificationError(DesignConstructionError):
    """A designed population path violates the maintained simulation design.

    Carries the full diagnostic: the population configuration, the
    changing block, the path, the departure size, the failed condition,
    the spectral-radius range, the minimum sign coordinate and the seed.
    """

    def __init__(self, report: dict):
        self.report = dict(report)
        super().__init__(
            "path verification failed: "
            + ", ".join(f"{key}={value!r}" for key, value in self.report.items())
        )


# =========================================================
# (a) The canonical population
# =========================================================
@dataclass(frozen=True)
class CanonicalTruth:
    """One deterministic population parameter in canonical coordinates.

    Attributes
    ----------
    m, n : int
        Panel dimensions.
    persistence : float
        The designed joint spectral radius.
    configuration : str
        Which of the three population configurations this is.
    A_bar : numpy.ndarray
        Unit-Frobenius row shape, sign-anchored on the designated coordinate.
    B_breve : numpy.ndarray
        Unit-Frobenius column shape.
    gamma : float
        Amplitude, chosen so that the joint spectral radius is the designed one.
    attempt_id : int
        Which attempt of the retry ladder produced this population.
    seed_key : tuple
        The exact seed identity the accepted draw came from.
    rejected_draws : tuple
        One entry per rejected attempt: its seed key and the condition it failed.
    """

    m: int
    n: int
    persistence: float
    configuration: str
    A_bar: np.ndarray
    B_breve: np.ndarray
    gamma: float
    attempt_id: int
    seed_key: tuple = ()
    rejected_draws: tuple = ()

    @property
    def kappa(self) -> float:
        """Log amplitude."""
        return float(np.log(self.gamma))

    @property
    def B_bar(self) -> np.ndarray:
        """The scale-carrying column operator."""
        return self.gamma * self.B_breve

    @property
    def k(self) -> int:
        """Canonical dimension ``k = m^2 + n^2 - 1``."""
        return self.m * self.m + self.n * self.n - 1

    @property
    def Phi(self) -> np.ndarray:
        """The transition matrix of the population."""
        return np.kron(self.B_bar, self.A_bar)

    @property
    def sign_coordinate(self) -> float:
        """The designated sign coordinate of the row shape."""
        return float(self.A_bar.flatten(order="F")[SIGN_COORDINATE])

    def operators(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the operator pair used to simulate."""
        return self.A_bar, self.B_bar


def _population_seed_key(m: int, n: int, persistence: float, configuration: str,
                         attempt: int) -> tuple:
    """Seed identity of one population draw.

    The reference configuration's seed key omits the configuration name,
    so its population, its directions and every replication stream that
    depends on them stay bit-identical however the configuration set
    changes. The two additional configurations add their own fixed
    integer to the key and are independent draws of the same procedure.

    Parameters
    ----------
    m, n : int
        Panel dimensions.
    persistence : float
        Designed joint spectral radius.
    configuration : str
        Which population configuration.
    attempt : int
        Attempt index of the retry ladder.

    Returns
    -------
    tuple
        The seed key.

    Raises
    ------
    ValueError
        For an unknown configuration name.
    """
    if configuration not in simcfg.POPULATION_CONFIGURATIONS:
        raise ValueError(
            f"unknown population configuration={configuration!r}; expected one "
            f"of {simcfg.POPULATION_CONFIGURATIONS}"
        )

    seed = simcfg.POPULATION_CONFIG_SEEDS[configuration]
    if seed is None:
        return ("truth", m, n, persistence, "attempt", attempt)
    return ("truth", m, n, persistence, "population_seed", seed, "attempt", attempt)


def _shape_admissibility(A_bar: np.ndarray, B_breve: np.ndarray) -> str | None:
    """Return the pre-specified validity condition a candidate fails, or None.

    A population draw is rejected only on a mathematical validity
    condition fixed in advance, not on anything a result depends on.
    The two conditions checkable at this stage are a numerically
    degenerate factor and the designated sign-coordinate margin.

    Parameters
    ----------
    A_bar, B_breve : numpy.ndarray
        Candidate unit-Frobenius shapes.

    Returns
    -------
    str or None
        The name of the failed condition, or None when the candidate is valid.
    """
    if abs(float(A_bar.flatten(order="F")[SIGN_COORDINATE])) < (
        simcfg.TRUTH_MIN_SIGN_COORDINATE
    ):
        return "sign_coordinate_margin"

    rho_A = float(np.max(np.abs(np.linalg.eigvals(A_bar))))
    rho_B = float(np.max(np.abs(np.linalg.eigvals(B_breve))))
    if min(rho_A, rho_B) < simcfg.TRUTH_MIN_SPECTRAL_RADIUS:
        return "degenerate_factor"

    return None


def build_truth(
    m: int,
    n: int,
    persistence: float,
    configuration: str = simcfg.REFERENCE_CONFIGURATION,
) -> CanonicalTruth:
    """Build one deterministic population by the shared generation procedure.

    Two dense standard-normal matrices are drawn, mapped to canonical
    shapes, and the amplitude is set so that the joint spectral radius
    equals the designed persistence. The seed contains the dimensions,
    the persistence and the configuration only, so the sample length
    plays no part in a population and every cell that shares those values
    shares the population.

    All three population configurations use this one procedure. They
    differ only in the fixed integer in their seed key.

    Parameters
    ----------
    m, n : int
        Panel dimensions.
    persistence : float
        The designed joint spectral radius.
    configuration : str, optional
        Which population configuration.

    Returns
    -------
    CanonicalTruth
        The accepted population, with every rejected draw logged on it.

    Raises
    ------
    DesignConstructionError
        If no attempt in the retry ladder is valid.
    """
    rejected: list[tuple] = []

    for attempt in range(simcfg.TRUTH_MAX_ATTEMPTS):
        key = _population_seed_key(m, n, persistence, configuration, attempt)
        rng = seeds.cell_rng(*key)
        A_bar, B_breve, _ = canonical_triple(
            rng.standard_normal((m, m)), rng.standard_normal((n, n))
        )

        failure = _shape_admissibility(A_bar, B_breve)
        if failure is not None:
            rejected.append((key, failure))
            continue

        rho_A = float(np.max(np.abs(np.linalg.eigvals(A_bar))))
        rho_B = float(np.max(np.abs(np.linalg.eigvals(B_breve))))
        return CanonicalTruth(
            m=m,
            n=n,
            persistence=float(persistence),
            configuration=configuration,
            A_bar=A_bar,
            B_breve=B_breve,
            gamma=float(persistence / (rho_A * rho_B)),
            attempt_id=attempt,
            seed_key=key,
            rejected_draws=tuple(rejected),
        )

    raise DesignConstructionError(
        f"build_truth: no valid population for (m, n, persistence, "
        f"configuration) = ({m}, {n}, {persistence}, {configuration!r}) within "
        f"{simcfg.TRUTH_MAX_ATTEMPTS} attempts; rejected {rejected}."
    )


def align_joint_sign(
    A_bar: np.ndarray, B_breve: np.ndarray, A_bar_population: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align an estimated canonical pair with the population on the joint sign.

    The invariant object is the transition matrix, so flipping both
    shapes leaves the model unchanged. The canonical map anchors the
    sign on a fixed coordinate, which is enough when the population
    keeps that coordinate away from zero. This function makes the
    alignment explicit and truth-referenced.

    Parameters
    ----------
    A_bar, B_breve : numpy.ndarray
        Estimated canonical shapes.
    A_bar_population : numpy.ndarray
        The population row shape to align against.

    Returns
    -------
    tuple of numpy.ndarray
        The aligned pair.
    """
    inner = float(
        A_bar.flatten(order="F") @ np.asarray(A_bar_population).flatten(order="F")
    )
    if inner < 0.0:
        return -A_bar, -B_breve
    return A_bar, B_breve


# =========================================================
# (b) Population moments, information, score covariance and metrics
# =========================================================
def stationary_second_moment(Phi: np.ndarray, sigma: float) -> np.ndarray:
    """Solve the discrete Lyapunov equation for the stationary second moment.

    Parameters
    ----------
    Phi : numpy.ndarray
        The transition matrix of the population.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    numpy.ndarray
        The stationary second-moment matrix of the vectorised process.
    """
    size = Phi.shape[0]
    return solve_discrete_lyapunov(Phi, (sigma**2) * np.eye(size))


def chart_rescaling(truth: CanonicalTruth) -> np.ndarray:
    """Diagonal map from methodology chart coordinates to appendix chart ones.

    The appendix chart is the one the score and the Hessian are expressed
    in: its ambient coordinates are the two vectorised operators. The
    methodology chart parameterises the unit-norm column shape and the
    log amplitude. Differentiating the scale-carrying column operator
    gives ``dvec(B_bar) = e^kappa (dkappa vec(B_breve) + dvec(B_breve))``,
    so the factor attaches to the whole second block: to the column-shape
    coordinates as much as to the amplitude. The row block carries no
    factor.

    Getting this wrong on the column-shape coordinates is invisible to every
    self-consistent check, but it would size the planted column direction by a
    factor of the amplitude and its signal by its square.

    Parameters
    ----------
    truth : CanonicalTruth
        The population the charts are built at.

    Returns
    -------
    numpy.ndarray
        The diagonal of the rescaling, length ``k``.
    """
    scale = np.ones(truth.k)
    scale[truth.m * truth.m - 1 :] = truth.gamma
    return scale


def tangent_bases(truth: CanonicalTruth) -> tuple[np.ndarray, np.ndarray]:
    """Return the orthonormal ambient tangent bases of the two shape blocks.

    ``T_A`` has ``k_A`` orthonormal columns spanning the tangent space of
    the unit sphere at the row shape, and ``T_B`` has ``k_B`` columns
    spanning it at the column shape. An ambient tangent direction is
    built as ``T_j v_tilde_j`` from a block-coordinate direction, and is
    orthogonal to the shape it perturbs by construction.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.

    Returns
    -------
    tuple of numpy.ndarray
        ``(T_A, T_B)`` of shapes ``(m^2, k_A)`` and ``(n^2, k_B)``.
    """
    Q = structural_tests.chart_basis(truth.A_bar, truth.B_breve)
    k_A = truth.m * truth.m - 1
    k_B = truth.n * truth.n - 1
    T_A = Q[: truth.m * truth.m, :k_A]
    T_B = Q[truth.m * truth.m :, k_A : k_A + k_B]
    return np.ascontiguousarray(T_A), np.ascontiguousarray(T_B)


def selection_matrix(truth: CanonicalTruth, block: str) -> np.ndarray:
    """Return the selection matrix that extracts one canonical block.

    ``R_j`` is ``(k_j, k)``: it maps an embedded direction to its block
    coordinates, and its transpose embeds a block direction back into the full
    parameter.

    Parameters
    ----------
    truth : CanonicalTruth
        The population whose chart is used.
    block : str
        Which canonical block.

    Returns
    -------
    numpy.ndarray
        The selection matrix.
    """
    mode_A, mode_B, mode_gamma = structural_tests.mode_slices(truth.m, truth.n)
    chosen = {"row": mode_A, "column": mode_B, "amplitude": mode_gamma}[block]
    return np.eye(truth.k)[chosen, :]


def population_information(truth: CanonicalTruth, sigma: float) -> np.ndarray:
    """Analytic population Hessian in the methodology chart.

    The appendix gives the Hessian as twice the chart-transformed
    expected outer product of the regressor Jacobian. Because that
    Jacobian is linear in the observation, the expectation is assembled
    from the eigendecomposition of the stationary second moment, without
    simulating data or forming the Kronecker product of the unrestricted
    size.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    numpy.ndarray
        The ``(k, k)`` Hessian in the methodology chart.
    """
    A_bar, B_bar = truth.operators()
    m, n = truth.m, truth.n

    Gamma_0 = stationary_second_moment(truth.Phi, sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(Gamma_0)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    K_nn = structural_tests.commutation_matrix(n)
    ambient = np.zeros((m * m + n * n, m * m + n * n))

    for weight, vector in zip(eigenvalues, eigenvectors.T):
        if weight <= 0.0:
            continue
        W = vector.reshape(m, n, order="F")
        Z = structural_tests.regressor_jacobian(A_bar, B_bar, W, K_nn=K_nn)
        ambient += weight * (Z.T @ Z)

    Q = structural_tests.chart_basis(truth.A_bar, truth.B_breve)
    H_appendix = 2.0 * (Q.T @ ambient @ Q)

    scale = chart_rescaling(truth)
    return H_appendix * np.outer(scale, scale)


def population_score_covariance(truth: CanonicalTruth, sigma: float) -> np.ndarray:
    """Analytic population score covariance in the methodology chart.

    Under the isotropic Gaussian design the least-squares algebra gives
    ``Omega = 2 sigma^2 H``. :func:`moment_check_signal_metric` is an
    independent check of that identity.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    numpy.ndarray
        The ``(k, k)`` score covariance.
    """
    return 2.0 * sigma**2 * population_information(truth, sigma)


def parameter_covariance(truth: CanonicalTruth, sigma: float) -> np.ndarray:
    """Population sandwich covariance ``V = H^{-1} Omega H^{-1}``.

    Computed from ``H`` and ``Omega`` directly, instead of the analytic
    simplification.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    numpy.ndarray
        The ``(k, k)`` sandwich covariance.

    Raises
    ------
    DesignConstructionError
        If the population Hessian is singular.
    """
    H = population_information(truth, sigma)
    Omega = population_score_covariance(truth, sigma)
    inverse = _checked_inverse(H, "population Hessian H")
    return inverse @ Omega @ inverse


def signal_metric(truth: CanonicalTruth, sigma: float) -> np.ndarray:
    """Population signal metric ``G = H' Omega^{-1} H`` in the methodology chart.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    numpy.ndarray
        The ``(k, k)`` signal metric.

    Raises
    ------
    DesignConstructionError
        If the population score covariance is singular.
    """
    H = population_information(truth, sigma)
    Omega = population_score_covariance(truth, sigma)
    return H.T @ _checked_inverse(Omega, "population score covariance Omega") @ H


def _checked_inverse(matrix: np.ndarray, label: str) -> np.ndarray:
    """Invert a symmetric positive definite matrix, or fail the design.

    The check is the Cholesky factorisation, the same way as in the shared
    statistics: it succeeds precisely when the matrix is numerically
    positive definite.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix to invert.
    label : str
        Name used in the failure message.

    Returns
    -------
    numpy.ndarray
        The inverse.

    Raises
    ------
    DesignConstructionError
        If the matrix is not finite or not numerically positive definite.
    """
    matrix = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(matrix)):
        raise DesignConstructionError(f"{label} is not finite.")
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise DesignConstructionError(
            f"{label} is singular or not positive definite (shape "
            f"{matrix.shape}); the design requires it to be invertible and no "
            "regularisation is applied."
        ) from exc
    identity = np.eye(matrix.shape[0])
    return np.linalg.solve(factor.T, np.linalg.solve(factor, identity))


def moment_check_signal_metric(
    truth: CanonicalTruth, sigma: float, n_obs: int, rng: np.random.Generator
) -> dict:
    """Check the analytic signal metric against one long independent panel.

    The panel is simulated at the population under isotropic Gaussian
    innovations and the chart quantities are evaluated there, giving the metric
    without the analytic proportionality. The check is on the relative Frobenius
    difference.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    sigma : float
        Per-entry innovation standard deviation.
    n_obs : int
        Length of the independent panel.
    rng : numpy.random.Generator
        Source of the innovations.

    Returns
    -------
    dict
        Keys ``relative_difference``, ``tolerance``, ``passed`` and ``n_obs``.
    """
    from mardynamics.experiments.dgp import simulate_static_tensor
    from mardynamics.experiments.innovations import IsotropicGaussian

    tensor = simulate_static_tensor(
        truth, n_obs, IsotropicGaussian(sigma), rng, burn_in=simcfg.BURN_IN
    )
    quantities = structural_tests.chart_score_quantities(tensor, *truth.operators())

    H_hat = quantities["H_hat"]
    Omega_hat = quantities["Omega_hat"]
    G_hat_appendix = H_hat.T @ np.linalg.solve(Omega_hat, H_hat)

    scale = chart_rescaling(truth)
    G_hat = G_hat_appendix * np.outer(scale, scale)
    G = signal_metric(truth, sigma)

    relative = float(
        np.linalg.norm(G_hat - G, ord="fro") / np.linalg.norm(G, ord="fro")
    )
    return {
        "relative_difference": relative,
        "tolerance": simcfg.G_MOMENT_CHECK_TOLERANCE,
        "passed": bool(relative < simcfg.G_MOMENT_CHECK_TOLERANCE),
        "n_obs": int(n_obs),
    }


# =========================================================
# (c) Planted directions: block coordinates and embedded directions
# =========================================================
@dataclass(frozen=True)
class PlantedDirection:
    """One change direction of a canonical block, in both of its coordinates.

    Attributes
    ----------
    block : str
        Which canonical block moves.
    configuration : str
        Which population configuration the direction belongs to.
    block_coordinates : numpy.ndarray
        ``v_tilde_j``, length ``k_j``. The block-coordinate direction.
    coordinates : numpy.ndarray
        ``v_j = R_j' v_tilde_j``, length ``k``. The embedded direction,
        scaled so that the global signal equals one.
    ambient_tangent : numpy.ndarray
        ``w_j = T_j v_tilde_j`` for a shape block, the ambient tangent that the
        exponential map moves along; an empty array for the amplitude block,
        which does not move on a sphere.
    tangent_orthogonality : float
        The realised inner product of the ambient tangent with the shape it
        perturbs; zero for the amplitude block.
    global_signal : float
        The realised ``v_j' G v_j``.
    global_signal_deviation : float
        Its absolute deviation from one.
    blockwise_signal : float
        ``lambda_j = v_tilde_j' V_j^{-1} v_tilde_j``, the blockwise signal
        measure. It is not implied by the global normalisation and is
        reported separately from block dimension.
    k_block : int
        ``k_j``, asserted against the length of ``block_coordinates``.
    k_full : int
        ``k``, asserted against the length of ``coordinates``.
    seed_key : tuple
        The seed identity the direction was drawn from; empty for the amplitude
        block, whose direction is deterministic.
    """

    block: str
    configuration: str
    block_coordinates: np.ndarray
    coordinates: np.ndarray
    ambient_tangent: np.ndarray
    tangent_orthogonality: float
    global_signal: float
    global_signal_deviation: float
    blockwise_signal: float
    k_block: int
    k_full: int
    seed_key: tuple = ()


def blockwise_signal_measure(
    block_coordinates: np.ndarray, V: np.ndarray, R: np.ndarray
) -> float:
    """Return the blockwise signal measure of a block-coordinate direction.

    ``lambda_j = v_tilde_j' V_j^{-1} v_tilde_j`` with ``V_j = R_j V R_j'``. It
    is a different quantity from the global signal. Two directions
    with the same ``v' G v`` generally have different ``lambda_j``, and that
    difference is one of the two things a difference in blockwise rejection
    rates can come from. The other is block dimension. This function measures
    the first; it says nothing about the second.

    Parameters
    ----------
    block_coordinates : numpy.ndarray
        ``v_tilde_j``, length ``k_j``.
    V : numpy.ndarray
        The population sandwich covariance, ``(k, k)``.
    R : numpy.ndarray
        The selection matrix of the block, ``(k_j, k)``.

    Returns
    -------
    float
        The blockwise signal measure.

    Raises
    ------
    DesignConstructionError
        If the block covariance is singular, or the dimensions disagree.
    """
    block_coordinates = np.asarray(block_coordinates, dtype=float)
    if block_coordinates.ndim != 1 or block_coordinates.size != R.shape[0]:
        raise DesignConstructionError(
            f"blockwise_signal_measure: block direction has length "
            f"{block_coordinates.size}, the selection matrix expects "
            f"{R.shape[0]}."
        )

    V_block = R @ V @ R.T
    inverse = _checked_inverse(V_block, f"block covariance V_j (k_j={R.shape[0]})")
    return float(block_coordinates @ inverse @ block_coordinates)


def planted_direction(
    truth: CanonicalTruth, block: str, G: np.ndarray, V: np.ndarray
) -> PlantedDirection:
    """Build the one fixed change direction of a block, in both coordinates.

    For a shape block a dense standard-normal vector is drawn in the
    block coordinates of that block. Its ambient tangent is read off the
    orthonormal tangent basis, so it is orthogonal to the shape it
    perturbs by construction. For the amplitude the block direction is
    the unit direction in the log magnitude.

    The embedded direction is then scaled to make the global signal
    equal one, and the block coordinates are recovered from the scaled
    embedded direction.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    block : str
        Which canonical block.
    G : numpy.ndarray
        The population signal metric in the methodology chart.
    V : numpy.ndarray
        The population sandwich covariance in the methodology chart.

    Returns
    -------
    PlantedDirection
        The direction with its realised normalisation and blockwise signal.

    Raises
    ------
    DesignConstructionError
        If the drawn direction is degenerate, if the normalisation misses
        the tolerance, or if the ambient tangent is not orthogonal to the
        shape it perturbs.
    """
    k_block = simcfg.block_dimension(block, truth.m, truth.n)
    R = selection_matrix(truth, block)
    T_A, T_B = tangent_bases(truth)

    if block == "amplitude":
        # The amplitude direction is deterministic: there is one unit direction
        # in the log magnitude, and no draw is needed to find it.
        seed_key: tuple = ()
        block_coordinates = np.ones(1)
    else:
        seed_key = ("direction", truth.m, truth.n, truth.persistence, block)
        if truth.configuration != simcfg.REFERENCE_CONFIGURATION:
            # The reference configuration's seed key omits the
            # configuration name, so its directions match the objects
            # earlier runs used.
            seed_key = seed_key + (truth.configuration,)
        # The draw is ambient and is read into block coordinates through
        # the tangent basis, instead of being drawn in block coordinates
        # directly. The two are the same distribution, but only this
        # order reproduces the reference directions.
        ambient_basis = T_A if block == "row" else T_B
        drawn = seeds.cell_rng(*seed_key).standard_normal(ambient_basis.shape[0])
        block_coordinates = ambient_basis.T @ drawn

    if block_coordinates.size != k_block:
        raise DesignConstructionError(
            f"planted_direction: {block} block direction has length "
            f"{block_coordinates.size}, expected k_j = {k_block}."
        )

    coordinates = R.T @ block_coordinates
    if coordinates.size != truth.k:
        raise DesignConstructionError(
            f"planted_direction: embedded {block} direction has length "
            f"{coordinates.size}, expected k = {truth.k}."
        )

    quadratic = float(coordinates @ G @ coordinates)
    if not np.isfinite(quadratic) or quadratic <= 0.0:
        raise DesignConstructionError(
            f"planted_direction: degenerate {block} direction (v'Gv = {quadratic})."
        )

    coordinates = coordinates / np.sqrt(quadratic)
    block_coordinates = R @ coordinates

    realised = float(coordinates @ G @ coordinates)
    deviation = abs(realised - 1.0)
    if deviation > simcfg.G_NORMALISATION_TOLERANCE:
        raise DesignConstructionError(
            f"planted_direction: {block} global-signal normalisation off by "
            f"{deviation:.3e}, above {simcfg.G_NORMALISATION_TOLERANCE:g}."
        )

    if block == "row":
        ambient = T_A @ block_coordinates
        anchor = truth.A_bar.flatten(order="F")
    elif block == "column":
        ambient = T_B @ block_coordinates
        anchor = truth.B_breve.flatten(order="F")
    else:
        ambient = np.zeros(0)
        anchor = np.zeros(0)

    orthogonality = float(abs(ambient @ anchor)) if ambient.size else 0.0
    if orthogonality > simcfg.TANGENT_ORTHOGONALITY_TOLERANCE:
        raise DesignConstructionError(
            f"planted_direction: the {block} ambient tangent is not orthogonal "
            f"to the shape it perturbs ({orthogonality:.3e} > "
            f"{simcfg.TANGENT_ORTHOGONALITY_TOLERANCE:g})."
        )

    return PlantedDirection(
        block=block,
        configuration=truth.configuration,
        block_coordinates=block_coordinates,
        coordinates=coordinates,
        ambient_tangent=ambient,
        tangent_orthogonality=orthogonality,
        global_signal=realised,
        global_signal_deviation=deviation,
        blockwise_signal=blockwise_signal_measure(block_coordinates, V, R),
        k_block=int(k_block),
        k_full=int(truth.k),
        seed_key=seed_key,
    )


# =========================================================
# (d) Scalar paths, departures and the exact exponential map
# =========================================================
def path_function(path: str, u: np.ndarray) -> np.ndarray:
    """Evaluate the scalar path on the unit interval.

    Parameters
    ----------
    path : {"break", "drift", "recurrent"}
        Which path.
    u : numpy.ndarray
        Sample fractions.

    Returns
    -------
    numpy.ndarray
        The indicator of the second half, the fraction itself, or one full sine
        excursion.

    Raises
    ------
    ValueError
        For an unknown path name.
    """
    u = np.asarray(u, dtype=float)
    if path == "break":
        return (u > simcfg.BREAK_FRACTION).astype(float)
    if path == "drift":
        return u
    if path == "recurrent":
        return np.sin(2.0 * np.pi * u)
    raise ValueError(f"unknown path={path!r}")


def departure_scalar(magnitude: str, N: int) -> float:
    """Return the scalar that multiplies the planted direction.

    A local departure carries ``N^(-1/2) c``. The fixed departure carries
    its own coefficient and does not shrink with the sample.

    Parameters
    ----------
    magnitude : {"small", "medium", "large"}
        Which magnitude.
    N : int
        Number of score observations.

    Returns
    -------
    float
        The scalar.

    Raises
    ------
    ValueError
        For an unknown magnitude.
    """
    if magnitude in simcfg.LOCAL_MAGNITUDE_SCALARS:
        return simcfg.LOCAL_MAGNITUDE_SCALARS[magnitude] / np.sqrt(N)
    if magnitude == "large":
        return simcfg.LARGE_MAGNITUDE
    raise ValueError(f"unknown magnitude={magnitude!r}")


def equivalent_local_scalar(magnitude: str, N: int) -> float:
    """Return the local coefficient a departure is equivalent to at this sample.

    A local magnitude contributes its own coefficient. The fixed magnitude
    contributes the local coefficient that would produce the same
    departure at this sample size, which is why it is not local: that
    coefficient grows with the sample.

    Parameters
    ----------
    magnitude : str
        Which magnitude.
    N : int
        Number of score observations.

    Returns
    -------
    float
        The equivalent local coefficient.
    """
    if magnitude in simcfg.LOCAL_MAGNITUDE_SCALARS:
        return float(simcfg.LOCAL_MAGNITUDE_SCALARS[magnitude])
    return float(simcfg.LARGE_MAGNITUDE * np.sqrt(N))


def sample_fractions(T: int) -> np.ndarray:
    """Return the sample fractions of the retained observations.

    The first retained observation sits at zero and still carries the
    initial parameter, so the designed change occupies the retained
    sample.

    The fractions index the ``T`` retained observations, while the
    statistics are built on the ``N = T - 1`` scored transitions, which
    run from index 1 to ``T - 1``. The two grids are one apart, so a
    break placed at the fraction one half does not split the scored
    sample evenly. At the reference sample length the break falls at
    observation 200, leaving 200 of the 399 scored transitions after it:
    a fraction of 0.5013, not 0.5. The offset is ``1 / (2 N)``, a
    property of scoring one-step transitions instead of a rounding
    choice. It is reported instead of removed: moving the break to
    equalise the scored halves would change the designed path.

    Parameters
    ----------
    T : int
        Number of retained observations.

    Returns
    -------
    numpy.ndarray
        The ``T`` fractions.
    """
    return np.arange(T, dtype=float) / float(T - 1)


def sphere_exponential(
    base: np.ndarray, direction: np.ndarray, displacement: float = 1.0
) -> np.ndarray:
    """Exact exponential map of the unit Frobenius sphere.

    Returns ``cos(a |w|) x_0 + sin(a |w|) w / |w|`` for base point ``x_0``,
    non-zero ambient tangent direction ``w`` and signed scalar
    displacement ``a``. The displacement keeps its sign: a negative one
    moves the opposite way along the same geodesic, which is what the
    recurrent path needs, and it is not replaced by its absolute value.

    A shape departure is not built by ambient addition followed by Frobenius
    renormalisation. The two constructions differ at second order, and only the
    geodesic keeps a designed step length meaningful.

    Parameters
    ----------
    base : numpy.ndarray
        Unit-norm base point, flattened.
    direction : numpy.ndarray
        Ambient tangent direction at the base point, flattened. Must be
        non-zero.
    displacement : float, optional
        The signed scalar displacement.

    Returns
    -------
    numpy.ndarray
        The unit-norm image point.

    Raises
    ------
    DesignConstructionError
        If the direction is numerically zero. A caller that does not move a
        shape must not call the exponential map at all.
    """
    direction = np.asarray(direction, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-15:
        raise DesignConstructionError(
            "sphere_exponential: the tangent direction is numerically zero; the "
            "exponential map is undefined there and a block that does not move "
            "must not be routed through it."
        )

    angle = float(displacement) * norm
    return np.cos(angle) * np.asarray(base, dtype=float) + np.sin(angle) * (
        direction / norm
    )


def signature_scale(path: str, n_grid: int = 20001) -> float:
    """Integrated squared cumulative demeaned scalar signature of a path.

    This is the number reported per designed path: the integral over the
    unit interval of the square of the cumulative demeaned path function.
    It is a property of the path shape alone and carries no magnitude.

    It is not called an integrated squared demeaned path: the integrand
    is the cumulative demeaned signature, not the demeaned path itself.

    Parameters
    ----------
    path : str
        Which scalar path.
    n_grid : int, optional
        Trapezoid grid size.

    Returns
    -------
    float
        The integral.
    """
    r = np.linspace(0.0, 1.0, n_grid)
    q = path_function(path, r)
    antiderivative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(r))]
    )
    cumulative_demeaned = antiderivative - r * antiderivative[-1]
    return float(np.trapezoid(cumulative_demeaned**2, r))


def path_noncentrality(path: str, local_scalar: float) -> float:
    """Noncentrality of the limiting statistic for a path and a local coefficient.

    Because every planted direction carries unit global signal, the
    standardised mean shift reduces to the local coefficient squared
    times the signature scale of the path. The three paths are not
    equalised on it: the recurrent path spends part of its excursion
    cancelling itself, a property of the design.

    The coefficient is the local one, not the scalar that multiplies the
    direction in the path itself: the noncentrality is a property of the
    local sequence, so it does not shrink with the sample.

    Parameters
    ----------
    path : str
        Which scalar path.
    local_scalar : float
        The local coefficient, or its equivalent for a fixed magnitude.

    Returns
    -------
    float
        The noncentrality.
    """
    return float(local_scalar**2 * signature_scale(path))


def theoretical_signature(path: str, r: np.ndarray, n_grid: int = 20001) -> np.ndarray:
    """Theoretical mean signature of a scalar path, per unit magnitude.

    The cumulative demeaned path function, which the standardised influence path
    converges to with a minus sign.

    Parameters
    ----------
    path : str
        Which scalar path.
    r : numpy.ndarray
        Fractions in the unit interval.
    n_grid : int, optional
        Trapezoid grid size.

    Returns
    -------
    numpy.ndarray
        The signature at each fraction.
    """
    r = np.asarray(r, dtype=float)
    grid = np.linspace(0.0, 1.0, n_grid)
    q = path_function(path, grid)
    antiderivative = np.concatenate(
        [[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * np.diff(grid))]
    )
    cumulative_demeaned = antiderivative - grid * antiderivative[-1]
    return np.interp(r, grid, cumulative_demeaned)


# =========================================================
# (e) The designed canonical path
# =========================================================
@dataclass(frozen=True)
class ParameterPath:
    """A designed canonical parameter path over one retained sample.

    Attributes
    ----------
    A_bar : numpy.ndarray
        Row shapes, ``(T, m, m)``.
    B_breve : numpy.ndarray
        Column shapes, ``(T, n, n)``.
    gamma : numpy.ndarray
        Amplitudes, length ``T``.
    kappa : numpy.ndarray
        Log amplitudes, length ``T``.
    spectral_radius : numpy.ndarray
        The realised joint spectral radius along the path.
    block, path, magnitude : str or None
        The design coordinates, or None for a constant path.
    configuration : str
        Which population configuration the path starts from.
    scalar : float
        The scalar multiplying the planted direction in the path itself.
    local_scalar : float or None
        The coefficient of the local alternative, or None for the fixed
        magnitude, whose standardised path has no bounded signature.
    equivalent_local_scalar : float
        The coefficient the noncentrality is built from.
    signature_scale : float
        The integrated squared cumulative demeaned scalar signature of the path.
    noncentrality : float
        The equivalent local coefficient squared times the signature scale.
    """

    A_bar: np.ndarray
    B_breve: np.ndarray
    gamma: np.ndarray
    kappa: np.ndarray
    spectral_radius: np.ndarray
    block: str | None
    path: str | None
    magnitude: str | None
    configuration: str
    scalar: float
    local_scalar: float | None
    equivalent_local_scalar: float
    signature_scale: float
    noncentrality: float

    def operators_at(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the operator pair driving the transition into one observation.

        Parameters
        ----------
        index : int
            Retained observation index.

        Returns
        -------
        tuple of numpy.ndarray
            The operator pair.
        """
        return self.A_bar[index], self.gamma[index] * self.B_breve[index]

    @property
    def is_constant(self) -> bool:
        """Whether the path stays at its initial parameter throughout."""
        return bool(
            np.allclose(self.A_bar, self.A_bar[0])
            and np.allclose(self.B_breve, self.B_breve[0])
            and np.allclose(self.gamma, self.gamma[0])
        )


def build_parameter_path(
    truth: CanonicalTruth,
    direction: PlantedDirection,
    path: str,
    magnitude: str,
    T: int,
) -> ParameterPath:
    """Build the canonical parameter path of one planted design.

    A shape block moves along the sphere geodesic through the exact
    exponential map, driven by the signed displacement ``a_t = scalar *
    q(u_t)``. The amplitude moves additively in the log magnitude. A
    block that does not move is left at its population value and is
    not routed through the exponential map, whose zero-direction case
    is an error.

    The path is returned unverified. :func:`verify_parameter_path`
    applies the required conditions and is what fails a violating
    design.

    Parameters
    ----------
    truth : CanonicalTruth
        The population the path starts from.
    direction : PlantedDirection
        The planted direction.
    path : str
        Which scalar path.
    magnitude : str
        Which departure size.
    T : int
        Number of retained observations.

    Returns
    -------
    ParameterPath
        The designed path.
    """
    m, n = truth.m, truth.n
    N = simcfg.score_observations(T)
    u = sample_fractions(T)

    q = path_function(path, u)
    scalar = departure_scalar(magnitude, N)

    a_flat = truth.A_bar.flatten(order="F")
    b_flat = truth.B_breve.flatten(order="F")

    moves_row = direction.block == "row"
    moves_column = direction.block == "column"
    kappa_direction = (
        float(direction.block_coordinates[0])
        if direction.block == "amplitude"
        else 0.0
    )

    A_path = np.empty((T, m, m))
    B_path = np.empty((T, n, n))
    kappa_path = np.empty(T)

    for j in range(T):
        displacement = scalar * float(q[j])

        if moves_row:
            A_flat_t = sphere_exponential(
                a_flat, direction.ambient_tangent, displacement
            )
        else:
            A_flat_t = a_flat

        if moves_column:
            B_flat_t = sphere_exponential(
                b_flat, direction.ambient_tangent, displacement
            )
        else:
            B_flat_t = b_flat

        A_path[j] = A_flat_t.reshape(m, m, order="F")
        B_path[j] = B_flat_t.reshape(n, n, order="F")
        kappa_path[j] = truth.kappa + displacement * kappa_direction

    gamma_path = np.exp(kappa_path)
    radii = np.array(
        [
            float(
                np.max(
                    np.abs(np.linalg.eigvals(np.kron(gamma_path[j] * B_path[j], A_path[j])))
                )
            )
            for j in range(T)
        ]
    )

    local = simcfg.LOCAL_MAGNITUDE_SCALARS.get(magnitude)
    equivalent = equivalent_local_scalar(magnitude, N)
    scale = signature_scale(path)

    return ParameterPath(
        A_bar=A_path,
        B_breve=B_path,
        gamma=gamma_path,
        kappa=kappa_path,
        spectral_radius=radii,
        block=direction.block,
        path=path,
        magnitude=magnitude,
        configuration=truth.configuration,
        scalar=float(scalar),
        local_scalar=None if local is None else float(local),
        equivalent_local_scalar=float(equivalent),
        signature_scale=float(scale),
        noncentrality=float(equivalent**2 * scale),
    )


# =========================================================
# (f) Population-path verification
# =========================================================
def verify_parameter_path(
    truth: CanonicalTruth,
    direction: PlantedDirection,
    built: ParameterPath,
    strict: bool = True,
) -> dict:
    """Verify one designed path against the maintained simulation design.

    Computes and returns every required quantity. When ``strict`` is
    set, a violation raises :class:`PathVerificationError` with the full
    diagnostic. A path that violates the maintained design is a design
    failure and is reported as one.

    Parameters
    ----------
    truth : CanonicalTruth
        The population the path starts from.
    direction : PlantedDirection
        The planted direction the path was built along.
    built : ParameterPath
        The designed path to verify.
    strict : bool, optional
        Whether a violation raises instead of only being reported.

    Returns
    -------
    dict
        The verification record.

    Raises
    ------
    PathVerificationError
        On the first violated condition when ``strict`` is set.
    """
    a_flat = truth.A_bar.flatten(order="F")
    b_flat = truth.B_breve.flatten(order="F")

    row_norm_error = float(
        np.max(np.abs(np.linalg.norm(built.A_bar, ord="fro", axis=(1, 2)) - 1.0))
    )
    column_norm_error = float(
        np.max(np.abs(np.linalg.norm(built.B_breve, ord="fro", axis=(1, 2)) - 1.0))
    )

    # Geodesic displacement is the arc length on the unit sphere between
    # the population shape and the shape at each retained index, which is
    # what the exponential map moves by. The Frobenius displacement is
    # the chord.
    row_inner = np.clip(
        np.array([x.flatten(order="F") @ a_flat for x in built.A_bar]), -1.0, 1.0
    )
    column_inner = np.clip(
        np.array([x.flatten(order="F") @ b_flat for x in built.B_breve]), -1.0, 1.0
    )
    geodesic = float(max(np.max(np.arccos(row_inner)), np.max(np.arccos(column_inner))))

    frobenius = float(
        max(
            np.max(np.linalg.norm(built.A_bar - truth.A_bar, ord="fro", axis=(1, 2))),
            np.max(
                np.linalg.norm(built.B_breve - truth.B_breve, ord="fro", axis=(1, 2))
            ),
        )
    )
    log_magnitude = float(np.max(np.abs(built.kappa - truth.kappa)))
    sign_coordinates = np.array([x.flatten(order="F")[SIGN_COORDINATE] for x in built.A_bar])

    tangent_error = direction.tangent_orthogonality
    signal_error = direction.global_signal_deviation

    record = {
        "configuration": truth.configuration,
        "m": truth.m,
        "n": truth.n,
        "T": int(built.A_bar.shape[0]),
        "block": built.block,
        "path": built.path,
        "magnitude": built.magnitude,
        "population_seed_key": "|".join(str(part) for part in truth.seed_key),
        "direction_seed_key": "|".join(str(p) for p in direction.seed_key)
        or "deterministic",
        "row_shape_norm_error": row_norm_error,
        "column_shape_norm_error": column_norm_error,
        "tangent_orthogonality_error": float(tangent_error),
        "global_signal_error": float(signal_error),
        "max_geodesic_displacement": geodesic,
        "max_frobenius_displacement": frobenius,
        "max_abs_log_magnitude_displacement": log_magnitude,
        "min_gamma": float(built.gamma.min()),
        "max_gamma": float(built.gamma.max()),
        "min_spectral_radius": float(built.spectral_radius.min()),
        "max_spectral_radius": float(built.spectral_radius.max()),
        "min_sign_coordinate": float(sign_coordinates.min()),
        "integrated_squared_cumulative_demeaned_signature": built.signature_scale,
        "departure_scalar": built.scalar,
        "equivalent_local_scalar": built.equivalent_local_scalar,
        "noncentrality": built.noncentrality,
        "stability_limit": simcfg.MAX_RADIUS,
        "sign_coordinate_margin": PATH_MIN_SIGN_COORDINATE,
        "verification_tolerance": simcfg.VERIFICATION_TOLERANCE,
    }

    failures = []
    if row_norm_error > simcfg.SHAPE_NORM_TOLERANCE:
        failures.append("row_shape_norm")
    if column_norm_error > simcfg.SHAPE_NORM_TOLERANCE:
        failures.append("column_shape_norm")
    if tangent_error > simcfg.TANGENT_ORTHOGONALITY_TOLERANCE:
        failures.append("tangent_orthogonality")
    if signal_error > simcfg.G_NORMALISATION_TOLERANCE:
        failures.append("global_signal_normalisation")
    if not np.all(np.isfinite(built.gamma)) or built.gamma.min() <= 0.0:
        failures.append("non_positive_amplitude")
    if built.spectral_radius.max() >= simcfg.MAX_RADIUS:
        failures.append("stability_region")
    if sign_coordinates.min() < PATH_MIN_SIGN_COORDINATE:
        failures.append("sign_coordinate_margin")

    # Written as an explicit word instead of an empty string: an empty
    # cell in a CSV reads back as a missing value, and "this path failed
    # nothing" is a finding, not a gap.
    record["failed_conditions"] = "|".join(failures) if failures else "none"
    record["inside_maintained_space"] = not failures

    if failures and strict:
        raise PathVerificationError(
            {
                "population_configuration": truth.configuration,
                "changing_block": built.block,
                "path": built.path,
                "departure_size": built.magnitude,
                "failed_condition": "|".join(failures),
                "min_spectral_radius": record["min_spectral_radius"],
                "max_spectral_radius": record["max_spectral_radius"],
                "min_sign_coordinate": record["min_sign_coordinate"],
                "population_seed_key": record["population_seed_key"],
                "direction_seed_key": record["direction_seed_key"],
            }
        )

    return record


# =========================================================
# (g) The unrestricted comparison of the matrix restriction
# =========================================================
def kronecker_jacobian(truth: CanonicalTruth) -> np.ndarray:
    """Jacobian of the vectorised transition matrix in the methodology chart.

    A row perturbation moves the transition matrix by one Kronecker
    product and a column perturbation by the other, so the ambient
    Jacobian is assembled column by column and composed with the chart
    basis.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.

    Returns
    -------
    numpy.ndarray
        The ``(K, k)`` Jacobian.
    """
    A_bar, B_bar = truth.operators()
    m, n = truth.m, truth.n
    K = (m * n) ** 2

    ambient = np.empty((K, m * m + n * n))
    for i in range(m * m):
        unit = np.zeros(m * m)
        unit[i] = 1.0
        ambient[:, i] = np.kron(B_bar, unit.reshape(m, m, order="F")).flatten(order="F")
    for j in range(n * n):
        unit = np.zeros(n * n)
        unit[j] = 1.0
        ambient[:, m * m + j] = np.kron(unit.reshape(n, n, order="F"), A_bar).flatten(
            order="F"
        )

    Q = structural_tests.chart_basis(truth.A_bar, truth.B_breve)
    return (ambient @ Q) * chart_rescaling(truth)[None, :]


def unrestricted_signal_metric(truth: CanonicalTruth, sigma: float) -> np.ndarray:
    """The unrestricted population signal metric on the transition matrix.

    Under the isotropic design the unrestricted least-squares algebra gives the
    same proportionality as the restricted one, so the metric equals the
    population regressor metric divided by the innovation variance.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    numpy.ndarray
        The ``(K, K)`` unrestricted signal metric.
    """
    size = truth.m * truth.n
    Gamma_0 = stationary_second_moment(truth.Phi, sigma)
    return np.kron(Gamma_0, np.eye(size)) / sigma**2


@dataclass(frozen=True)
class InducedSeparableDirection:
    """The unrestricted transition direction induced by a canonical direction.

    Attributes
    ----------
    matrix : numpy.ndarray
        The ``(mn, mn)`` induced direction.
    vector : numpy.ndarray
        Its vectorisation, length ``K``.
    unrestricted_signal : float
        The realised unrestricted signal norm. Under the isotropic
        design the canonical unit normalisation implies this equals one.
        The value is verified and not rescaled a second time.
    deviation : float
        Its absolute deviation from one.
    """

    matrix: np.ndarray
    vector: np.ndarray
    unrestricted_signal: float
    deviation: float


def induced_separable_direction(
    truth: CanonicalTruth, direction: PlantedDirection, sigma: float
) -> InducedSeparableDirection:
    """Induce the unrestricted transition direction from a canonical direction.

    The canonical direction is pushed through the Kronecker Jacobian. Its
    unrestricted signal norm is then measured. Under the isotropic
    design the two normalisations coincide by construction, because the
    restricted information is the Jacobian-transformed unrestricted one.
    Applying a second scaling here would change the intended canonical
    departure, so it is not applied. A numerically incompatible pair is
    reported as a discrepancy and stops the design.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    direction : PlantedDirection
        The canonical direction, normalised under the restricted metric.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    InducedSeparableDirection
        The induced direction and its verified norm.

    Raises
    ------
    DesignConstructionError
        If the two unit normalisations are not numerically compatible.
    """
    J = kronecker_jacobian(truth)
    vector = J @ direction.coordinates

    G_phi = unrestricted_signal_metric(truth, sigma)
    realised = float(vector @ G_phi @ vector)
    deviation = abs(realised - 1.0)

    if deviation > simcfg.G_PERP_ORTHOGONALITY_TOLERANCE:
        raise DesignConstructionError(
            "induced_separable_direction: the canonical and unrestricted unit "
            f"normalisations are not numerically compatible for the {direction.block} "
            f"direction; the induced unrestricted signal is {realised:.12f}, off by "
            f"{deviation:.3e}. The design is not silently rescaled."
        )

    size = truth.m * truth.n
    return InducedSeparableDirection(
        matrix=vector.reshape(size, size, order="F"),
        vector=vector,
        unrestricted_signal=realised,
        deviation=deviation,
    )


@dataclass(frozen=True)
class OrthogonalDeparture:
    """A non-separable departure direction outside the Kronecker tangent space.

    Attributes
    ----------
    matrix : numpy.ndarray
        The ``(mn, mn)`` direction, normalised under the unrestricted metric.
    vector : numpy.ndarray
        Its vectorisation.
    orthogonality : float
        The realised maximum absolute entry of the tangent-orthogonality
        condition.
    unrestricted_signal : float
        The realised unrestricted signal norm.
    rank_feasibility : dict
        The rank quantities of the restricted tangent space and the ambient
        transition space at this dimension.
    seed_key : tuple
        The seed identity the direction was drawn from.
    """

    matrix: np.ndarray
    vector: np.ndarray
    orthogonality: float
    unrestricted_signal: float
    rank_feasibility: dict
    seed_key: tuple


def orthogonal_departure(truth: CanonicalTruth, sigma: float) -> OrthogonalDeparture:
    """Build the non-separable direction the restricted projection cannot see.

    A dense standard-normal matrix is projected onto the complement of
    the Kronecker tangent space under the population regressor metric,
    the condition that the departure does not move the least-squares
    Kronecker projection at first order. It is then normalised under the
    unrestricted signal metric, so a separable and a non-separable
    departure of the same nominal magnitude carry the same unrestricted
    first-order signal.

    Parameters
    ----------
    truth : CanonicalTruth
        The population.
    sigma : float
        Per-entry innovation standard deviation.

    Returns
    -------
    OrthogonalDeparture
        The direction with its realised orthogonality, norm and rank quantities.

    Raises
    ------
    DesignConstructionError
        If the projection or the normalisation misses its tolerance.
    """
    m, n = truth.m, truth.n
    size = m * n

    Gamma_0 = stationary_second_moment(truth.Phi, sigma)
    M = np.kron(Gamma_0, np.eye(size))
    J = kronecker_jacobian(truth)

    seed_key = ("g_perp", m, n, truth.persistence, truth.configuration)
    if truth.configuration == simcfg.REFERENCE_CONFIGURATION:
        seed_key = ("g_perp", m, n, truth.persistence)
    w = seeds.cell_rng(*seed_key).standard_normal((size, size)).flatten(order="F")

    MJ = M @ J
    coefficients = np.linalg.solve(J.T @ MJ, MJ.T @ w)
    p = w - J @ coefficients

    orthogonality = float(np.max(np.abs(MJ.T @ p)))
    if orthogonality > simcfg.G_PERP_ORTHOGONALITY_TOLERANCE:
        raise DesignConstructionError(
            f"orthogonal_departure: the tangent-orthogonality residual is "
            f"{orthogonality:.3e}, above {simcfg.G_PERP_ORTHOGONALITY_TOLERANCE:g}."
        )

    quadratic = float(p @ M @ p) / sigma**2
    if not np.isfinite(quadratic) or quadratic <= 0.0:
        raise DesignConstructionError(
            "orthogonal_departure: degenerate unrestricted signal norm."
        )

    p = p / np.sqrt(quadratic)
    realised = float(p @ M @ p) / sigma**2
    if abs(realised - 1.0) > simcfg.G_PERP_ORTHOGONALITY_TOLERANCE:
        raise DesignConstructionError(
            f"orthogonal_departure: unrestricted normalisation off by "
            f"{abs(realised - 1.0):.3e}."
        )

    K = size * size
    return OrthogonalDeparture(
        matrix=p.reshape(size, size, order="F"),
        vector=p,
        orthogonality=float(np.max(np.abs(MJ.T @ p))),
        unrestricted_signal=realised,
        rank_feasibility={
            "k": int(truth.k),
            "K": int(K),
            "jacobian_rank": int(np.linalg.matrix_rank(J)),
            "complement_dimension": int(K - truth.k),
        },
        seed_key=seed_key,
    )


def unrestricted_transition_path(
    truth: CanonicalTruth,
    departure: OrthogonalDeparture,
    path: str,
    magnitude: str,
    T: int,
) -> np.ndarray:
    """Build the non-separable transition path and check its stability.

    Parameters
    ----------
    truth : CanonicalTruth
        The population supplying the initial transition matrix.
    departure : OrthogonalDeparture
        The normalised non-separable direction.
    path : str
        Which scalar path drives the displacement.
    magnitude : str
        Which magnitude scales it.
    T : int
        Number of retained observations.

    Returns
    -------
    numpy.ndarray
        The ``(T, mn, mn)`` transition matrices.

    Raises
    ------
    PathVerificationError
        If any designed transition matrix reaches the stability boundary.
    """
    u = sample_fractions(T)
    q = path_function(path, u)
    scalar = departure_scalar(magnitude, simcfg.score_observations(T))

    Phi_path = truth.Phi[None, :, :] + (scalar * q)[:, None, None] * departure.matrix
    radii = np.array(
        [float(np.max(np.abs(np.linalg.eigvals(Phi)))) for Phi in Phi_path]
    )

    if np.any(radii >= simcfg.MAX_RADIUS):
        raise PathVerificationError(
            {
                "population_configuration": truth.configuration,
                "changing_block": "unrestricted_non_separable",
                "path": path,
                "departure_size": magnitude,
                "failed_condition": "stability_region",
                "min_spectral_radius": float(radii.min()),
                "max_spectral_radius": float(radii.max()),
                "min_sign_coordinate": truth.sign_coordinate,
                "population_seed_key": "|".join(str(p) for p in truth.seed_key),
                "direction_seed_key": "|".join(str(p) for p in departure.seed_key),
            }
        )

    return Phi_path

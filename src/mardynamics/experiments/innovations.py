"""The four innovation designs of the simulation study.

Each generator produces a complete stream of ``m x n`` innovation
matrices for one replication, burn-in included, from one generator.
Streams are drawn whole instead of observation by observation because
the conditionally heteroskedastic design is recursive and its burn-in
must run before the retained sample, and because a whole stream keeps a
replication's draw reproducible independently of how the recursion
consumes it.

All four have the same unconditional per-entry variance ``sigma^2``.
They differ in the way Assumption A1 allows the innovations to differ
from the independent isotropic Gaussian benchmark: contemporaneous
separable correlation, conditional heteroskedasticity through a common
scalar factor, and heavy tails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _apply_scale(draws: np.ndarray, sigma_scale: np.ndarray | None) -> np.ndarray:
    """Multiply a drawn stream by an optional per-observation scale path.

    The innovation-variance break design is the only one that needs it:
    the conditional mean stays constant while the innovation scale
    steps at a fixed sample fraction.

    Parameters
    ----------
    draws : numpy.ndarray
        Innovation stream of shape ``(n_obs, m, n)``.
    sigma_scale : numpy.ndarray or None
        Per-observation multiplier, or None.

    Returns
    -------
    numpy.ndarray
        The scaled stream.
    """
    if sigma_scale is None:
        return draws
    return draws * np.asarray(sigma_scale, dtype=float)[:, None, None]


@dataclass(frozen=True)
class IsotropicGaussian:
    """Independent Gaussian innovations with per-entry variance ``sigma^2``.

    The benchmark design, and the one under which Assumption A6 holds:
    the analytic proportionality between the score covariance and the
    Hessian can be used for the population signal metric.

    Attributes
    ----------
    sigma : float
        Per-entry standard deviation.
    """

    sigma: float
    name: str = "isotropic"

    def draw(
        self,
        n_obs: int,
        m: int,
        n: int,
        rng: np.random.Generator,
        sigma_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw one complete innovation stream.

        Parameters
        ----------
        n_obs : int
            Stream length, burn-in included.
        m, n : int
            Innovation matrix dimensions.
        rng : numpy.random.Generator
            Source of randomness.
        sigma_scale : numpy.ndarray or None, optional
            Per-observation scale multiplier.

        Returns
        -------
        numpy.ndarray
            The stream, shape ``(n_obs, m, n)``.
        """
        return _apply_scale(
            rng.normal(0.0, self.sigma, size=(n_obs, m, n)), sigma_scale
        )


@dataclass(frozen=True)
class SeparableCorrelated:
    """Homoskedastic Gaussian innovations with separable Toeplitz correlation.

    The covariance is ``sigma^2 (R_col (x) R_row)`` with
    ``R[i, j] = rho^|i - j|`` in each mode. Both correlation matrices have unit
    diagonal, so the per-entry variance is already ``sigma^2`` and no further
    rescaling is applied. Contemporaneous correlation across rows and columns
    breaks the isotropy of Assumption A6 while leaving the martingale-difference
    structure of Assumption A1 intact.

    Attributes
    ----------
    sigma : float
        Per-entry standard deviation.
    rho : float
        Toeplitz correlation parameter of both modes.
    """

    sigma: float
    rho: float
    name: str = "correlated"

    def _factor(self, size: int) -> np.ndarray:
        """Cholesky factor of the Toeplitz correlation matrix of one mode.

        Parameters
        ----------
        size : int
            Mode dimension.

        Returns
        -------
        numpy.ndarray
            The lower-triangular factor.
        """
        index = np.arange(size)
        correlation = self.rho ** np.abs(index[:, None] - index[None, :])
        return np.linalg.cholesky(correlation)

    def draw(
        self,
        n_obs: int,
        m: int,
        n: int,
        rng: np.random.Generator,
        sigma_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw one complete innovation stream.

        Parameters
        ----------
        n_obs : int
            Stream length, burn-in included.
        m, n : int
            Innovation matrix dimensions.
        rng : numpy.random.Generator
            Source of randomness.
        sigma_scale : numpy.ndarray or None, optional
            Per-observation scale multiplier.

        Returns
        -------
        numpy.ndarray
            The stream, shape ``(n_obs, m, n)``.
        """
        L_row = self._factor(m)
        L_col = self._factor(n)
        white = rng.normal(0.0, 1.0, size=(n_obs, m, n))
        correlated = self.sigma * np.einsum("ij,tjk,lk->til", L_row, white, L_col)
        return _apply_scale(correlated, sigma_scale)


@dataclass(frozen=True)
class CommonScalarGarch:
    """Conditionally heteroskedastic innovations with one common scalar factor.

    With ``vec(E_t) = sigma sqrt(h_t) z_t`` and standardised Gaussian shocks,
    the driver is the mean squared entry of the previous innovation matrix,
    ``s_t = ||vec(E_t)||^2 / (mn sigma^2)``, and

        ``h_t = (1 - alpha - beta) + alpha s_{t-1} + beta h_{t-1}``,

    started at ``h_1 = 1``. This gives ``E[h_t] = 1`` and unconditional per-entry
    variance ``sigma^2``. The innovations remain a martingale difference, so the
    design sits inside Assumption A1 and outside Assumption A6.

    Averaging the squared shock over ``mn`` entries removes most of its
    variance, so the realised conditional-variance factor moves far less
    than a univariate GARCH with the same coefficients would. The
    realised strength is measured and reported per replication, not
    assumed.

    Attributes
    ----------
    sigma : float
        Per-entry standard deviation.
    alpha, beta : float
        The GARCH coefficients.
    h_initial : float
        The conditional variance factor of the first drawn observation.
    """

    sigma: float
    alpha: float
    beta: float
    h_initial: float = 1.0
    name: str = "garch"

    def draw(
        self,
        n_obs: int,
        m: int,
        n: int,
        rng: np.random.Generator,
        sigma_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw one complete innovation stream, recursion and burn-in included.

        Parameters
        ----------
        n_obs : int
            Stream length, burn-in included.
        m, n : int
            Innovation matrix dimensions.
        rng : numpy.random.Generator
            Source of randomness.
        sigma_scale : numpy.ndarray or None, optional
            Per-observation scale multiplier.

        Returns
        -------
        numpy.ndarray
            The stream, shape ``(n_obs, m, n)``.
        """
        z = rng.normal(0.0, 1.0, size=(n_obs, m, n))
        draws = np.empty((n_obs, m, n))

        omega = 1.0 - self.alpha - self.beta
        h_t = float(self.h_initial)
        s_prev = 1.0

        for t in range(n_obs):
            if t > 0:
                h_t = omega + self.alpha * s_prev + self.beta * h_t
            draws[t] = self.sigma * np.sqrt(h_t) * z[t]
            s_prev = float((draws[t] ** 2).sum() / (m * n * self.sigma**2))

        return _apply_scale(draws, sigma_scale)


def garch_strength(
    stream: np.ndarray, sigma: float, m: int, n: int, alpha: float, beta: float
) -> dict:
    """Measure the realised strength of a drawn GARCH innovation stream.

    The realised conditional-variance movement is reported, not assumed,
    because averaging the squared shock over ``mn`` entries damps it.
    The factor is recovered from the stream itself, so the measurement
    needs no state from the generator.

    Parameters
    ----------
    stream : numpy.ndarray
        Innovation stream of shape ``(n_obs, m, n)``.
    sigma : float
        Per-entry standard deviation.
    m, n : int
        Innovation matrix dimensions.
    alpha, beta : float
        The GARCH coefficients, replayed to recover the factor path.

    Returns
    -------
    dict
        Keys ``h_std``, ``h_q05``, ``h_q95`` and
        ``standardised_excess_kurtosis``.
    """
    stream = np.asarray(stream, dtype=float)
    n_obs = stream.shape[0]

    s = (stream**2).sum(axis=(1, 2)) / (m * n * sigma**2)
    omega = 1.0 - alpha - beta

    h = np.empty(n_obs)
    h[0] = 1.0
    for t in range(1, n_obs):
        h[t] = omega + alpha * s[t - 1] + beta * h[t - 1]

    standardised = stream / (sigma * np.sqrt(h)[:, None, None])
    flat = standardised.reshape(-1)
    centred = flat - flat.mean()
    variance = float((centred**2).mean())
    kurtosis = float((centred**4).mean() / variance**2) if variance > 0 else np.nan

    return {
        "h_std": float(np.std(h)),
        "h_q05": float(np.quantile(h, 0.05)),
        "h_q95": float(np.quantile(h, 0.95)),
        "standardised_excess_kurtosis": kurtosis - 3.0,
    }


@dataclass(frozen=True)
class StandardisedStudentT:
    """Independent standardised Student-t innovations with heavy tails.

    Each entry is an independent ``t(df)`` draw rescaled by
    ``sqrt((df - 2) / df)``, so the per-entry variance is ``sigma^2`` while the
    fourth moment is far larger than the Gaussian one.

    Attributes
    ----------
    sigma : float
        Per-entry standard deviation.
    df : int
        Degrees of freedom; must exceed four for the moment conditions of
        Assumption A1.
    """

    sigma: float
    df: int
    name: str = "student_t"

    def draw(
        self,
        n_obs: int,
        m: int,
        n: int,
        rng: np.random.Generator,
        sigma_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw one complete innovation stream.

        Parameters
        ----------
        n_obs : int
            Stream length, burn-in included.
        m, n : int
            Innovation matrix dimensions.
        rng : numpy.random.Generator
            Source of randomness.
        sigma_scale : numpy.ndarray or None, optional
            Per-observation scale multiplier.

        Returns
        -------
        numpy.ndarray
            The stream, shape ``(n_obs, m, n)``.

        Raises
        ------
        ValueError
            If the degrees of freedom do not give a finite variance.
        """
        if self.df <= 2:
            raise ValueError("Student-t innovations need df > 2 for finite variance.")
        scale = self.sigma * np.sqrt((self.df - 2) / self.df)
        return _apply_scale(
            rng.standard_t(self.df, size=(n_obs, m, n)) * scale, sigma_scale
        )


def build_innovation(design: str, sigma: float, **overrides):
    """Build the innovation generator of a named design.

    Parameters
    ----------
    design : {"isotropic", "correlated", "garch", "student_t"}
        Which innovation design.
    sigma : float
        Per-entry standard deviation.
    **overrides
        Design constants, defaulting to the configured values.

    Returns
    -------
    object
        The generator, with a ``draw`` method and a ``name``.

    Raises
    ------
    ValueError
        For an unknown design name.
    """
    import mardynamics.experiments.sim_config as simcfg

    if design == "isotropic":
        return IsotropicGaussian(sigma=sigma)
    if design == "correlated":
        return SeparableCorrelated(
            sigma=sigma, rho=overrides.get("rho", simcfg.TOEPLITZ_RHO)
        )
    if design == "garch":
        return CommonScalarGarch(
            sigma=sigma,
            alpha=overrides.get("alpha", simcfg.GARCH_ALPHA),
            beta=overrides.get("beta", simcfg.GARCH_BETA),
            h_initial=overrides.get("h_initial", simcfg.GARCH_H_INITIAL),
        )
    if design == "student_t":
        return StandardisedStudentT(
            sigma=sigma, df=overrides.get("df", simcfg.STUDENT_T_DF)
        )
    raise ValueError(f"unknown innovation design={design!r}")

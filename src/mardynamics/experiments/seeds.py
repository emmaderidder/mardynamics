"""Reproducible per-cell random generators for the simulation study.

Every random draw in the study derives from the single ``sim_config.SIM_MASTER_SEED``
together with the *logical identity* of the draw (for example
``("null", 5, 3, 400, "isotropic")``). The identity is hashed with BLAKE2b into a
128-bit ``SeedSequence`` spawn key, never with Python's ``hash``, which is
randomised per process and would break reproducibility between runs.

Because the stream depends on the cell identity and not on execution order, a
rerun reproduces the same numbers whatever the worker count is, and cells can be
recomputed individually.
"""

from __future__ import annotations

import hashlib

import numpy as np

import mardynamics.experiments.sim_config as simcfg


def _key_token(part: object) -> str:
    """Render one key component as a platform-stable string.

    Floats are formatted to ten significant digits, so ``0.5`` and
    ``0.5000000000000001`` collapse to the same token (grid values
    arriving from different arithmetic must not silently produce
    different streams), while genuinely different magnitudes stay
    distinct.

    Parameters
    ----------
    part : object
        One component of a seed key, for example a design name, a dimension or
        a grid value.

    Returns
    -------
    str
        The token for this component: a ten-significant-digit representation
        for floats, otherwise ``str(part)``.
    """
    if isinstance(part, float):
        return f"{part:.10g}"
    return str(part)


def _seed_key_words(*parts) -> tuple[int, ...]:
    """Stable 128-bit ``SeedSequence`` spawn key from a draw's logical identity.

    The key is a BLAKE2b digest of the joined key components, split into four
    32-bit words. It is a cryptographic digest of the *meaning* of the draw (for
    example ``("null", 4, 800, "gaussian")``), never Python's ``hash``, which is
    randomised per process and would break reproducibility across runs.

    Parameters
    ----------
    *parts
        Components identifying the draw: a role tag plus the cell coordinates.

    Returns
    -------
    tuple of int
        Four 32-bit words for ``numpy.random.SeedSequence(spawn_key=...)``.
    """
    digest = hashlib.blake2b(
        "|".join(_key_token(p) for p in parts).encode("utf-8"), digest_size=16
    ).digest()
    return tuple(int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4))


def seed_key_words(*parts) -> tuple[int, ...]:
    """Public view of the spawn key of one logical draw.

    The seed registry records this so two seed identities can be shown
    to be different streams, not assumed to be. Because the key is a
    digest of the draw's meaning, "the seed ranges do not overlap" is
    the statement that two different identities do not produce the
    same key.

    Parameters
    ----------
    *parts
        Components identifying the draw.

    Returns
    -------
    tuple of int
        The four 32-bit spawn-key words.
    """
    return _seed_key_words(*parts)


def _cell_seed_sequence(*parts) -> np.random.SeedSequence:
    """``SeedSequence`` for one logical draw: master entropy + identity spawn key.

    Two calls with the same ``parts`` return the same sequence, in any process,
    at any point in a run. Two calls with different ``parts`` are independent
    streams. This is what preserves common random numbers (competing
    statistics that must see the same data pass the same key) while
    keeping unrelated cells independent.

    Parameters
    ----------
    *parts
        Components identifying the draw (see :func:`_seed_key_words`).

    Returns
    -------
    numpy.random.SeedSequence
        Seeded from :data:`simcfg.SIM_MASTER_SEED` with the identity-derived spawn key.
    """
    return np.random.SeedSequence(
        entropy=simcfg.SIM_MASTER_SEED, spawn_key=_seed_key_words(*parts)
    )


def cell_rngs(n: int, *parts) -> list[np.random.Generator]:
    """Build ``n`` independent generators for one logical cell.

    The ``i``-th generator belongs to replication ``i`` of the cell identified by
    ``parts``, regardless of how the replications are later distributed over
    workers, so a run at one worker and ten workers draws
    identical data.

    Parameters
    ----------
    n : int
        Number of replications.
    *parts
        Components identifying the cell (see :func:`_seed_key_words`).

    Returns
    -------
    list of numpy.random.Generator
        The ``n`` replication generators, in replication order.
    """
    return [cell_rng(*parts, index) for index in range(n)]


def cell_rng(*parts) -> np.random.Generator:
    """Single generator for a one-off draw (a truth construction, a long path).

    Parameters
    ----------
    *parts
        Components identifying the draw (see :func:`_seed_key_words`).

    Returns
    -------
    numpy.random.Generator
        The seeded generator.
    """
    return np.random.default_rng(_cell_seed_sequence(*parts))

"""The dispatch from a cell kind to its routines.

Each module implements one branch of the design and exposes a
``build_context`` that constructs everything a cell needs once, together
with one executor per cell kind it owns. The dispatch table below is the
single place that maps a registry cell kind to those routines.

Sections
--------
``calibration``
    Null calibration, the fixed-regressor wild-bootstrap comparison,
    canonical recovery under stability and the multiple-start check.
``detection``
    Detection, conditional mode-specific attribution and influence paths. It
    also runs the population-and-direction sensitivity, which differs
    from detection only in which population and which directions it
    plants.
``matrix_restriction``
    The value and the limits of the matrix restriction.

The design layer has no executable cell of its own. Its outputs (the
population matrices, the change directions, the signal measures, the
realised departure sizes, the path verification and the seed registry)
are produced by the design audit before any replication runs.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

# ``kind -> (module, context builder, executor)``. The module is imported
# only when a cell of that kind is built or run, so a worker loads only
# the section it executes.
_ROUTES = {
    "null_calibration": ("calibration", "build_context", "execute", "simulate"),
    "wild_bootstrap": (
        "calibration", "build_context", "execute_bootstrap", "simulate",
    ),
    "multiple_start": (
        "calibration", "build_context", "execute_multiple_start", "simulate",
    ),
    "detection": ("detection", "build_context", "execute", "simulate"),
    "matrix_restriction": (
        "matrix_restriction", "build_context", "execute", "simulate",
    ),
}


def _resolve(kind: str, position: int) -> Callable:
    """Import the section module of a kind and return one of its two routines.

    Parameters
    ----------
    kind : str
        The cell kind.
    position : int
        1 for the context builder, 2 for the executor, 3 for the simulator.

    Returns
    -------
    callable
        The routine.

    Raises
    ------
    KeyError
        For an unknown kind.
    """
    if kind not in _ROUTES:
        raise KeyError(f"no section route for cell kind {kind!r}")

    module_name, *attributes = _ROUTES[kind]
    module = import_module(f"mardynamics.experiments.designs.{module_name}")
    return getattr(module, attributes[position - 1])


def simulator_for(kind: str) -> Callable:
    """Return the dataset simulator of a cell kind.

    The simulator is the half of a replication that draws the data, without the
    fit or the statistics. Exposing it separately is what lets a stored
    replication be regenerated exactly from its seed.

    Parameters
    ----------
    kind : str
        The cell kind.

    Returns
    -------
    callable
        The simulator.

    Raises
    ------
    KeyError
        For an unknown kind.
    """
    return _resolve(kind, 3)


def context_for(kind: str) -> Callable:
    """Return the context builder of a cell kind.

    Parameters
    ----------
    kind : str
        The cell kind.

    Returns
    -------
    callable
        The builder.

    Raises
    ------
    KeyError
        For an unknown kind.
    """
    return _resolve(kind, 1)


def executor_for(kind: str) -> Callable:
    """Return the replication executor of a cell kind.

    Parameters
    ----------
    kind : str
        The cell kind.

    Returns
    -------
    callable
        The executor.

    Raises
    ------
    KeyError
        For an unknown kind.
    """
    return _resolve(kind, 2)

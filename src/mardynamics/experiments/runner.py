"""The execution engine: threading, chunked checkpoints and failure accounting.

One routine runs every cell of the registry, and the rules it enforces
are about running, not about design.

* Threading is set once, before NumPy is imported anywhere, to one
  thread per process, and there is one parallelisation layer: process
  workers over replications. ``threadpoolctl`` is used as a second
  check at run time.
* A replication's data depends only on its dataset stream and its
  replication index, so a run at one worker and a run at ten produce
  identical records.
* Work is checkpointed in contiguous chunks of replications. A chunk is
  reused only when its marker carries the cell's current configuration
  hash, so an interrupted run resumes and a changed design does not.
* A failed replication is recorded and not replaced. The attempted
  count stays the requested one, so successful and failed always sum to
  it and no denominator is quietly repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import traceback

import numpy as np
import pandas as pd

import mardynamics.experiments.seeds as seeds
import mardynamics.experiments.sim_config as simcfg
import mardynamics.experiments.sim_io as simio

# Replications per checkpoint chunk. An internal batching choice: it
# changes how often work is saved, not what is computed.
CHUNK_SIZE = 50


def configure_threads() -> None:
    """Pin every numerical library to one thread, before NumPy is imported.

    Must be called at the very top of the launcher. Worker processes inherit the
    environment, so the same limits apply there.
    """
    for name in simcfg.THREAD_ENV_VARS:
        os.environ[name] = "1"


# Kept alive at module scope: the threadpoolctl controller releases its
# limits when it is garbage collected, so the process-wide limit must
# outlive the call that set it.
_THREAD_LIMITER = None


def enforce_thread_limits() -> None:
    """Pin the numerical libraries to one thread, at load time and at run time.

    The environment variables only bind when a library is first loaded,
    so a process that imported NumPy before reading them keeps whatever
    thread pool it started with. ``threadpoolctl`` is a second control:
    it reaches the already-loaded pools and limits them directly. Both
    are applied, because the launcher can guarantee the first and only
    the second survives an interpreter that was started some other way.
    """
    global _THREAD_LIMITER

    configure_threads()

    if _THREAD_LIMITER is not None:
        return

    try:
        import threadpoolctl
    except ImportError:
        return

    _THREAD_LIMITER = threadpoolctl.threadpool_limits(limits=1)


@dataclass(frozen=True)
class CellOutcome:
    """The accounting of one executed cell.

    Attributes
    ----------
    cell_id : str
        Cell identity.
    attempted, successful, failed : int
        Replication counts. They always satisfy
        ``attempted = successful + failed``: a failure is recorded, not
        replaced.
    seconds : float
        Wall time spent on the replications actually executed in this run.
    reused_chunks : int
        Chunks read from an existing checkpoint instead of recomputed.
    source : str
        ``"new"`` when the cell was executed in this run, or
        ``"reused:<run_id>"`` when its records were carried in from a finished
        earlier run whose reuse key matched.
    """

    cell_id: str
    attempted: int
    successful: int
    failed: int
    seconds: float
    reused_chunks: int
    source: str = "new"


def _chunk_ranges(n_rep: int, chunk_size: int = CHUNK_SIZE) -> list[tuple[int, int]]:
    """Split ``n_rep`` replications into contiguous half-open chunks.

    Parameters
    ----------
    n_rep : int
        Number of replications.
    chunk_size : int, optional
        Replications per chunk.

    Returns
    -------
    list of tuple of int
        The ``(start, stop)`` ranges.
    """
    return [
        (start, min(start + chunk_size, n_rep)) for start in range(0, n_rep, chunk_size)
    ]


def execute_replication(cell, context: dict, replication_id: int) -> list[dict]:
    """Run one replication of a cell and return its records.

    Any exception is caught here and turned into one failure record, so
    a single bad replication cannot end a cell and cannot be silently
    replaced by a good one.

    Parameters
    ----------
    cell : Cell
        The cell.
    context : dict
        The per-cell objects built once before dispatch.
    replication_id : int
        Zero-based replication index inside the cell.

    Returns
    -------
    list of dict
        The replication's records, each tagged with its identity and status.
    """
    from mardynamics.experiments.designs import executor_for

    enforce_thread_limits()

    started = datetime.now(timezone.utc)
    try:
        rows = executor_for(cell.kind)(cell, context, replication_id)
        status, failure_type, message = "ok", "", ""
    except (
        Exception
    ) as exc:  # noqa: BLE001 - every failure mode is recorded, not raised
        rows = [{}]
        status = "failed"
        failure_type = type(exc).__name__
        message = f"{exc}"[:500]
        if os.environ.get("MAR_SIM_TRACEBACKS"):
            message = traceback.format_exc()[-2000:]

    seconds = (datetime.now(timezone.utc) - started).total_seconds()

    # The spawn key of the replication's dataset stream, written into
    # the record itself. The stream is derivable from the registry, but
    # a record that carries its own seed can be regenerated from the
    # parquet alone, which is what makes a targeted regeneration
    # possible without the registry.
    spawn_key = "-".join(
        str(word) for word in seeds.seed_key_words(*cell.dataset_key, replication_id)
    )

    tagged = []
    for index, row in enumerate(rows):
        record = {
            "part": cell.part,
            "cell_id": cell.cell_id,
            "kind": cell.kind,
            "replication_id": int(replication_id),
            "record_index": int(index),
            "status": status,
            "failure_type": failure_type,
            "failure_message": message,
            "seconds": seconds if index == 0 else 0.0,
            "dataset_spawn_key": spawn_key,
        }
        record.update(row)
        tagged.append(record)
    return tagged


def run_cell(
    cell,
    paths: simio.RunPaths,
    n_rep: int,
    workers: int,
    context: dict | None = None,
    output_dir=None,
    run_id: str | None = None,
) -> CellOutcome:
    """Execute one cell with checkpointing, resuming or importing what it can.

    Two kinds of existing output are honoured. Chunks already in this
    run directory are resumed, which is what makes an interrupted run
    continue. Chunks in a finished earlier run whose reuse key matches
    are imported. When nothing matches, the cell is rerun.

    Parameters
    ----------
    cell : Cell
        The cell to run.
    paths : RunPaths
        The run layout.
    n_rep : int
        Replications to run.
    workers : int
        Process workers.
    context : dict or None, optional
        Pre-built cell context; built here when omitted.
    output_dir : str or pathlib.Path or None, optional
        Project output root. Passing it enables cross-run reuse.
    run_id : str or None, optional
        The run being written, which is not its own reuse source.

    Returns
    -------
    CellOutcome
        The accounting of this cell.
    """
    from joblib import Parallel, delayed
    from mardynamics.experiments.designs import context_for

    source = "new"
    if output_dir is not None and run_id is not None:
        imported = simio.import_reusable_chunks(paths, cell, output_dir, run_id)
        if imported is not None:
            source = f"reused:{imported}"

    if context is None:
        context = context_for(cell.kind)(cell)

    started = datetime.now(timezone.utc)
    reused = 0

    for start, stop in _chunk_ranges(n_rep):
        if simio.chunk_is_complete(paths, cell, start, stop):
            reused += 1
            continue

        if workers <= 1:
            batches = [
                execute_replication(cell, context, rid) for rid in range(start, stop)
            ]
        else:
            batches = Parallel(n_jobs=workers, backend="loky")(
                delayed(execute_replication)(cell, context, rid)
                for rid in range(start, stop)
            )

        frame = pd.DataFrame([row for batch in batches for row in batch])
        simio.write_chunk(paths, cell, start, stop, frame)

    seconds = (datetime.now(timezone.utc) - started).total_seconds()

    records = simio.read_cell_records(paths, cell)
    if records.empty:
        return CellOutcome(cell.cell_id, n_rep, 0, n_rep, seconds, reused, source)

    first_rows = records[records["record_index"] == 0]
    successful = int((first_rows["status"] == "ok").sum())
    failed = int((first_rows["status"] == "failed").sum())

    return CellOutcome(
        cell_id=cell.cell_id,
        attempted=int(len(first_rows)),
        successful=successful,
        failed=failed,
        seconds=seconds,
        reused_chunks=reused,
        source=source,
    )


def replications_for_mode(cell, mode: str) -> int:
    """Return how many outer replications a cell runs in a given mode.

    Smoke and preview run the capped counts their registry already carries,
    because in those modes the caps are part of the cell identity and therefore
    of its checkpoints. Production runs the configured count.

    Parameters
    ----------
    cell : Cell
        The cell.
    mode : str
        Run mode.

    Returns
    -------
    int
        The replication count.

    Raises
    ------
    ValueError
        For an unknown mode.
    """
    if mode in ("smoke", "preview", "production"):
        return cell.n_rep
    raise ValueError(f"unknown run mode={mode!r}")


def warm_limit_laws(cache_dir, dimensions: list[int]) -> list[dict]:
    """Simulate and cache every limit law the run needs, in this process.

    The disk cache is written lazily on first use, so warming it before the
    workers start prevents ten processes from simulating the same law at once
    and racing on the same file. The laws live outside the run directory
    because they are shared objects: the empirical pipeline and the simulation
    must read identical critical values for the same dimension.

    Parameters
    ----------
    cache_dir : str or pathlib.Path
        The shared limit-law cache directory.
    dimensions : list of int
        The tested dimensions to warm.

    Returns
    -------
    list of dict
        One record per dimension with its draws and 95% quantile.
    """
    import mardynamics.core.structural_tests as structural_tests

    records = []
    for k in sorted(set(int(k) for k in dimensions)):
        draws = structural_tests.load_limit_law(k, cache_dir)
        records.append(
            {
                "k": k,
                "n_draws": int(draws.size),
                "q95": float(np.quantile(draws, 0.95)),
                "mean": float(draws.mean()),
                "k_over_6": k / 6.0,
            }
        )
    return records


def required_limit_law_dimensions(cells: list) -> list[int]:
    """Every tested dimension the registry will ask a limit law for.

    Parameters
    ----------
    cells : list
        The registry.

    Returns
    -------
    list of int
        The dimensions, ascending.
    """
    import mardynamics.core.structural_tests as structural_tests

    dimensions: set[int] = set()
    for cell in cells:
        if any(name.startswith("nyblom") for name in cell.statistics):
            dimensions.update(
                {
                    cell.m * cell.m + cell.n * cell.n - 1,
                    cell.m * cell.m - 1,
                    cell.n * cell.n - 1,
                    1,
                }
            )
        if "nyblom_var" in cell.statistics:
            dimensions.add(structural_tests.var_dimension(cell.m, cell.n))
    return sorted(dimensions)


def failure_table(paths: simio.RunPaths, cells: list) -> pd.DataFrame:
    """Collect every failed replication of the run.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    cells : list
        The registry.

    Returns
    -------
    pandas.DataFrame
        One row per failed replication.
    """
    rows = []
    for cell in cells:
        records = simio.read_cell_records(paths, cell)
        if records.empty or "status" not in records:
            continue
        failed = records[
            (records["status"] == "failed") & (records["record_index"] == 0)
        ]
        for _, row in failed.iterrows():
            rows.append(
                {
                    "part": cell.part,
                    "cell_id": cell.cell_id,
                    "replication_id": int(row["replication_id"]),
                    "failure_type": row["failure_type"],
                    "failure_message": row["failure_message"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "part",
            "cell_id",
            "replication_id",
            "failure_type",
            "failure_message",
        ],
    )


def replication_telemetry(paths: simio.RunPaths, cells: list) -> pd.DataFrame:
    """Collect the per-replication telemetry of the whole run.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    cells : list
        The registry.

    Returns
    -------
    pandas.DataFrame
        One row per replication with its identity, status and timing.
    """
    frames = []
    for cell in cells:
        records = simio.read_cell_records(paths, cell)
        if records.empty:
            continue
        first = records[records["record_index"] == 0].copy()
        keep = [
            c
            for c in (
                "part",
                "cell_id",
                "kind",
                "replication_id",
                "status",
                "failure_type",
                "seconds",
                "dataset_spawn_key",
                # The numerical-reliability record, in a fixed order.
                "converged",
                "n_iterations",
                "initial_objective",
                "final_objective",
                "foc_norm_canonical",
                "stability_contracted",
                "n_stability_contractions",
                "covariance_regularised",
                "pseudoinverse_used",
                "min_eigenvalue_H",
                "cond_H",
                "min_eigenvalue_Omega",
                "cond_Omega",
                "hessian_inversion_failed",
                "score_covariance_inversion_failed",
                "rho_fitted",
                "stability_binding",
                "projection_regressor_rank",
                "projection_minimum_norm",
            )
            if c in first
        ]
        frames.append(first[keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

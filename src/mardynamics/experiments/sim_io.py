"""The output layout: run directory, manifest, checkpoints and telemetry.

The layout is::

    outputs/experiments/<run_id>/
        simulation_manifest.json
        checkpoints/<part>/<cell_id>/chunk_<start>_<stop>.parquet + .done
        tables/          tidy CSVs, one row per cell/statistic/level
        figures/         only the pre-specified figures
        telemetry/       replications.parquet, failures.csv, runtime files

Four rules are enforced here instead of trusted to callers. Every write
goes to a temporary file and is renamed atomically, so a chunk file is
either absent or complete. A chunk counts as finished only when its
marker exists and records the configuration hash the reader expects, so
a resumed run cannot mix two designs. A run directory that already
holds a finished manifest is not overwritten. And every table is
tagged with the output group it belongs to and registered in an index,
so the run can check its own completeness instead of a reader having
to.

Output reuse across runs is keyed on the cell's ``reuse_key``, which
covers the data-generating process, the estimator controls, the
statistics, the algorithm, the seeds, the counts and the record schema
version. A stored chunk is carried into a new run only when that key
matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import mardynamics.experiments.sim_config as simcfg

# The simulation study writes below this directory.
SIM_SUBDIR = "experiments"

# The manifest file name required by the run layout.
MANIFEST_NAME = "simulation_manifest.json"


def make_run_id(mode: str, now: datetime | None = None) -> str:
    """Build a run identifier from the run mode and the wall-clock start.

    Parameters
    ----------
    mode : str
        Run mode, for example ``"smoke"`` or ``"production"``.
    now : datetime.datetime or None, optional
        Start time; the current UTC time when omitted.

    Returns
    -------
    str
        The run identifier.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{mode}_{stamp}"


@dataclass
class RunPaths:
    """The directories and files of one run.

    Attributes
    ----------
    root : pathlib.Path
        The run directory.
    mode : str
        The run mode this layout belongs to. Every table written through
        :func:`save_table` carries it in its first column, so an artefact states
        on its own face which kind of run produced it.
    index : list of dict
        One entry per written table: its file name, its output group and its
        row count. It becomes ``output_index.csv``.
    """

    root: Path
    mode: str = "unknown"
    index: list = field(default_factory=list)

    @property
    def checkpoints(self) -> Path:
        """Checkpoint root."""
        return self.root / "checkpoints"

    @property
    def tables(self) -> Path:
        """Tidy CSV output directory."""
        return self.root / "tables"

    @property
    def figures(self) -> Path:
        """Figure output directory."""
        return self.root / "figures"

    @property
    def telemetry(self) -> Path:
        """Telemetry directory."""
        return self.root / "telemetry"

    @property
    def manifest(self) -> Path:
        """Manifest path."""
        return self.root / MANIFEST_NAME

    def create(self) -> "RunPaths":
        """Create every directory of the layout.

        Returns
        -------
        RunPaths
            This object, for chaining.
        """
        for directory in (
            self.root,
            self.checkpoints,
            self.tables,
            self.figures,
            self.telemetry,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self


def run_paths(output_dir: str | Path, run_id: str, mode: str = "unknown") -> RunPaths:
    """Return the paths of one run, creating the directories.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Project output root.
    run_id : str
        Run identifier.
    mode : str, optional
        The run mode, stamped into every table this layout writes.

    Returns
    -------
    RunPaths
        The created layout.
    """
    return RunPaths(Path(output_dir) / SIM_SUBDIR / run_id, mode).create()


# =========================================================
# Atomic writes
# =========================================================
def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes through a temporary file and an atomic rename.

    Parameters
    ----------
    path : pathlib.Path
        Destination.
    payload : bytes
        Content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text through a temporary file and an atomic rename.

    Parameters
    ----------
    path : pathlib.Path
        Destination.
    text : str
        Content.
    """
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    """Write a DataFrame to parquet or CSV atomically, chosen by suffix.

    Parameters
    ----------
    path : pathlib.Path
        Destination, ``.parquet`` or ``.csv``.
    frame : pandas.DataFrame
        The frame to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


# =========================================================
# Manifest
# =========================================================
# Everything under these roots identifies the source of a run: the
# whole source tree, including the configuration and the table builders.
SOURCE_ROOTS = ("src/mardynamics", "scripts")


def source_identity(project_root: Path | None = None) -> dict:
    """Identify the source that produced a run, with or without a repository.

    Reproducibility requires more than "unavailable" on its own. This
    returns the commit when there is a repository to read one from, and
    when there is not it says plainly why and supplies a content digest
    of the whole source tree in its place. A tree digest is in
    one respect a stronger identifier than a commit: a commit names a state the
    working tree may have since departed from, whereas the digest is computed
    from the files that actually ran.

    Parameters
    ----------
    project_root : pathlib.Path or None, optional
        The project root; inferred from this module's location when omitted.

    Returns
    -------
    dict
        Keys ``git_commit`` (or None), ``git_status`` explaining its absence,
        ``source_tree_digest``, ``source_file_count`` and ``source_roots``.
    """
    root = (
        Path(__file__).resolve().parents[3] if project_root is None
        else Path(project_root)
    )

    commit, status = None, "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
            status = "read from the repository"
        else:
            message = (result.stderr or "").strip().splitlines()
            detail = message[0] if message else "no commit reported"
            status = (
                "no git repository: " + detail + ". A tree digest of the source "
                "that ran is recorded below in its place."
            )
    except FileNotFoundError:
        status = (
            "git is not installed on this machine. A tree digest of the source "
            "that ran is recorded below in its place."
        )
    except Exception as exc:  # noqa: BLE001 - the reason is recorded, not raised
        status = f"git could not be queried ({type(exc).__name__}). A tree digest "
        status += "of the source that ran is recorded below in its place."

    digest = hashlib.sha256()
    count = 0
    for relative in SOURCE_ROOTS:
        base = root / relative
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
            count += 1

    return {
        "git_commit": commit,
        "git_status": status,
        "source_tree_digest": digest.hexdigest(),
        "source_file_count": count,
        "source_roots": list(SOURCE_ROOTS),
    }


def config_snapshot() -> dict:
    """Collect every run-wide configuration value a result can depend on.

    Returns
    -------
    dict
        A JSON-serialisable snapshot of the simulation configuration.
    """
    return {
        "master_seed": simcfg.SIM_MASTER_SEED,
        "sigma": simcfg.SIGMA,
        "sigma_squared": simcfg.SIGMA_SQUARED,
        "burn_in": simcfg.BURN_IN,
        "nominal_levels": list(simcfg.NOMINAL_LEVELS),
        "primary_level": simcfg.PRIMARY_LEVEL,
        "dim_grid": [list(pair) for pair in simcfg.DIM_GRID],
        "T_grid": list(simcfg.T_GRID),
        "reference_population": {
            "m": simcfg.BASE_M,
            "n": simcfg.BASE_N,
            "T": simcfg.BASE_T,
            "persistence": simcfg.BASE_PERSISTENCE,
            "description": "fixed reference population",
        },
        "high_persistence": simcfg.HIGH_PERSISTENCE,
        "population_configurations": list(simcfg.POPULATION_CONFIGURATIONS),
        "population_config_seeds": dict(simcfg.POPULATION_CONFIG_SEEDS),
        "paths": list(simcfg.PATHS),
        "break_fraction": simcfg.BREAK_FRACTION,
        "magnitudes": list(simcfg.MAGNITUDES),
        "local_scalars": dict(simcfg.LOCAL_MAGNITUDE_SCALARS),
        "large_magnitude": simcfg.LARGE_MAGNITUDE,
        "calibration_reps": simcfg.CALIBRATION_REPS,
        "power_reps": simcfg.POWER_REPS,
        "matrix_value_reps": simcfg.MATRIX_VALUE_REPS,
        "design_sensitivity": {
            "reps": simcfg.DESIGN_SENSITIVITY_REPS,
            "path": simcfg.DESIGN_SENSITIVITY_PATH,
            "magnitude": simcfg.DESIGN_SENSITIVITY_MAGNITUDE,
        },
        "multiple_start": {
            "reps": simcfg.MULTIPLE_START_REPS,
            "n_perturbed": simcfg.MULTIPLE_START_PERTURBED,
            "perturbation": simcfg.MULTIPLE_START_PERTURBATION,
            "objective_tolerance": simcfg.MULTIPLE_START_OBJECTIVE_TOLERANCE,
        },
        "bootstrap": {
            "outer_reps": simcfg.BOOTSTRAP_OUTER_REPS,
            "draws": simcfg.BOOTSTRAP_DRAWS,
            "kind": "fixed_regressor_wild_rademacher",
        },
        "innovation_designs": list(simcfg.INNOVATION_DESIGNS),
        "toeplitz_rho": simcfg.TOEPLITZ_RHO,
        "garch": [simcfg.GARCH_ALPHA, simcfg.GARCH_BETA, simcfg.GARCH_H_INITIAL],
        "student_t_df": simcfg.STUDENT_T_DF,
        "tolerances": {
            "verification": simcfg.VERIFICATION_TOLERANCE,
            "global_signal_normalisation": simcfg.G_NORMALISATION_TOLERANCE,
            "shape_norm": simcfg.SHAPE_NORM_TOLERANCE,
            "tangent_orthogonality": simcfg.TANGENT_ORTHOGONALITY_TOLERANCE,
            "unrestricted_orthogonality": simcfg.G_PERP_ORTHOGONALITY_TOLERANCE,
            "moment_check": simcfg.G_MOMENT_CHECK_TOLERANCE,
            "classification_sum": simcfg.CLASSIFICATION_SUM_TOLERANCE,
            "static_als": simcfg.STATIC_ALS_TOL,
        },
        "estimator_controls": {
            "static_als": [
                simcfg.STATIC_ALS_MAX_ITER,
                simcfg.STATIC_ALS_TOL,
                simcfg.STATIC_ALS_STOPPING_RULE,
            ],
            "stability": [simcfg.ENFORCE_STABILITY, simcfg.MAX_RADIUS],
        },
        "output_schema_version": simcfg.OUTPUT_SCHEMA_VERSION,
    }


def write_manifest(
    paths: RunPaths,
    run_id: str,
    mode: str,
    profile: str,
    workers: int,
    cells: list,
    started: str,
    finished: str | None = None,
    reuse: dict | None = None,
    seed_registry: pd.DataFrame | None = None,
    extra: dict | None = None,
) -> dict:
    """Write the run manifest atomically.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    run_id : str
        Run identifier.
    mode : str
        Run mode.
    profile : str
        Runtime profile.
    workers : int
        Number of process workers.
    cells : list
        The registry, recorded by identity, configuration hash and reuse key.
    started : str
        ISO start timestamp.
    finished : str or None, optional
        ISO end timestamp, when the run has completed.
    reuse : dict or None, optional
        Per-cell record of whether its output was newly run or validly reused.
    seed_registry : pandas.DataFrame or None, optional
        The seed registry, recorded in the manifest by role.
    extra : dict or None, optional
        Further manifest entries, for example the output index.

    Returns
    -------
    dict
        The manifest that was written.
    """
    manifest = {
        "run_id": run_id,
        "mode": mode,
        "profile": profile,
        "workers": workers,
        "started": started,
        "finished": finished,
        "config": config_snapshot(),
        "cells": {
            cell.cell_id: {
                "part": cell.part,
                "kind": cell.kind,
                "config_hash": cell.config_hash,
                "reuse_key": cell.reuse_key,
                "n_rep": cell.n_rep,
                "n_resample": cell.n_resample,
                "dataset_key": list(cell.dataset_key),
                "source": (reuse or {}).get(cell.cell_id, "new"),
            }
            for cell in cells
        },
        "seeds": (
            {}
            if seed_registry is None or seed_registry.empty
            else {
                role: group[["seed_key", "spawn_key", "n_streams"]].to_dict("records")
                for role, group in seed_registry.groupby("role")
            }
        ),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "source_identity": source_identity(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest.update(extra or {})
    atomic_write_text(paths.manifest, json.dumps(manifest, indent=2, default=str))
    return manifest


def existing_finished_run(output_dir: str | Path, run_id: str) -> bool:
    """Whether a finished run of this identifier already exists.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Project output root.
    run_id : str
        Run identifier.

    Returns
    -------
    bool
        True when a manifest with an end timestamp is already on disk, and also
        when a manifest exists but cannot be read.

    Notes
    -----
    This is a guard whose only purpose is to refuse to overwrite a
    finished run, so it fails closed. An unreadable manifest does not
    establish that the run is unfinished (the end timestamp may well be
    on disk and merely unparseable), so treating that case as "not
    finished" would withdraw the protection in the situation where the
    directory is already known to be damaged. The caller can still
    proceed under a different run identifier.
    """
    manifest = Path(output_dir) / SIM_SUBDIR / run_id / MANIFEST_NAME
    if not manifest.exists():
        return False
    try:
        return (
            json.loads(manifest.read_text(encoding="utf-8")).get("finished") is not None
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return True


# =========================================================
# Checkpoints
# =========================================================
def chunk_paths(paths: RunPaths, cell, start: int, stop: int) -> tuple[Path, Path]:
    """Return the parquet and marker paths of one replication chunk.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    cell : Cell
        The cell the chunk belongs to.
    start, stop : int
        Half-open replication range of the chunk.

    Returns
    -------
    tuple of pathlib.Path
        The data and marker paths.
    """
    directory = paths.checkpoints / cell.part / cell.cell_id
    return (
        directory / f"chunk_{start}_{stop}.parquet",
        directory / f"chunk_{start}_{stop}.done",
    )


def chunk_is_complete(paths: RunPaths, cell, start: int, stop: int) -> bool:
    """Whether a chunk is finished and was written under this configuration.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    cell : Cell
        The cell.
    start, stop : int
        Half-open replication range.

    Returns
    -------
    bool
        True when the chunk may be reused.
    """
    parquet, marker = chunk_paths(paths, cell, start, stop)
    if not (parquet.exists() and marker.exists()):
        return False
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if recorded.get("config_hash") != cell.config_hash:
        return False

    return True


def write_chunk(
    paths: RunPaths, cell, start: int, stop: int, frame: pd.DataFrame
) -> None:
    """Write one chunk and its marker atomically, data first.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    cell : Cell
        The cell.
    start, stop : int
        Half-open replication range.
    frame : pandas.DataFrame
        The chunk's records.
    """
    parquet, marker = chunk_paths(paths, cell, start, stop)
    atomic_write_dataframe(parquet, frame)
    atomic_write_text(
        marker,
        json.dumps(
            {
                "cell_id": cell.cell_id,
                "config_hash": cell.config_hash,
                "reuse_key": cell.reuse_key,
                "schema_version": simcfg.OUTPUT_SCHEMA_VERSION,
                "start": start,
                "stop": stop,
                "rows": int(len(frame)),
                "written": datetime.now(timezone.utc).isoformat(),
            }
        ),
    )


def read_cell_records(paths: RunPaths, cell) -> pd.DataFrame:
    """Read every completed chunk of a cell, in replication order.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    cell : Cell
        The cell.

    Returns
    -------
    pandas.DataFrame
        The concatenated records, or an empty frame when nothing is stored.
    """
    directory = paths.checkpoints / cell.part / cell.cell_id
    if not directory.exists():
        return pd.DataFrame()

    frames = []
    for marker in sorted(directory.glob("chunk_*.done")):
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if recorded.get("config_hash") != cell.config_hash:
            continue
        parquet = marker.with_suffix(".parquet")
        if parquet.exists():
            frames.append(pd.read_parquet(parquet))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    sort_columns = [c for c in ("replication_id", "record_index") if c in combined]
    return (
        combined.sort_values(sort_columns).reset_index(drop=True)
        if sort_columns
        else combined
    )


# =========================================================
# Cross-run output reuse
# =========================================================
def reuse_candidates(output_dir: str | Path, cell, current_run_id: str) -> list[dict]:
    """Find finished runs whose stored chunks of this cell may be reused.

    A candidate qualifies only when the run is finished and every stored
    marker of the cell carries this cell's reuse key and this record
    schema version. The reuse key covers the data-generating process,
    the estimator controls, the statistic, the bootstrap algorithm, the
    seeds and the replication and draw counts.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Project output root.
    cell : Cell
        The cell whose output might be reused.
    current_run_id : str
        The run being written, which is not its own source.

    Returns
    -------
    list of dict
        One entry per qualifying source, with its run identifier, its chunk
        directory and how many chunks it holds. Newest first.
    """
    root = Path(output_dir) / SIM_SUBDIR
    if not root.exists():
        return []

    found = []
    for run in sorted(root.iterdir(), reverse=True):
        if not run.is_dir() or run.name == current_run_id:
            continue
        if not existing_finished_run(output_dir, run.name):
            continue

        directory = run / "checkpoints" / cell.part / cell.cell_id
        if not directory.exists():
            continue

        markers = sorted(directory.glob("chunk_*.done"))
        if not markers:
            continue

        matched = []
        for marker in markers:
            try:
                recorded = json.loads(marker.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                matched = []
                break
            if recorded.get("reuse_key") != cell.reuse_key:
                matched = []
                break
            if recorded.get("schema_version") != simcfg.OUTPUT_SCHEMA_VERSION:
                matched = []
                break
            if not marker.with_suffix(".parquet").exists():
                matched = []
                break
            matched.append(marker)

        if matched:
            found.append(
                {"run_id": run.name, "directory": directory, "n_chunks": len(matched)}
            )

    return found


def import_reusable_chunks(
    paths: RunPaths, cell, output_dir: str | Path, run_id: str
) -> str | None:
    """Copy a qualifying source run's chunks of a cell into this run.

    Parameters
    ----------
    paths : RunPaths
        The run layout being written.
    cell : Cell
        The cell.
    output_dir : str or pathlib.Path
        Project output root.
    run_id : str
        The run being written.

    Returns
    -------
    str or None
        The source run identifier when chunks were imported, otherwise None.
    """
    candidates = reuse_candidates(output_dir, cell, run_id)
    if not candidates:
        return None

    source = candidates[0]
    destination = paths.checkpoints / cell.part / cell.cell_id
    destination.mkdir(parents=True, exist_ok=True)

    for marker in sorted(source["directory"].glob("chunk_*.done")):
        parquet = marker.with_suffix(".parquet")
        shutil.copy2(parquet, destination / parquet.name)
        shutil.copy2(marker, destination / marker.name)

    return source["run_id"]


# =========================================================
# Tables, figures and telemetry
# =========================================================
def save_table(
    paths: RunPaths, name: str, frame: pd.DataFrame, group: str
) -> Path:
    """Write one tidy CSV, stamped with the run mode and its output group.

    The two stamps are the first two columns of every table. They are
    what stops a preview artefact from being mistaken for a production
    result once the file has been copied out of its run directory, and
    what places the table in the main text or in one of the two
    supplementary groups without anyone having to remember which.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    name : str
        File stem.
    frame : pandas.DataFrame
        The table.
    group : str
        Which output group the table belongs to.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    stamped = frame.copy()
    for column in ("run_mode", "output_group"):
        if column in stamped.columns:
            stamped = stamped.drop(columns=column)
    stamped.insert(0, "output_group", group)
    stamped.insert(0, "run_mode", paths.mode)

    target = paths.tables / f"{name}.csv"
    atomic_write_dataframe(target, stamped)

    paths.index = [entry for entry in paths.index if entry["file"] != target.name]
    paths.index.append(
        {
            "file": target.name,
            "table": name,
            "output_group": group,
            "rows": int(len(stamped)),
            "columns": int(stamped.shape[1]),
        }
    )
    return target


def save_telemetry(paths: RunPaths, name: str, frame: pd.DataFrame) -> Path:
    """Write one telemetry file.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    name : str
        File name including its suffix.
    frame : pandas.DataFrame
        The telemetry.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    target = paths.telemetry / name
    atomic_write_dataframe(target, frame)
    return target


def save_figure(paths: RunPaths, name: str, figure) -> Path:
    """Save one figure and close it.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    name : str
        File stem.
    figure : matplotlib.figure.Figure
        The figure.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    import matplotlib.pyplot as plt

    target = paths.figures / f"{name}.png"
    figure.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return target


def write_output_index(paths: RunPaths) -> pd.DataFrame:
    """Write the index of every table this run produced.

    Parameters
    ----------
    paths : RunPaths
        The run layout.

    Returns
    -------
    pandas.DataFrame
        The index.
    """
    frame = pd.DataFrame(
        paths.index, columns=["file", "table", "output_group", "rows", "columns"]
    ).sort_values(["output_group", "table"])

    stamped = frame.copy()
    stamped.insert(0, "run_mode", paths.mode)
    atomic_write_dataframe(paths.tables / "output_index.csv", stamped)
    return frame

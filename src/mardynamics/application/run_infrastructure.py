"""Run infrastructure of the empirical study.

Provides separate preview and full directories, a config hash over the
complete design, deterministic seeds derived from full logical
coordinates, atomic writes, checkpoints that a foreign run cannot
consume, failure accounting and the stamping rules that put ``run_mode``
first in every table and inside every JSON object.

It has no simulation consumer and imports nothing from the simulation
package. The simulation's own runner, seed and IO modules are not
imported, moved or edited here; this module is a separate implementation
of the same conventions for the empirical layer.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import mardynamics.application.emp_config as emp_config

# Marker file written when a run has completed every stage.
COMPLETION_MARKER = "RUN_COMPLETE.json"


def enforce_thread_limits() -> None:
    """Pin the numerical libraries to one thread.

    Sets ``OMP_NUM_THREADS=1``, ``MKL_NUM_THREADS=1`` and the other
    thread-count environment variables. These bind only when a library
    loads, so a process-wide limiter is also installed and kept alive at
    module scope.
    """
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"

    global _THREAD_LIMITER
    if _THREAD_LIMITER is None:
        try:
            import threadpoolctl

            _THREAD_LIMITER = threadpoolctl.threadpool_limits(1)
        except Exception:  # pragma: no cover - threadpoolctl is optional
            _THREAD_LIMITER = False


_THREAD_LIMITER = None


# =========================================================
# Seeds
# =========================================================
def derive_seed(*coordinates) -> int:
    """Derive one reproducible seed from the master seed and a coordinate tuple.

    The seed depends on MASTER_SEED and the full coordinate tuple, not on
    execution order.

    Parameters
    ----------
    *coordinates
        The complete logical identity of the draw, for example
        ``("wild_bootstrap", "full_sample", draw_index)``.

    Returns
    -------
    int
        A 63-bit seed.
    """
    payload = "|".join(str(part) for part in (emp_config.MASTER_SEED, *coordinates))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def rng(*coordinates) -> np.random.Generator:
    """Return the generator of one logically identified stochastic object.

    Parameters
    ----------
    *coordinates
        The complete logical identity, as in :func:`derive_seed`.

    Returns
    -------
    numpy.random.Generator
        The generator.
    """
    return np.random.default_rng(derive_seed(*coordinates))


# =========================================================
# Config hash
# =========================================================
def config_hash(payload: dict) -> str:
    """Hash the complete design payload of a run.

    Combines the run mode, sample bounds, preprocessing versions, tuning
    grids, numerical settings, seeds and counts into one hash.

    Parameters
    ----------
    payload : dict
        JSON-serialisable design description.

    Returns
    -------
    str
        Sixteen hexadecimal characters.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def design_payload(mode: str) -> dict:
    """Build the hashed description of one empirical run.

    Parameters
    ----------
    mode : str
        The run mode.

    Returns
    -------
    dict
        Every constant that can change a number in the run.
    """
    return {
        "run_mode": mode,
        "schema_version": emp_config.SCHEMA_VERSION,
        "countries": list(emp_config.COUNTRIES),
        "indicators": list(emp_config.INDICATORS),
        "sample": [emp_config.SAMPLE_START, emp_config.SAMPLE_END],
        "raw": [emp_config.RAW_START, emp_config.RAW_END],
        "validation": [emp_config.VALIDATION_START, emp_config.VALIDATION_END],
        "test": [emp_config.TEST_START, emp_config.TEST_END],
        "preprocessor_cutoffs": dict(emp_config.PREPROCESSOR_CUTOFFS),
        "substitutions": [dict(s) for s in emp_config.APPROVED_SUBSTITUTIONS],
        "rank_grid": list(emp_config.RANK_GRID),
        "window_grid": list(emp_config.WINDOW_GRID),
        "tie_tolerance": emp_config.TIE_TOLERANCE,
        "n_regimes": emp_config.N_REGIMES,
        "n_starts": emp_config.N_STARTS,
        "start_perturbation": emp_config.START_PERTURBATION,
        "static": [emp_config.STATIC_MAX_ITER, emp_config.STATIC_TOL],
        "als": [emp_config.ALS_MAX_ITER, emp_config.ALS_TOL, emp_config.ALS_STOPPING_RULE],
        "stability": [emp_config.ENFORCE_STABILITY, emp_config.MAX_RADIUS],
        "em": [
            emp_config.EM_MAX_ITER,
            emp_config.EM_MIN_ITER,
            emp_config.EM_LOGLIK_TOL,
            emp_config.EM_PARAM_TOL,
            emp_config.EM_PARAM_RULE,
            emp_config.EM_ALS_ITER_PER_MSTEP,
        ],
        "probability_floor": emp_config.PROBABILITY_FLOOR,
        "gem_rollback_tolerance": emp_config.GEM_ROLLBACK_TOLERANCE,
        # These four are author decisions, but they decide a written
        # interpretation status, so two runs that would report different
        # regime statuses must not share a config hash.
        "regime_interpretation_thresholds": [
            emp_config.REGIME_MIN_OCCUPANCY_SHARE,
            emp_config.REGIME_MIN_EFFECTIVE_SAMPLE,
            emp_config.REGIME_MULTISTART_LOGLIK_TOL,
            emp_config.REGIME_MULTISTART_MIN_AGREEING_FRACTION,
        ],
        "limit_law": [
            emp_config.LIMITLAW_N_DRAWS,
            emp_config.LIMITLAW_N_GRID,
            emp_config.LIMITLAW_SEED,
        ],
        "wild_bootstrap": [emp_config.wild_bootstrap_draws(mode), emp_config.WILD_BOOT_SEED],
        "hac_lag": emp_config.HAC_LAG,
        "master_seed": emp_config.MASTER_SEED,
        "origin_stride": (
            emp_config.PREVIEW_ORIGIN_STRIDE if mode == "preview" else 1
        ),
        "adf": [emp_config.ADF_REGRESSION, emp_config.ADF_MAX_LAG, emp_config.ADF_AUTOLAG],
    }


# =========================================================
# Run layout
# =========================================================
@dataclass
class RunPaths:
    """The directories and identity of one empirical run.

    Attributes
    ----------
    root : pathlib.Path
        ``outputs/application/<run_id>``.
    mode : str
        The run mode, stamped into every artifact.
    hash : str
        The config hash of the run.
    """

    root: Path
    mode: str
    hash: str

    @property
    def run_id(self) -> str:
        """Directory name of the run."""
        return self.root.name

    @property
    def tables(self) -> Path:
        """Machine-readable CSV outputs."""
        return self.root / "tables"

    @property
    def json(self) -> Path:
        """Machine-readable JSON outputs."""
        return self.root / "json"

    @property
    def figures(self) -> Path:
        """Rendered figure previews."""
        return self.root / "figures"

    @property
    def checkpoints(self) -> Path:
        """Stage checkpoints."""
        return self.root / "checkpoints"

    @property
    def telemetry(self) -> Path:
        """Failure and timing telemetry."""
        return self.root / "telemetry"

    def create(self) -> None:
        """Create every directory of the layout."""
        for directory in (
            self.tables,
            self.json,
            self.figures,
            self.checkpoints,
            self.telemetry,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def create_run(mode: str, output_root: str | Path = "outputs/application") -> RunPaths:
    """Open a new run directory for one mode.

    Preview and full runs get separate directories, hashes, manifests and
    checkpoints. Refuses to overwrite a completed run.

    Parameters
    ----------
    mode : str
        ``"preview"`` or ``"full"``.
    output_root : str or pathlib.Path, optional
        Parent directory of all empirical runs.

    Returns
    -------
    RunPaths
        The layout of the new run.
    """
    if mode not in emp_config.RUN_MODES:
        raise ValueError(f"unknown run mode {mode!r}")

    digest = config_hash(design_payload(mode))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_root) / f"{mode}_{stamp}"
    if (root / COMPLETION_MARKER).exists():
        raise FileExistsError(f"refusing to overwrite the completed run {root}")

    paths = RunPaths(root=root, mode=mode, hash=digest)
    paths.create()
    return paths


def mark_complete(paths: RunPaths, summary: dict) -> None:
    """Write the completion marker of a finished run.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    summary : dict
        Closing counts and timings.
    """
    _atomic_write_text(
        paths.root / COMPLETION_MARKER,
        json.dumps(
            {"run_mode": paths.mode, "config_hash": paths.hash, **summary},
            indent=2,
            default=str,
        ),
    )


# =========================================================
# Atomic writes and stamping
# =========================================================
def _atomic_write_text(path: Path, text: str) -> None:
    """Write text through a temporary file in the same directory.

    Parameters
    ----------
    path : pathlib.Path
        Destination.
    text : str
        Content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def stamp_frame(paths: RunPaths, frame: pd.DataFrame, **identity) -> pd.DataFrame:
    """Put ``run_mode`` first and add the identity columns.

    Every table begins with ``run_mode`` and records a schema version, the
    complete config hash, the sample dates, the preprocessing version and
    the result identity.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    frame : pandas.DataFrame
        The table.
    **identity
        Extra identity columns, for example ``preprocessing="test"``.

    Returns
    -------
    pandas.DataFrame
        The stamped table.
    """
    stamped = frame.copy()
    trailing = {
        "schema_version": emp_config.SCHEMA_VERSION,
        "config_hash": paths.hash,
        "sample_start": emp_config.SAMPLE_START,
        "sample_end": emp_config.SAMPLE_END,
        **identity,
    }
    for column, value in trailing.items():
        stamped[column] = value
    stamped.insert(0, "run_mode", paths.mode)
    return stamped


def save_table(paths: RunPaths, name: str, frame: pd.DataFrame, **identity) -> Path:
    """Stamp and atomically write one machine-readable table.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    name : str
        File stem, without the ``.csv`` suffix.
    frame : pandas.DataFrame
        The table.
    **identity
        Extra identity columns.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    destination = paths.tables / f"{name}.csv"
    _atomic_write_text(destination, stamp_frame(paths, frame, **identity).to_csv(index=False))
    return destination


def save_json(paths: RunPaths, name: str, payload: dict) -> Path:
    """Stamp and atomically write one machine-readable JSON object.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    name : str
        File stem, without the ``.json`` suffix.
    payload : dict
        The object.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    stamped = {
        "run_mode": paths.mode,
        "schema_version": emp_config.SCHEMA_VERSION,
        "config_hash": paths.hash,
        "sample_start": emp_config.SAMPLE_START,
        "sample_end": emp_config.SAMPLE_END,
        **payload,
    }
    destination = paths.json / f"{name}.json"
    _atomic_write_text(destination, json.dumps(stamped, indent=2, default=_json_default))
    return destination


def _json_default(value):
    """Serialise NumPy scalars and arrays inside a JSON payload.

    Parameters
    ----------
    value : object
        The unserialisable value.

    Returns
    -------
    object
        A JSON-friendly replacement.
    """
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


# =========================================================
# Checkpoints
# =========================================================
def checkpoint_path(paths: RunPaths, key: str) -> Path:
    """Return the checkpoint file of one stage.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    key : str
        Stage key.

    Returns
    -------
    pathlib.Path
        The checkpoint path.
    """
    return paths.checkpoints / f"{key}.json"


def write_checkpoint(paths: RunPaths, key: str, payload: dict) -> None:
    """Store one stage checkpoint under the run's mode and hash.

    Parameters
    ----------
    paths : RunPaths
        The run layout.
    key : str
        Stage key.
    payload : dict
        The stage result.
    """
    _atomic_write_text(
        checkpoint_path(paths, key),
        json.dumps(
            {"run_mode": paths.mode, "config_hash": paths.hash, "payload": payload},
            indent=2,
            default=_json_default,
        ),
    )


# =========================================================
# Source and dependency provenance
# =========================================================
def _git_state(repo_root: Path = Path(".")) -> dict:
    """Return the source Git commit and working-tree cleanliness.

    Parameters
    ----------
    repo_root : pathlib.Path, optional
        Repository root.

    Returns
    -------
    dict
        ``commit`` (or an explanatory string when Git is unavailable) and
        ``dirty`` (``True``/``False``/``None`` when it cannot be determined).
    """
    import subprocess

    def _run(*args):
        return subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        commit = _run("rev-parse", "HEAD")
        dirty = bool(_run("status", "--porcelain"))
        return {"commit": commit, "dirty": dirty}
    except Exception as error:  # pragma: no cover - environment-dependent
        return {"commit": f"unavailable: {error}", "dirty": None}


def executor_file_hashes(package_root: str | Path = "src/mardynamics/application") -> dict:
    """Hash every active empirical executor file and the source tree as a whole.

    Parameters
    ----------
    package_root : str or pathlib.Path, optional
        The active empirical package.

    Returns
    -------
    dict
        ``files`` (path to SHA-256) and ``source_tree_digest`` (one hash of
        every file digest, sorted by path, so any change to any active file
        changes it).
    """
    root = Path(package_root)
    files = {}
    for path in sorted(root.glob("*.py")):
        digest = hashlib.sha256()
        digest.update(path.read_bytes())
        files[path.as_posix()] = digest.hexdigest()

    tree_digest = hashlib.sha256(
        "|".join(f"{name}:{value}" for name, value in sorted(files.items())).encode(
            "utf-8"
        )
    ).hexdigest()
    return {"files": files, "source_tree_digest": tree_digest}


def dependency_versions() -> dict:
    """Return the installed versions of the packages that can change a result.

    Returns
    -------
    dict
        Package name to installed version, or ``"not_installed"``.
    """
    from importlib import metadata

    versions = {}
    for package in ("numpy", "pandas", "scipy", "statsmodels", "joblib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def source_provenance(package_root: str | Path = "src/mardynamics/application") -> dict:
    """Assemble the complete provenance block.

    Parameters
    ----------
    package_root : str or pathlib.Path, optional
        The active empirical package.

    Returns
    -------
    dict
        Git state, executor file hashes, source-tree digest and dependency
        versions.
    """
    hashes = executor_file_hashes(package_root)
    return {
        "git": _git_state(),
        "executor_file_hashes": hashes["files"],
        "source_tree_digest": hashes["source_tree_digest"],
        "dependency_versions": dependency_versions(),
    }


# =========================================================
# Failure accounting
# =========================================================
@dataclass
class FailureLog:
    """Accounting of attempted, successful and failed units of work.

    Tracks planned, attempted, successful and failed counts, the complete
    dropped-origin list and a record of every hard failure. A failed
    forecast is not replaced.
    """

    records: list = field(default_factory=list)

    def record(self, *, unit: str, identity: dict, reason: str, message: str) -> None:
        """Register one hard failure.

        Parameters
        ----------
        unit : str
            What failed, for example ``"forecast"``.
        identity : dict
            The full logical coordinates of the failed unit.
        reason : str
            The failure category.
        message : str
            Human-readable detail.
        """
        self.records.append(
            {
                "unit": unit,
                **{f"id_{key}": value for key, value in identity.items()},
                "failure_reason": reason,
                "failure_message": str(message)[:500],
                "recorded": datetime.now(timezone.utc).isoformat(),
            }
        )

    def frame(self) -> pd.DataFrame:
        """Return the failures as a table.

        Returns
        -------
        pandas.DataFrame
            One row per failure; empty when nothing failed.
        """
        return pd.DataFrame(self.records)

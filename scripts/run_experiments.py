"""The driver for the MAR simulation study.

Press Run. The one choice that decides what happens is the first constant below,
and it is plain Python so that nothing has to be typed on a command line:

    DEFAULT_RUN_MODE   "smoke", "preview" or "production"

A preview run sits between the smoke and the production run. It
executes the complete registry (the same cells, the same tables, the
same figures) with every outer replication count capped and every
bootstrap draw count capped: it shows what the finished study will look
like, at counts far below the precision any claim rests on. Every
preview table carries ``run_mode = "preview"`` in its first column and
its manifest records the same, so a preview artefact cannot be mistaken
for a result.

A smoke run is the same thing at very small counts: its purpose is that
every applicable field is finite and every design object passes its
verification.

Environment overrides exist for convenience: ``MAR_SIM_MODE``,
``MAR_SIM_WORKERS``, ``MAR_SIM_PARTS``.

The study generates synthetic data only: it reads no empirical data, no
empirical estimate and no empirical output. Everything is written to
``outputs/experiments/<run_id>/``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The line the user changes.
# ---------------------------------------------------------------------------
DEFAULT_RUN_MODE = "production"
# ---------------------------------------------------------------------------

import os
import time

# Threading is pinned before NumPy or SciPy are imported anywhere, by
# this process and by the workers that inherit its environment. One
# parallelisation layer only: process workers over replications.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")  # headless: write figures to disk, not a window

import mardynamics.config as paths_cfg
import mardynamics.experiments.sim_config as simcfg
import mardynamics.experiments.study as study


def _resolve_settings() -> dict:
    """Read the run settings from the constant above and the environment.

    Returns
    -------
    dict
        The resolved mode, worker count and optional part filter.
    """
    parts = os.environ.get("MAR_SIM_PARTS")
    return {
        "mode": os.environ.get("MAR_SIM_MODE", DEFAULT_RUN_MODE),
        "workers": min(
            int(os.environ.get("MAR_SIM_WORKERS", simcfg.MAX_WORKERS)),
            simcfg.MAX_WORKERS,
        ),
        "parts": tuple(parts.split(",")) if parts else None,
    }


def main() -> None:
    """Run the study in the configured mode."""
    settings = _resolve_settings()
    mode = settings["mode"]

    print(
        f"mode={mode} profile={simcfg.DEFAULT_PROFILE} "
        f"workers={settings['workers']} master_seed={simcfg.SIM_MASTER_SEED} "
        f"schema={simcfg.OUTPUT_SCHEMA_VERSION}"
    )

    if mode in ("smoke", "preview"):
        started = time.perf_counter()
        summary = study.run_study(
            mode=mode,
            workers=settings["workers"],
            output_dir=paths_cfg.OUTPUT_DIR,
            parts=settings["parts"],
        )
        elapsed = time.perf_counter() - started

        print("")
        print("=" * 78)
        print(f"{mode.upper()} COMPLETE (NOT A RESULT)")
        caps = summary["registry_summary"].get("caps", {})
        if caps:
            print(
                f"  outer replications capped at {caps['outer']}, "
                f"bootstrap draws capped at {caps['inner']}"
            )
        print(
            f"  wall time {elapsed / 60.0:.1f} min "
            f"({elapsed:.0f} s) at {settings['workers']} workers"
        )
        print(f"  {len(summary['tables'])} tables, {len(summary['figures'])} figures")
        print(f"  every table carries run_mode = {mode!r} in its first column")
        print("=" * 78)
        print(f"{mode} complete in {elapsed:.0f}s -> {summary['paths'].root}")
        return

    if mode != "production":
        raise SystemExit(
            f"unknown run mode {mode!r}; expected smoke, preview or production."
        )

    started = time.perf_counter()
    summary = study.run_study(
        mode="production",
        workers=settings["workers"],
        output_dir=paths_cfg.OUTPUT_DIR,
        parts=settings["parts"],
    )
    elapsed = time.perf_counter() - started

    accounting = summary["accounting"]
    print("")
    print("=" * 78)
    print("PRODUCTION RUN COMPLETE")
    print(f"  wall time {elapsed / 3600.0:.2f} h at {settings['workers']} workers")
    print(
        f"  replications: attempted {int(accounting['n_attempted'].sum())} "
        f"completed {int(accounting['n_completed'].sum())} "
        f"failed {int(accounting['n_failed'].sum())} "
        f"excluded {int(accounting['n_excluded'].sum())}"
    )
    reused = accounting[accounting["source"] != "new"]
    print(f"  outputs reused rather than rerun: {len(reused)} cells")
    print(f"  output directory: {summary['paths'].root}")
    print("=" * 78)


if __name__ == "__main__":
    main()

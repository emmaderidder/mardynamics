"""Canonical launcher of the empirical study.

Press Run. The one choice that decides what happens is the first constant
below, and it is plain Python so that nothing has to be typed on a command
line:

    DEFAULT_RUN_MODE   "preview" or "full"

A full run re-runs every transformation, tuning choice, estimate, forecast
and test from the raw inputs, over all validation origins, all 105 test
origins and 499 wild-bootstrap draws, and it does not promote a preview
selection, estimate, checkpoint or cached table.

A preview run executes the complete pipeline and writes every registered output
with its final schema, using 49 wild-bootstrap draws and every fourth ordered
validation and test origin beginning with the first in each list. It keeps the
complete 50,000-draw limit-law cache. Preview values are technical outputs only
and may not enter the thesis.

The pipeline calls the shared numerical layer for every estimator, canonical
map, statistic, limit law, bootstrap and Holm adjustment. It reads no
simulation module and writes nothing outside ``outputs/application/``.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The line the user changes.
# ---------------------------------------------------------------------------
DEFAULT_RUN_MODE = "full"
# ---------------------------------------------------------------------------

import os
import sys
from pathlib import Path

# Pin the numerical libraries before anything imports them.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_name, "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import mardynamics.application.emp_config as ecfg  # noqa: E402
import mardynamics.application.study as study  # noqa: E402

# Convenience overrides.
RUN_MODE = os.environ.get("MAR_EMP_MODE", DEFAULT_RUN_MODE)
WORKERS = int(os.environ.get("MAR_EMP_WORKERS", ecfg.MAX_WORKERS))
DATA_DIR = os.environ.get("MAR_EMP_DATA", "data")


def main() -> None:
    """Run the empirical study in the selected mode."""
    if RUN_MODE not in ecfg.RUN_MODES:
        raise SystemExit(f"unknown run mode {RUN_MODE!r}; use preview or full")

    workers = max(1, min(WORKERS, ecfg.MAX_WORKERS))
    print(f"[mode] {RUN_MODE} with {workers} worker(s)")

    summary = study.run(RUN_MODE, workers, DATA_DIR)

    print("\n" + "=" * 74)
    print(f"{RUN_MODE.upper()} RUN COMPLETE -> {summary['paths'].root}")
    print(
        f"  wall {summary['elapsed_seconds'] / 60:.1f} min | "
        f"validation units {summary['validation_units']} "
        f"({summary['validation_failed']} failed) | "
        f"test units {summary['test_units']} ({summary['test_failed']} failed)"
    )
    print(
        f"  common origins {summary['common_origins']}"
        + (
            f" | dropped {summary['dropped_origins']}"
            if summary["dropped_origins"]
            else ""
        )
    )
    if RUN_MODE == "preview":
        print(
            "\n  To run the full run, change the line at the top of this file:\n"
            '      DEFAULT_RUN_MODE = "full"'
        )
    print("=" * 74)


if __name__ == "__main__":
    main()

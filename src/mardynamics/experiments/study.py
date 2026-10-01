"""The study orchestrator: one run, from the registry to the tables.

The order is fixed. The registry is built. The design is constructed
and verified before any data is simulated. The limit laws are warmed in
this process before any worker starts. The manifest is written before
the first cell executes and closed only when the last one has.

Nothing here decides how many replications a cell has: that is the
registry's. Nothing here generates narrative text. The outputs are
validated numbers, table data, figure data, manifests and execution
records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import mardynamics.config as paths_cfg
import mardynamics.experiments.design_audit as design_audit
import mardynamics.experiments.figures as figures
import mardynamics.experiments.registry as registry
import mardynamics.experiments.runner as runner
import mardynamics.experiments.sim_config as simcfg
import mardynamics.experiments.sim_io as simio
from mardynamics.experiments.designs import calibration, detection, matrix_restriction
from mardynamics.experiments.designs.common import LIMIT_LAW_CACHE

# The five quantities the main-text attribution table carries. The
# remaining three measures are supplementary only.
MAIN_ATTRIBUTION_MEASURES = (
    "global_rejection",
    "correct_joint",
    "correct_given_global",
    "correct_only_given_global",
    "false_given_global",
)


class SeedStreamError(RuntimeError):
    """Two seed identities share a spawn stream."""


def _cells_of(cells: list, **criteria) -> list:
    """Select the cells matching a set of exact attribute values.

    Parameters
    ----------
    cells : list
        The registry.
    **criteria
        Attribute values a cell must match.

    Returns
    -------
    list
        The matching cells, in registry order.
    """
    return [
        cell
        for cell in cells
        if all(getattr(cell, key) == value for key, value in criteria.items())
    ]


def _wide_attribution(rows: list[dict], level: float, measures: tuple) -> pd.DataFrame:
    """Pivot long attribution rows into one row per cell at one level.

    Parameters
    ----------
    rows : list of dict
        Long attribution rows.
    level : float
        Which nominal level to keep.
    measures : tuple of str
        Which measures become columns, in order.

    Returns
    -------
    pandas.DataFrame
        The wide table.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame()

    at_level = frame[frame["level"] == level]
    identifiers = [
        "part",
        "cell_id",
        "population_configuration",
        "m",
        "n",
        "T",
        "changing_block",
        "path",
        "magnitude",
        "block_dimension",
        "blockwise_signal",
        "global_signal_error",
        "noncentrality",
        "n_completed",
    ]

    wide = []
    for cell_id, group in at_level.groupby("cell_id", sort=False):
        row = {key: group[key].iloc[0] for key in identifiers}
        row["level"] = level
        for measure in measures:
            selected = group[group["measure"] == measure]
            if selected.empty:
                row[measure] = np.nan
                row[f"{measure}_se"] = np.nan
                continue
            entry = selected.iloc[0]
            row[measure] = entry["rate"]
            row[f"{measure}_se"] = (
                entry["mcse"] if entry["scope"] == "unconditional" else entry["descriptive_mc_se"]
            )
            if entry["scope"] == "conditional":
                row["conditional_denominator"] = entry["conditional_denominator"]
        row["conditional_uncertainty_kind"] = (
            "descriptive_binomial_at_observed_denominator"
        )
        wide.append(row)

    return pd.DataFrame(wide)


# =========================================================
# Table assembly
# =========================================================
def build_tables(cells: list, paths: simio.RunPaths, audit: dict) -> dict:
    """Aggregate the stored records of every cell into the output tables.

    Parameters
    ----------
    cells : list
        The registry.
    paths : RunPaths
        The run layout.
    audit : dict
        The design-audit result, whose design tables are written here.

    Returns
    -------
    dict
        Mapping from table name to the written DataFrame.
    """
    written: dict[str, pd.DataFrame] = {}

    def _write(name, frame, group):
        if frame is None or len(frame) == 0:
            return
        simio.save_table(paths, name, frame, group)
        written[name] = frame

    records = {cell.cell_id: simio.read_cell_records(paths, cell) for cell in cells}

    # ---------- The constructed design ----------
    for name in (
        "population_matrices",
        "change_directions",
        "signal_measures",
        "departure_size_summary",
        "path_verification",
        "rejected_population_draws",
    ):
        _write(name, audit[name], simcfg.TABLE_GROUP_CONSTRUCTION)
    _write(
        "unrestricted_directions",
        audit["unrestricted_directions"],
        simcfg.TABLE_GROUP_COMPLETE,
    )
    _write(
        "var_feasibility",
        pd.DataFrame(registry.var_feasibility_table()),
        simcfg.TABLE_GROUP_COMPLETE,
    )

    # ---------- Calibration, bootstrap, recovery, starts ----------
    null_cells = _cells_of(cells, kind="null_calibration")
    _write(
        "null_calibration",
        pd.concat(
            [calibration.null_calibration_table(c, records[c.cell_id]) for c in null_cells],
            ignore_index=True,
        ),
        simcfg.TABLE_GROUP_MAIN,
    )

    recovery = pd.concat(
        [calibration.recovery_table(c, records[c.cell_id]) for c in null_cells],
        ignore_index=True,
    )
    if not recovery.empty:
        main_m, main_n = simcfg.RECOVERY_MAIN_DIMENSION
        main = recovery[
            (recovery["m"] == main_m)
            & (recovery["n"] == main_n)
            & (recovery["innovation"] == "isotropic")
            & (recovery["persistence"] == simcfg.BASE_PERSISTENCE)
        ]
        _write("numerical_recovery_main", main, simcfg.TABLE_GROUP_MAIN)
        _write(
            "numerical_recovery_appendix",
            recovery,
            simcfg.TABLE_GROUP_CONSTRUCTION,
        )

    for cell in _cells_of(cells, kind="wild_bootstrap"):
        _write(
            "wild_bootstrap_comparison",
            calibration.wild_bootstrap_table(cell, records[cell.cell_id]),
            simcfg.TABLE_GROUP_COMPLETE,
        )

    for cell in _cells_of(cells, kind="multiple_start"):
        _write(
            "multiple_start_summary",
            calibration.multiple_start_table(cell, records[cell.cell_id]),
            simcfg.TABLE_GROUP_CONSTRUCTION,
        )

    for cell in cells:
        if "garch_strength" in cell.statistics:
            _write(
                "garch_strength",
                calibration.garch_strength_table(cell, records[cell.cell_id]),
                simcfg.TABLE_GROUP_COMPLETE,
            )
        if "nyblom_var" in cell.statistics and cell.kind == "null_calibration":
            frame = calibration.unrestricted_null_dimension_table(
                records[cell.cell_id], cell.m, cell.n
            )
            if not frame.empty:
                frame.insert(0, "cell_id", cell.cell_id)
                _write(
                    "matrix_restriction_null_dimension",
                    frame,
                    simcfg.TABLE_GROUP_COMPLETE,
                )

    # ---------- Detection and the sensitivity branch: attribution ----------
    detection_cells = _cells_of(cells, kind="detection")
    main_detection = [c for c in detection_cells if c.part == "C"]

    attribution_rows: list[dict] = []
    unconditional_rows: list[dict] = []
    for cell in main_detection:
        attribution_rows.extend(detection.attribution_rows(cell, records[cell.cell_id]))
        unconditional_rows.extend(
            detection.unconditional_block_rows(cell, records[cell.cell_id])
        )

    _write(
        "attribution_full_appendix",
        pd.DataFrame(attribution_rows),
        simcfg.TABLE_GROUP_COMPLETE,
    )
    _write(
        "attribution_unconditional_blocks",
        pd.DataFrame(unconditional_rows),
        simcfg.TABLE_GROUP_COMPLETE,
    )
    _write(
        "attribution_classification_check",
        pd.DataFrame(detection.classification_check(attribution_rows)),
        simcfg.TABLE_GROUP_COMPLETE,
    )

    medium = [row for row in attribution_rows if row["magnitude"] == "medium"]
    _write(
        "attribution_medium_main",
        _wide_attribution(medium, simcfg.PRIMARY_LEVEL, MAIN_ATTRIBUTION_MEASURES),
        simcfg.TABLE_GROUP_MAIN,
    )

    _write(
        "design_sensitivity_full",
        _design_sensitivity(cells, records),
        simcfg.TABLE_GROUP_COMPLETE,
    )
    if "design_sensitivity_full" in written:
        _write(
            "design_sensitivity_summary",
            _design_sensitivity_summary(written["design_sensitivity_full"]),
            simcfg.TABLE_GROUP_COMPLETE,
        )

    # ---------- The matrix restriction ----------
    restriction_cells = _cells_of(cells, kind="matrix_restriction")
    restriction, restriction_appendix = [], []
    for cell in restriction_cells:
        context = matrix_restriction.build_context(cell)
        restriction.append(
            matrix_restriction.matrix_restriction_table(
                cell, context, records[cell.cell_id]
            )
        )
        restriction_appendix.append(
            matrix_restriction.mean_statistic_table(cell, records[cell.cell_id])
        )
    if restriction:
        _write(
            "matrix_restriction",
            pd.concat(restriction, ignore_index=True),
            simcfg.TABLE_GROUP_MAIN,
        )
        _write(
            "matrix_restriction_appendix",
            pd.concat(restriction_appendix, ignore_index=True),
            simcfg.TABLE_GROUP_COMPLETE,
        )

    # ---------- Numerical telemetry and replication accounting ----------
    _write(
        "numerical_telemetry",
        _numerical_telemetry(cells, records),
        simcfg.TABLE_GROUP_CONSTRUCTION,
    )

    return written


def _design_sensitivity(cells: list, records: dict) -> pd.DataFrame:
    """Build the population-and-direction sensitivity table.

    The reference configuration reuses the leading replications of the
    corresponding main detection cell instead of drawing new ones,
    because its population, direction, path, departure size, dimensions,
    sample length, persistence, innovation design and replication seeds
    are identical to what a new cell would draw. The two additional
    configurations have their own cells.

    The three configurations vary the population value and the changing
    direction while holding everything else fixed. They are not draws
    defining an average power distribution, and the table says so on
    its own face.

    Parameters
    ----------
    cells : list
        The registry.
    records : dict
        Stored records by cell identity.

    Returns
    -------
    pandas.DataFrame
        One row per population configuration and changing block.
    """
    rows: list[dict] = []
    n_rep = simcfg.DESIGN_SENSITIVITY_REPS

    for block in simcfg.PLANTED_BLOCKS:
        reference = next(
            (
                cell
                for cell in cells
                if cell.part == "C"
                and cell.block == block
                and cell.path == simcfg.DESIGN_SENSITIVITY_PATH
                and cell.magnitude == simcfg.DESIGN_SENSITIVITY_MAGNITUDE
            ),
            None,
        )
        if reference is not None:
            reference_rows = detection.attribution_rows(
                reference, records[reference.cell_id], n_rep=n_rep
            )
            for row in reference_rows:
                row["replication_source"] = (
                    f"first {n_rep} replications reused from {reference.cell_id}"
                )
            rows.extend(reference_rows)

        for cell in cells:
            if cell.part != "S" or cell.block != block:
                continue
            additional = detection.attribution_rows(cell, records[cell.cell_id])
            for row in additional:
                row["replication_source"] = "new"
            rows.extend(additional)

    if not rows:
        return pd.DataFrame()

    wide = _wide_attribution(rows, simcfg.PRIMARY_LEVEL, MAIN_ATTRIBUTION_MEASURES)
    if wide.empty:
        return wide

    source = {row["cell_id"]: row["replication_source"] for row in rows}
    wide["replication_source"] = wide["cell_id"].map(source)
    wide["purpose"] = (
        "shows whether the attribution conclusion is unique to the reference "
        "population; not an average power distribution"
    )
    return wide


def _design_sensitivity_summary(full: pd.DataFrame) -> pd.DataFrame:
    """Minimum, median and maximum across configurations, by changing block.

    Parameters
    ----------
    full : pandas.DataFrame
        The full sensitivity table.

    Returns
    -------
    pandas.DataFrame
        One row per changing block and measure.
    """
    rows = []
    for block, group in full.groupby("changing_block"):
        for measure in MAIN_ATTRIBUTION_MEASURES + ("blockwise_signal",):
            values = group[measure].dropna()
            rows.append(
                {
                    "changing_block": block,
                    "measure": measure,
                    "n_configurations": int(len(group)),
                    "minimum": float(values.min()) if len(values) else np.nan,
                    "median": float(values.median()) if len(values) else np.nan,
                    "maximum": float(values.max()) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)




def _numerical_telemetry(cells: list, records: dict) -> pd.DataFrame:
    """Per-cell summary of the numerical-reliability record.

    Parameters
    ----------
    cells : list
        The registry.
    records : dict
        Stored records by cell identity.

    Returns
    -------
    pandas.DataFrame
        One row per cell that carries a numerical-reliability record and
        has at least one successful replication. A cell is absent when
        it stores no such record (the multiple-start check stores one
        row per start) and when every one of its replications failed,
        because there is then no fitted quantity to summarise. Absence
        here is not evidence that a cell ran cleanly. The replication
        accounting is what carries the attempted and failed counts of
        every cell.
    """
    rows = []
    for cell in cells:
        frame = records[cell.cell_id]
        # The multiple-start check stores one row per start instead of
        # the numerical-reliability record. Its ordinary start is the
        # same fit, on the same dataset, that the reference null cell
        # already reports.
        if frame.empty or "foc_norm_canonical" not in frame:
            continue
        successful = frame[frame["status"] == "ok"] if "status" in frame else frame
        if successful.empty:
            continue

        attempted = int(len(frame[frame["record_index"] == 0]))
        row = {
            "part": cell.part,
            "kind": cell.kind,
            "cell_id": cell.cell_id,
            "m": cell.m,
            "n": cell.n,
            "T": cell.T,
            "innovation": cell.innovation,
            "population_configuration": cell.configuration,
            "n_attempted": attempted,
            "n_completed": int(len(successful[successful["record_index"] == 0]))
            if "record_index" in successful
            else int(len(successful)),
            "n_failed": attempted
            - int(len(successful[successful["record_index"] == 0]))
            if "record_index" in successful
            else 0,
            "n_excluded": 0,
            "convergence_rate": float(successful["converged"].mean()),
            "median_iterations": float(successful["n_iterations"].median()),
            "median_initial_objective": float(successful["initial_objective"].median()),
            "median_final_objective": float(successful["final_objective"].median()),
            "median_foc_norm_canonical": float(
                successful["foc_norm_canonical"].median()
            ),
            "max_foc_norm_canonical": float(successful["foc_norm_canonical"].max()),
            "stability_contraction_rate": float(
                successful["stability_contracted"].mean()
            ),
            "mean_stability_contractions": float(
                successful["n_stability_contractions"].mean()
            ),
            "covariance_regularisation_rate": float(
                successful["covariance_regularised"].mean()
            ),
            "pseudoinverse_rate": float(successful["pseudoinverse_used"].mean()),
            "hessian_inversion_failure_rate": float(
                successful["hessian_inversion_failed"].mean()
            ),
            "score_covariance_inversion_failure_rate": float(
                successful["score_covariance_inversion_failed"].mean()
            ),
            "min_eigenvalue_H_min": float(successful["min_eigenvalue_H"].min()),
            "median_cond_H": float(successful["cond_H"].median()),
            "max_cond_H": float(successful["cond_H"].max()),
            "min_eigenvalue_Omega_min": float(
                successful["min_eigenvalue_Omega"].min()
            ),
            "median_cond_Omega": float(successful["cond_Omega"].median()),
            "max_cond_Omega": float(successful["cond_Omega"].max()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def replication_accounting(cells: list, outcomes: list) -> pd.DataFrame:
    """Completed, failed and excluded replication counts of every cell.

    A failed replication is recorded and not replaced, so attempted
    always equals completed plus failed. No replication is excluded
    anywhere in this design. The column exists so a reader can see it
    is zero, not assume it.

    Parameters
    ----------
    cells : list
        The registry.
    outcomes : list
        The per-cell outcomes of the run.

    Returns
    -------
    pandas.DataFrame
        One row per cell.
    """
    return pd.DataFrame(
        [
            {
                "part": cell.part,
                "kind": cell.kind,
                "cell_id": cell.cell_id,
                "n_requested": cell.n_rep,
                "n_attempted": outcome.attempted,
                "n_completed": outcome.successful,
                "n_failed": outcome.failed,
                "n_excluded": 0,
                "seconds": outcome.seconds,
                "reused_chunks": outcome.reused_chunks,
                "source": outcome.source,
            }
            for cell, outcome in zip(cells, outcomes)
        ]
    )


# =========================================================
# The run
# =========================================================
def run_study(
    mode: str,
    profile: str = simcfg.DEFAULT_PROFILE,
    workers: int = simcfg.MAX_WORKERS,
    output_dir: str | Path = paths_cfg.OUTPUT_DIR,
    run_id: str | None = None,
    parts: tuple[str, ...] | None = None,
) -> dict:
    """Run the whole study in one mode and return its summary.

    Parameters
    ----------
    mode : {"smoke", "preview", "production"}
        Which run mode. Smoke and preview run the same cells with capped
        replication and bootstrap draw counts.
    profile : str, optional
        Runtime profile.
    workers : int, optional
        Process workers, at most the configured maximum.
    output_dir : str or pathlib.Path, optional
        Project output root.
    run_id : str or None, optional
        Run identifier; derived from the mode and the clock when omitted.
    parts : tuple of str or None, optional
        Restrict execution to these design branches; the full registry is
        still built.

    Returns
    -------
    dict
        The run summary.

    Raises
    ------
    ValueError
        If the worker count exceeds the configured maximum, or a
        finished run of the same identifier already exists.
    SeedStreamError
        If two seed identities share a stream.
    """
    if workers > simcfg.MAX_WORKERS:
        raise ValueError(
            f"workers={workers} exceeds the configured maximum {simcfg.MAX_WORKERS}."
        )

    runner.enforce_thread_limits()

    full_registry = registry.build_registry(profile)
    registry_summary = registry.describe_registry(full_registry)

    # A capped run executes the same cells with capped counts, which
    # gives them their own identities, configuration hashes and reuse
    # keys.
    if mode == "smoke":
        cells = registry.capped_registry(
            full_registry, simcfg.SMOKE_REPS, simcfg.SMOKE_MAX_INNER
        )
        registry_summary["caps"] = {
            "outer": simcfg.SMOKE_REPS,
            "inner": simcfg.SMOKE_MAX_INNER,
        }
    elif mode == "preview":
        cells = registry.capped_registry(
            full_registry, simcfg.PREVIEW_MAX_OUTER, simcfg.PREVIEW_MAX_INNER
        )
        registry_summary["caps"] = {
            "outer": simcfg.PREVIEW_MAX_OUTER,
            "inner": simcfg.PREVIEW_MAX_INNER,
        }
    elif mode == "production":
        cells = full_registry
    else:
        raise ValueError(
            f"unknown run mode={mode!r}; expected smoke, preview or production."
        )

    run_id = run_id or simio.make_run_id(mode)
    if simio.existing_finished_run(output_dir, run_id):
        raise ValueError(f"run {run_id} is already finished; refusing to overwrite it.")

    paths = simio.run_paths(output_dir, run_id, mode)
    started = datetime.now(timezone.utc).isoformat()
    simio.write_manifest(paths, run_id, mode, profile, workers, cells, started)

    print(f"[study] registry: {registry_summary['n_cells']} cells", flush=True)
    audit = design_audit.audit(cells, output_dir)
    print("[study] design checks\n" + design_audit.summarise(audit), flush=True)

    seeds_frame = design_audit.seed_registry(cells, audit)
    disjoint = design_audit.seed_streams_are_disjoint(seeds_frame)
    if not disjoint["streams_disjoint"]:
        raise SeedStreamError(
            f"two seed identities share a stream: {disjoint}"
        )
    simio.save_table(
        paths, "seed_registry", seeds_frame, simcfg.TABLE_GROUP_CONSTRUCTION
    )

    laws = runner.warm_limit_laws(
        LIMIT_LAW_CACHE, runner.required_limit_law_dimensions(cells)
    )
    simio.save_telemetry(paths, "limit_laws.csv", pd.DataFrame(laws))

    selected = [c for c in cells if parts is None or c.part in parts]
    outcomes = []
    for index, cell in enumerate(selected, start=1):
        n_rep = runner.replications_for_mode(cell, mode)
        print(
            f"[study] {index}/{len(selected)} part {cell.part} {cell.kind} "
            f"({cell.m},{cell.n},{cell.T}) {cell.configuration} n_rep={n_rep}",
            flush=True,
        )
        outcome = runner.run_cell(
            cell, paths, n_rep, workers, output_dir=output_dir, run_id=run_id
        )
        outcomes.append(outcome)
        print(
            f"         attempted {outcome.attempted} completed {outcome.successful} "
            f"failed {outcome.failed} in {outcome.seconds:.1f}s "
            f"(source {outcome.source}, {outcome.reused_chunks} chunks resumed)",
            flush=True,
        )

    accounting = replication_accounting(selected, outcomes)
    simio.save_telemetry(
        paths, "replications.parquet", runner.replication_telemetry(paths, selected)
    )
    simio.save_telemetry(paths, "failures.csv", runner.failure_table(paths, selected))

    tables = build_tables(selected, paths, audit)
    simio.save_table(
        paths, "replication_accounting", accounting, simcfg.TABLE_GROUP_CONSTRUCTION
    )
    drawn = figures.build_influence_outputs(selected, paths)
    index = simio.write_output_index(paths)

    simio.write_manifest(
        paths,
        run_id,
        mode,
        profile,
        workers,
        cells,
        started,
        finished=datetime.now(timezone.utc).isoformat(),
        reuse={o.cell_id: o.source for o in outcomes},
        seed_registry=seeds_frame,
        extra={
            "registry_summary": registry_summary,
            "output_index": index.to_dict("records"),
            "seed_streams": disjoint,
        },
    )

    return {
        "run_id": run_id,
        "paths": paths,
        "registry_summary": registry_summary,
        "accounting": accounting,
        "tables": {name: len(table) for name, table in tables.items()},
        # build_influence_outputs returns the drawn figures and the
        # signature agreement table in one mapping. Only the entries
        # that have a rendered figure are reported as figures.
        "figures": sorted(
            name for name in drawn if (paths.figures / f"{name}.png").exists()
        ),
        "figure_tables": sorted(
            name for name in drawn if not (paths.figures / f"{name}.png").exists()
        ),
        "output_index": index,
    }

"""The typed cell registry.

One registry describes the whole study. Every cell carries its complete
design coordinates, so two cells that differ in any respect the design
names are different cells with different identities, different seed
streams and different checkpoints.

The registry is also where the reuse rules live. Two kinds of reuse
appear.

*Dataset sharing inside a run.* Cells that must see the same data
instead of drawing new ones share a ``dataset_key`` while keeping
distinct cell identities: the wild-bootstrap comparison and the
multiple-start check both run on the reference null datasets of the
calibration cell they are paired with.

*Output reuse across runs.* Each cell carries a ``reuse_key`` over the
substantive determinants of its records (the data-generating process,
the estimator controls, the statistics, the seeds and the counts) and
the record schema version. A stored chunk may be carried into a new run
only when that key matches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

import mardynamics.experiments.sim_config as simcfg

# The five main branches of the design. The "A" branch has no simulated
# cell: its outputs are the constructed design and its verification,
# which the design audit produces before any replication runs.
PARTS = {
    "A": "Simulation design and protocol",
    "B": "Finite-sample calibration and numerical reliability",
    "C": "Detection, mode-specific attribution and influence paths",
    "D": "The value and limits of the matrix restriction",
    "S": "Appendix: population-and-direction sensitivity",
}


@dataclass(frozen=True)
class Cell:
    """One simulation cell, with every design coordinate that identifies it.

    Attributes
    ----------
    part : str
        Which branch of the design, ``"A"`` to ``"D"`` or ``"S"``.
    kind : str
        Which routine executes the cell.
    m, n, T : int
        Panel dimensions and retained sample length.
    persistence : float
        The designed joint spectral radius of the population.
    configuration : str
        Which of the three population configurations.
    innovation : str
        Innovation design name.
    n_rep : int
        Outer replications.
    path : str or None
        Scalar path of a designed change, if any.
    block : str or None
        Changing canonical block, if any.
    magnitude : str or None
        Departure size, if any.
    n_resample : int
        Inner bootstrap draws; zero when the cell does not resample.
    dataset_key : tuple
        Identity of the dataset stream. Cells that must see the same
        data share it. Every other cell has its own.
    statistics : tuple of str
        Which quantities the cell records.
    extra : dict
        Cell-specific design values (condition, variance break, start count).
    """

    part: str
    kind: str
    m: int
    n: int
    T: int
    persistence: float
    configuration: str
    innovation: str
    n_rep: int
    path: str | None = None
    block: str | None = None
    magnitude: str | None = None
    n_resample: int = 0
    dataset_key: tuple = ()
    statistics: tuple = ()
    extra: dict = field(default_factory=dict)

    @property
    def design_coordinates(self) -> dict:
        """Every coordinate that distinguishes this cell from another."""
        data = asdict(self)
        data.pop("statistics")
        data["dataset_key"] = list(self.dataset_key)
        return data

    @property
    def substantive_determinants(self) -> dict:
        """Everything a stored record of this cell actually depends on.

        This is the DGP, the estimator, the statistic, the algorithm,
        the seeds and the counts being unchanged. The cell's part
        label, its human-readable identity and any purely
        presentational coordinate are absent: renaming a part does not
        make a previously computed replication a different number.
        """
        return {
            "dataset_key": list(self.dataset_key),
            "kind": self.kind,
            "m": self.m,
            "n": self.n,
            "T": self.T,
            "persistence": self.persistence,
            "configuration": self.configuration,
            "innovation": self.innovation,
            "path": self.path,
            "block": self.block,
            "magnitude": self.magnitude,
            "n_rep": self.n_rep,
            "n_resample": self.n_resample,
            "statistics": sorted(self.statistics),
            "extra": {key: str(value) for key, value in sorted(self.extra.items())},
            "schema_version": simcfg.OUTPUT_SCHEMA_VERSION,
            "master_seed": simcfg.SIM_MASTER_SEED,
            "sigma": simcfg.SIGMA,
            "burn_in": simcfg.BURN_IN,
            "levels": list(simcfg.NOMINAL_LEVELS),
            "static_als": [
                simcfg.STATIC_ALS_MAX_ITER,
                simcfg.STATIC_ALS_TOL,
                simcfg.STATIC_ALS_STOPPING_RULE,
            ],
            "stability": [simcfg.ENFORCE_STABILITY, simcfg.MAX_RADIUS],
            "local_scalars": dict(simcfg.LOCAL_MAGNITUDE_SCALARS),
            "large_magnitude": simcfg.LARGE_MAGNITUDE,
        }

    @property
    def reuse_key(self) -> str:
        """Hash of the substantive determinants: what output reuse is keyed on.

        Two cells with the same reuse key produce bit-identical records, so a
        stored chunk of one is a valid stored chunk of the other. Two cells with
        different reuse keys do not share a checkpoint, whatever their names.
        """
        blob = json.dumps(
            self.substantive_determinants, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    @property
    def config_hash(self) -> str:
        """Short hash of the full design coordinates and the run-wide constants.

        A resumed run reads only checkpoints whose hash matches, so any change
        to a cell's design or to a constant it depends on invalidates its stored
        chunks instead of silently mixing two designs.
        """
        payload = {
            "design": self.design_coordinates,
            "statistics": list(self.statistics),
            "substantive": self.substantive_determinants,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    @property
    def cell_id(self) -> str:
        """Readable identity: the design coordinates plus the configuration hash."""
        parts = [
            self.part,
            self.kind,
            f"m{self.m}n{self.n}T{self.T}",
            f"rho{self.persistence:g}",
            self.configuration,
            self.innovation,
        ]
        for label, value in (
            ("path", self.path),
            ("block", self.block),
            ("mag", self.magnitude),
        ):
            if value is not None:
                parts.append(f"{label}-{value}")
        for key in sorted(self.extra):
            parts.append(f"{key}-{self.extra[key]}")
        parts.append(f"rep{self.n_rep}")
        if self.n_resample:
            parts.append(f"res{self.n_resample}")
        parts.append(self.config_hash)
        return "_".join(str(part) for part in parts)


def _dataset_key(*parts) -> tuple:
    """Build a dataset-stream identity.

    Parameters
    ----------
    *parts
        Components of the stream identity.

    Returns
    -------
    tuple
        The key.
    """
    return tuple(parts)


def _calibration_dataset_key(m: int, n: int, T: int, innovation: str,
                             persistence: float) -> tuple:
    """Dataset identity of one null calibration cell.

    The calibration cells, and the wild-bootstrap comparison paired
    with one of them, must draw the same data.

    Parameters
    ----------
    m, n, T : int
        Panel dimensions and sample length.
    innovation : str
        Innovation design.
    persistence : float
        Designed joint spectral radius.

    Returns
    -------
    tuple
        The dataset key.
    """
    return _dataset_key("calibration", m, n, T, innovation, persistence)


def var_feasibility_table() -> list[dict]:
    """Analytic rank feasibility of the unrestricted statistic over the grid.

    The unrestricted statistic needs an invertible outer-product covariance of
    ``K = m^2 n^2`` parameters built from ``T - 1`` scores that sum to zero, so
    it exists only where ``T - 2 >= K``. This table is analytic: nothing is
    simulated to produce it.

    Returns
    -------
    list of dict
        One row per grid point with the two dimensions and its feasibility.
    """
    rows = []
    for m, n in simcfg.DIM_GRID:
        for T in simcfg.T_GRID:
            k = m * m + n * n - 1
            K = (m * n) ** 2
            rows.append(
                {
                    "m": m,
                    "n": n,
                    "T": T,
                    "k": k,
                    "K": K,
                    "scores": T - 1,
                    "usable_scores": T - 2,
                    "var_feasible": bool(T - 2 >= K),
                }
            )
    return rows


# =========================================================
# Calibration, bootstrap, recovery, multiple starts
# =========================================================
def _part_b_cells() -> list[Cell]:
    """Build the calibration, bootstrap and multiple-start cells.

    Every null cell simulates a stable population, so its replications
    are the stable-DGP replications recovery is measured on. No
    separate recovery cell exists: adding one would resimulate a design
    the study already runs.

    Returns
    -------
    list of Cell
        The part's cells.
    """
    cells: list[Cell] = []
    reference_dataset = _calibration_dataset_key(
        simcfg.BASE_M,
        simcfg.BASE_N,
        simcfg.BASE_T,
        "isotropic",
        simcfg.BASE_PERSISTENCE,
    )

    for m, n in simcfg.DIM_GRID:
        for T in simcfg.T_GRID:
            statistics = ["nyblom_global", "nyblom_blocks", "recovery", "telemetry"]
            if (m, n, T) == (simcfg.VAR_M, simcfg.VAR_N, simcfg.VAR_T):
                statistics.append("nyblom_var")
            cells.append(
                Cell(
                    part="B",
                    kind="null_calibration",
                    m=m,
                    n=n,
                    T=T,
                    persistence=simcfg.BASE_PERSISTENCE,
                    configuration=simcfg.REFERENCE_CONFIGURATION,
                    innovation="isotropic",
                    n_rep=simcfg.CALIBRATION_REPS,
                    dataset_key=_calibration_dataset_key(
                        m, n, T, "isotropic", simcfg.BASE_PERSISTENCE
                    ),
                    statistics=tuple(statistics),
                )
            )

    for design in simcfg.EXTRA_INNOVATION_DESIGNS:
        statistics = ["nyblom_global", "nyblom_blocks", "recovery", "telemetry"]
        if design == "garch":
            statistics.append("garch_strength")
        cells.append(
            Cell(
                part="B",
                kind="null_calibration",
                m=simcfg.BASE_M,
                n=simcfg.BASE_N,
                T=simcfg.BASE_T,
                persistence=simcfg.BASE_PERSISTENCE,
                configuration=simcfg.REFERENCE_CONFIGURATION,
                innovation=design,
                n_rep=simcfg.CALIBRATION_REPS,
                dataset_key=_calibration_dataset_key(
                    simcfg.BASE_M,
                    simcfg.BASE_N,
                    simcfg.BASE_T,
                    design,
                    simcfg.BASE_PERSISTENCE,
                ),
                statistics=tuple(statistics),
            )
        )

    cells.append(
        Cell(
            part="B",
            kind="null_calibration",
            m=simcfg.BASE_M,
            n=simcfg.BASE_N,
            T=simcfg.BASE_T,
            persistence=simcfg.HIGH_PERSISTENCE,
            configuration=simcfg.REFERENCE_CONFIGURATION,
            innovation="isotropic",
            n_rep=simcfg.CALIBRATION_REPS,
            dataset_key=_calibration_dataset_key(
                simcfg.BASE_M,
                simcfg.BASE_N,
                simcfg.BASE_T,
                "isotropic",
                simcfg.HIGH_PERSISTENCE,
            ),
            statistics=("nyblom_global", "nyblom_blocks", "recovery", "telemetry"),
        )
    )

    cells.append(
        Cell(
            part="B",
            kind="wild_bootstrap",
            m=simcfg.BASE_M,
            n=simcfg.BASE_N,
            T=simcfg.BASE_T,
            persistence=simcfg.BASE_PERSISTENCE,
            configuration=simcfg.REFERENCE_CONFIGURATION,
            innovation="isotropic",
            n_rep=simcfg.BOOTSTRAP_OUTER_REPS,
            n_resample=simcfg.BOOTSTRAP_DRAWS,
            dataset_key=reference_dataset,
            statistics=("nyblom_global", "bootstrap_pvalue", "telemetry"),
        )
    )

    cells.append(
        Cell(
            part="B",
            kind="multiple_start",
            m=simcfg.BASE_M,
            n=simcfg.BASE_N,
            T=simcfg.BASE_T,
            persistence=simcfg.BASE_PERSISTENCE,
            configuration=simcfg.REFERENCE_CONFIGURATION,
            innovation="isotropic",
            n_rep=simcfg.MULTIPLE_START_REPS,
            dataset_key=reference_dataset,
            statistics=("multiple_start",),
            extra={
                "n_perturbed_starts": simcfg.MULTIPLE_START_PERTURBED,
                "perturbation": simcfg.MULTIPLE_START_PERTURBATION,
            },
        )
    )

    return cells


# =========================================================
# Detection and conditional attribution
# =========================================================
def _detection_dataset_key(configuration, block, path, magnitude) -> tuple:
    """Dataset identity of one detection cell.

    The reference configuration's key omits the configuration name, so
    the sensitivity branch can reuse its first replications instead of
    drawing new ones.

    Parameters
    ----------
    configuration : str
        Which population configuration.
    block, path, magnitude : str
        The design coordinates of the change.

    Returns
    -------
    tuple
        The dataset key.
    """
    base = _dataset_key(
        "power", simcfg.BASE_M, simcfg.BASE_N, simcfg.BASE_T, block, path, magnitude
    )
    if configuration == simcfg.REFERENCE_CONFIGURATION:
        return base
    return base + (configuration,)


def _part_c_cells() -> list[Cell]:
    """Build the full detection and attribution factorial.

    Returns
    -------
    list of Cell
        The part's cells.
    """
    return [
        Cell(
            part="C",
            kind="detection",
            m=simcfg.BASE_M,
            n=simcfg.BASE_N,
            T=simcfg.BASE_T,
            persistence=simcfg.BASE_PERSISTENCE,
            configuration=simcfg.REFERENCE_CONFIGURATION,
            innovation="isotropic",
            n_rep=simcfg.POWER_REPS,
            path=path,
            block=block,
            magnitude=magnitude,
            dataset_key=_detection_dataset_key(
                simcfg.REFERENCE_CONFIGURATION, block, path, magnitude
            ),
            statistics=(
                "nyblom_global",
                "nyblom_blocks",
                "gated_attribution",
                "telemetry",
            )
            + (
                ("influence_path",)
                if (block, path, magnitude) in simcfg.INFLUENCE_FIGURE_CELLS
                else ()
            ),
        )
        for block in simcfg.PLANTED_BLOCKS
        for path in simcfg.PATHS
        for magnitude in simcfg.MAGNITUDES
    ]


# =========================================================
# The matrix restriction
# =========================================================
def _part_d_cells() -> list[Cell]:
    """Build the matrix-restriction cells.

    Returns
    -------
    list of Cell
        The part's cells.
    """
    conditions = {
        "separable_isotropic": ("isotropic", "row"),
        "separable_correlated": ("correlated", "row"),
        "unrestricted": ("isotropic", None),
    }

    return [
        Cell(
            part="D",
            kind="matrix_restriction",
            m=simcfg.VAR_M,
            n=simcfg.VAR_N,
            T=simcfg.VAR_T,
            persistence=simcfg.BASE_PERSISTENCE,
            configuration=simcfg.REFERENCE_CONFIGURATION,
            innovation=innovation,
            n_rep=simcfg.MATRIX_VALUE_REPS,
            path=simcfg.MATRIX_VALUE_PATH,
            block=block,
            magnitude=magnitude,
            dataset_key=_dataset_key(
                "matrix_value",
                simcfg.VAR_M,
                simcfg.VAR_N,
                simcfg.VAR_T,
                condition,
                magnitude,
            ),
            statistics=("nyblom_global", "nyblom_var", "telemetry"),
            extra={"condition": condition},
        )
        for condition, (innovation, block) in conditions.items()
        for magnitude in simcfg.MAGNITUDES
    ]


# =========================================================
# Sensitivity: populations and directions
# =========================================================
def _sensitivity_cells() -> list[Cell]:
    """Build the additional-population cells of the sensitivity branch.

    Only the two additional configurations get cells. The reference row
    of the sensitivity table reuses the first 500 replications of the
    corresponding main detection cell, whose population, direction,
    path, magnitude, dimensions, sample length, persistence, innovation
    design and replication seeds are identical to what a new reference
    cell would draw.

    Returns
    -------
    list of Cell
        The sensitivity cells.
    """
    additional = [
        c
        for c in simcfg.POPULATION_CONFIGURATIONS
        if c != simcfg.REFERENCE_CONFIGURATION
    ]

    return [
        Cell(
            part="S",
            kind="detection",
            m=simcfg.BASE_M,
            n=simcfg.BASE_N,
            T=simcfg.BASE_T,
            persistence=simcfg.BASE_PERSISTENCE,
            configuration=configuration,
            innovation="isotropic",
            n_rep=simcfg.DESIGN_SENSITIVITY_REPS,
            path=simcfg.DESIGN_SENSITIVITY_PATH,
            block=block,
            magnitude=simcfg.DESIGN_SENSITIVITY_MAGNITUDE,
            dataset_key=_detection_dataset_key(
                configuration,
                block,
                simcfg.DESIGN_SENSITIVITY_PATH,
                simcfg.DESIGN_SENSITIVITY_MAGNITUDE,
            ),
            statistics=(
                "nyblom_global",
                "nyblom_blocks",
                "gated_attribution",
                "telemetry",
            ),
        )
        for configuration in additional
        for block in simcfg.PLANTED_BLOCKS
    ]


# =========================================================
# Assembly
# =========================================================
def build_registry(profile: str = simcfg.DEFAULT_PROFILE) -> list[Cell]:
    """Build the complete typed registry.

    Parameters
    ----------
    profile : str, optional
        Runtime profile. Only one is configured.

    Returns
    -------
    list of Cell
        Every cell of the study.

    Raises
    ------
    ValueError
        For an unknown profile.
    """
    if profile not in simcfg.PROFILES:
        raise ValueError(
            f"unknown profile={profile!r}; expected one of {simcfg.PROFILES}"
        )

    return (
        _part_b_cells()
        + _part_c_cells()
        + _part_d_cells()
        + _sensitivity_cells()
    )


def capped_registry(cells: list, max_outer: int, max_inner: int) -> list[Cell]:
    """Cap a registry to smaller counts, leaving every design coordinate.

    A capped run executes the same cells as a production run and
    differs only in how many outer replications and inner bootstrap
    draws each of them gets. The counts are part of a cell's identity
    and of its reuse key, so a cell that is actually capped carries a
    different identifier and a different hash from its production
    counterpart, and its checkpoint cannot be resumed into a production
    run.

    A cap does not bind on a cell whose configured count is already at
    or below it, and such a cell is then not a capped cell at all:
    ``min`` returns the configured count, every design coordinate is
    the production one, and the identity, configuration hash and reuse
    key are identical to production's by construction. At the
    configured counts this is one cell, the multiple-start check, whose
    ``MULTIPLE_START_REPS = 100`` sits below ``PREVIEW_MAX_OUTER =
    200``. The smoke cap of two binds on every cell and collides with
    nothing.

    That identity is not a defect and nothing is mislabelled by it.
    Because the cell is the production cell in every coordinate, its
    dataset stream, its seeds and therefore its records are the ones a
    production run would itself draw, so a production run that reuses
    such a chunk obtains bit-identical numbers and records the reuse in
    its manifest. What must not be inferred from a reuse entry is that
    the chunk was produced under a production run: for this cell it may
    have been produced under a preview.

    Parameters
    ----------
    cells : list of Cell
        The full registry, already validated.
    max_outer : int
        Cap on outer replications.
    max_inner : int
        Cap on inner bootstrap draws.

    Returns
    -------
    list of Cell
        The capped registry, in the same order.
    """
    capped = []
    for cell in cells:
        coordinates = cell.design_coordinates
        coordinates["dataset_key"] = cell.dataset_key
        coordinates["n_rep"] = min(cell.n_rep, int(max_outer))
        coordinates["n_resample"] = (
            min(cell.n_resample, int(max_inner)) if cell.n_resample else 0
        )
        capped.append(Cell(**coordinates, statistics=cell.statistics))
    return capped


# =========================================================
# The registry summary
# =========================================================
def describe_registry(cells: list[Cell]) -> dict:
    """Describe a registry by part.

    Parameters
    ----------
    cells : list of Cell
        The registry.

    Returns
    -------
    dict
        Cell counts, replication and bootstrap-draw totals by part.
    """
    summary: dict = {
        "n_cells": len(cells),
        "by_part": {},
        "parts": dict(PARTS),
    }
    for cell in cells:
        entry = summary["by_part"].setdefault(
            cell.part, {"cells": 0, "outer_replications": 0, "inner_draws": 0}
        )
        entry["cells"] += 1
        entry["outer_replications"] += cell.n_rep
        entry["inner_draws"] += cell.n_rep * cell.n_resample

    return summary
